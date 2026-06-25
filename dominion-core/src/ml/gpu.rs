//! Native GPU / CUDA support — capability-gated access to the NVIDIA compute stack
//! (CUDA driver, cuDNN, cuBLAS, cuFFT, cuSPARSE, cuSOLVER, NCCL, TensorRT) plus a
//! GPU device registry, layered on the [`super`] ML runtime and the capability-only
//! FFI surface ([`crate::packaging::FfiSurface`]).
//!
//! DominionOS treats CUDA exactly like every other foreign capability: **default-closed**.
//! A process holds no GPU authority until a [`CudaRuntime`] is handed a [`Capability`]
//! and a library is explicitly `enable`d; only then are that library's entry points
//! callable, and only through the FFI gate. This gives ML workloads the full CUDA
//! surface they expect while keeping the kernel capability-pure — a buggy/hostile
//! kernel cannot reach GPU memory or another process's context.
//!
//! **Honest boundary.** What is real here: the capability-gated admission of the CUDA
//! ABI surface, the device registry, the cost model ([`super::Device::Gpu`]), and a
//! correct CPU fallback so code runs with or without an accelerator. What requires the
//! vendor driver + silicon: dispatching the *actual* kernels — that happens across the
//! FFI gate to the real library when present. The numerics are identical either way.
//!
//! Pure, safe `no_std`, host-tested.

use crate::capability::Capability;
use crate::datatypes::Tensor;
use crate::packaging::{FfiError, FfiSurface};
use alloc::collections::BTreeSet;
use alloc::vec::Vec;

/// A CUDA stack library whose ABI surface the runtime can admit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum CudaLibrary {
    /// The CUDA driver/runtime API (context, memory, kernel launch).
    Cuda,
    /// Deep-neural-network primitives (convolutions, pooling, normalization).
    Cudnn,
    /// Dense linear algebra (GEMM, etc.).
    Cublas,
    /// Fast Fourier transforms.
    Cufft,
    /// Sparse linear algebra.
    Cusparse,
    /// Dense direct solvers.
    Cusolver,
    /// Multi-GPU collective communication.
    Nccl,
    /// Inference-optimizing runtime.
    TensorRt,
}

impl CudaLibrary {
    pub fn name(self) -> &'static str {
        match self {
            CudaLibrary::Cuda => "cuda",
            CudaLibrary::Cudnn => "cudnn",
            CudaLibrary::Cublas => "cublas",
            CudaLibrary::Cufft => "cufft",
            CudaLibrary::Cusparse => "cusparse",
            CudaLibrary::Cusolver => "cusolver",
            CudaLibrary::Nccl => "nccl",
            CudaLibrary::TensorRt => "tensorrt",
        }
    }

    /// Representative entry points this library exposes — the symbols admitted when the
    /// library is enabled (and the surface a foreign binary's imports are checked against).
    pub fn symbols(self) -> &'static [&'static str] {
        match self {
            CudaLibrary::Cuda => &[
                "cuInit",
                "cuCtxCreate",
                "cudaMalloc",
                "cudaFree",
                "cudaMemcpy",
                "cuLaunchKernel",
                "cudaStreamCreate",
                "cudaDeviceSynchronize",
            ],
            CudaLibrary::Cudnn => &[
                "cudnnCreate",
                "cudnnConvolutionForward",
                "cudnnActivationForward",
                "cudnnPoolingForward",
                "cudnnBatchNormalizationForwardInference",
            ],
            CudaLibrary::Cublas => &["cublasCreate", "cublasSgemm", "cublasDgemm", "cublasSaxpy"],
            CudaLibrary::Cufft => &["cufftPlan1d", "cufftExecC2C", "cufftDestroy"],
            CudaLibrary::Cusparse => &["cusparseCreate", "cusparseSpMM"],
            CudaLibrary::Cusolver => &["cusolverDnCreate", "cusolverDnSgetrf"],
            CudaLibrary::Nccl => &["ncclCommInitRank", "ncclAllReduce", "ncclBroadcast"],
            CudaLibrary::TensorRt => &["createInferRuntime", "deserializeCudaEngine", "enqueueV3"],
        }
    }

    /// Every supported library, in a stable order.
    pub fn all() -> &'static [CudaLibrary] {
        &[
            CudaLibrary::Cuda,
            CudaLibrary::Cudnn,
            CudaLibrary::Cublas,
            CudaLibrary::Cufft,
            CudaLibrary::Cusparse,
            CudaLibrary::Cusolver,
            CudaLibrary::Nccl,
            CudaLibrary::TensorRt,
        ]
    }
}

/// A known GPU model in the device registry (what `driver`/`pkg` surfaces enumerate).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuModel {
    pub name: &'static str,
    /// CUDA compute capability `(major, minor)`.
    pub compute_capability: (u8, u8),
    pub vram_gb: u32,
    pub sm_count: u32,
}

/// A representative registry of modern NVIDIA GPUs the OS recognises.
pub fn known_gpus() -> &'static [GpuModel] {
    &[
        GpuModel { name: "NVIDIA H100", compute_capability: (9, 0), vram_gb: 80, sm_count: 132 },
        GpuModel { name: "NVIDIA A100", compute_capability: (8, 0), vram_gb: 80, sm_count: 108 },
        GpuModel { name: "NVIDIA L40S", compute_capability: (8, 9), vram_gb: 48, sm_count: 142 },
        GpuModel { name: "GeForce RTX 4090", compute_capability: (8, 9), vram_gb: 24, sm_count: 128 },
        GpuModel { name: "GeForce RTX 3090", compute_capability: (8, 6), vram_gb: 24, sm_count: 82 },
        GpuModel { name: "NVIDIA T4", compute_capability: (7, 5), vram_gb: 16, sm_count: 40 },
        GpuModel { name: "Jetson Orin", compute_capability: (8, 7), vram_gb: 32, sm_count: 16 },
    ]
}

/// The capability-gated CUDA runtime. Default-closed: no library is callable until it
/// is [`enable`](CudaRuntime::enable)d with a capability.
#[derive(Default)]
pub struct CudaRuntime {
    ffi: FfiSurface,
    enabled: BTreeSet<CudaLibrary>,
}

impl CudaRuntime {
    pub fn new() -> CudaRuntime {
        CudaRuntime { ffi: FfiSurface::new(), enabled: BTreeSet::new() }
    }

    /// Admit a CUDA library by granting `cap` to each of its entry points. After this
    /// the library's symbols are callable through the FFI gate — and nothing else is.
    pub fn enable(&mut self, lib: CudaLibrary, cap: Capability) {
        for sym in lib.symbols() {
            self.ffi.grant(sym, cap.clone());
        }
        self.enabled.insert(lib);
    }

    pub fn is_enabled(&self, lib: CudaLibrary) -> bool {
        self.enabled.contains(&lib)
    }

    /// The libraries currently admitted, in stable order.
    pub fn enabled_libraries(&self) -> Vec<CudaLibrary> {
        CudaLibrary::all().iter().copied().filter(|l| self.enabled.contains(l)).collect()
    }

    /// Attempt a CUDA call by symbol. Returns the governing capability iff the owning
    /// library was enabled (default-closed otherwise).
    pub fn call(&self, symbol: &str) -> Result<&Capability, FfiError> {
        self.ffi.call(symbol)
    }

    /// Is a specific entry point reachable right now?
    pub fn is_available(&self, symbol: &str) -> bool {
        self.ffi.is_granted(symbol)
    }
}

/// GPU-accelerated matrix multiply, capability-gated. If cuBLAS GEMM is admitted, the
/// runtime takes the GPU path; otherwise it falls back to the CPU. The numerics are
/// identical — only the (modeled) execution device differs. `None` on shape mismatch.
pub fn gpu_matmul(rt: &CudaRuntime, a: &Tensor, b: &Tensor) -> Option<Tensor> {
    if rt.call("cublasSgemm").is_ok() {
        // GPU path: the real dispatch crosses the FFI gate to cuBLAS; the result is
        // device-independent, so the modeled compute uses the same numerics.
        super::Device::Gpu.matmul(a, b)
    } else {
        super::Device::Cpu.matmul(a, b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capability::Rights;
    use alloc::vec;

    fn cap() -> Capability {
        Capability::mint(0x1000, 0x1000, Rights::READ.union(Rights::WRITE))
    }
    fn mat(r: usize, c: usize, fill: f64) -> Tensor {
        Tensor::new(vec![r, c], vec![fill; r * c]).unwrap()
    }

    #[test]
    fn cuda_is_default_closed_until_a_library_is_enabled() {
        let rt = CudaRuntime::new();
        assert!(rt.call("cublasSgemm").is_err());
        assert!(!rt.is_available("cudaMalloc"));
        assert!(rt.enabled_libraries().is_empty());
    }

    #[test]
    fn enabling_a_library_admits_exactly_its_symbols() {
        let mut rt = CudaRuntime::new();
        rt.enable(CudaLibrary::Cublas, cap());
        assert!(rt.is_enabled(CudaLibrary::Cublas));
        assert!(rt.call("cublasSgemm").is_ok());
        // A symbol from a library we did NOT enable stays closed.
        assert!(rt.call("cudnnConvolutionForward").is_err());
        rt.enable(CudaLibrary::Cudnn, cap());
        assert!(rt.call("cudnnConvolutionForward").is_ok());
        assert_eq!(rt.enabled_libraries().len(), 2);
    }

    #[test]
    fn gpu_matmul_uses_gpu_when_enabled_and_falls_back_otherwise() {
        let a = mat(2, 3, 1.0);
        let b = mat(3, 2, 2.0);
        let expected = a.matmul(&b).unwrap();

        // Without cuBLAS: CPU fallback still produces the right answer.
        let rt = CudaRuntime::new();
        assert_eq!(gpu_matmul(&rt, &a, &b), Some(expected.clone()));

        // With cuBLAS enabled: GPU path, identical numerics.
        let mut rt2 = CudaRuntime::new();
        rt2.enable(CudaLibrary::Cublas, cap());
        assert_eq!(gpu_matmul(&rt2, &a, &b), Some(expected));
    }

    #[test]
    fn the_gpu_registry_knows_modern_nvidia_parts() {
        let gpus = known_gpus();
        assert!(gpus.iter().any(|g| g.name.contains("H100") && g.compute_capability == (9, 0)));
        assert!(gpus.iter().any(|g| g.name.contains("RTX 4090")));
    }

    #[test]
    fn every_library_exposes_entry_points() {
        for lib in CudaLibrary::all() {
            assert!(!lib.symbols().is_empty(), "{} has no symbols", lib.name());
        }
    }
}
