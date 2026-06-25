#!/usr/bin/env bash
# Linux comparison harness for the DominionOS benchmark suite.
#
# Emits the SAME schema as the in-guest battery (kernel/src/bench.rs):
#     BENCH <category> key=value key=value ...
# so rows match up against ../../bench-results.json. Writes linux-results.json.
#
# This is the "later" half: a runnable skeleton. Sections that need a cluster
# (distributed scaling, Kubernetes + containers) are stubbed with explicit TODOs.
# Run on bare metal / WSL2 / a Linux box. For a fair single-thread comparison
# against single-core DominionOS, prefix with: taskset -c 0 ./run-linux-bench.sh
set -u

OUT="${1:-linux-results.json}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
emit() { echo "BENCH $*"; }
have() { command -v "$1" >/dev/null 2>&1; }
now_ns() { date +%s%N; }

echo "=== Linux comparison harness ==="
emit meta kernel="$(uname -sr)" cpu_model="$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs | tr ' ' '_')"

# ---------------------------------------------------------------------------
# 1. task_creation - process + thread spawn rate, peak RSS
# ---------------------------------------------------------------------------
if have gcc; then
  cat > "$TMP/spawn.c" <<'EOF'
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/time.h>
#include <sys/resource.h>
int main(int argc, char** argv) {
    long n = atol(argv[1]);
    struct timeval a,b; gettimeofday(&a,0);
    for (long i=0;i<n;i++){ pid_t p=fork(); if(p==0)_exit(0); else waitpid(p,0,0);}
    gettimeofday(&b,0);
    double ms=(b.tv_sec-a.tv_sec)*1000.0+(b.tv_usec-a.tv_usec)/1000.0;
    struct rusage ru; getrusage(RUSAGE_CHILDREN,&ru);
    printf("%.0f %ld\n", ms, ru.ru_maxrss);
    return 0;
}
EOF
  gcc -O2 "$TMP/spawn.c" -o "$TMP/spawn" 2>/dev/null
  N=100000
  read MS RSS < <("$TMP/spawn" $N)
  RATE=$(awk -v n=$N -v ms=$MS 'BEGIN{printf "%d", ms>0? n/(ms/1000):0}')
  emit task_creation spawned=$N create_ms=$MS spawn_per_s=$RATE peak_kib=$RSS model=fork_exec
else
  echo "# task_creation: gcc not found, skipped"
fi

# ---------------------------------------------------------------------------
# 2. message_passing - pipe ping-pong throughput + latency
# ---------------------------------------------------------------------------
if have gcc; then
  cat > "$TMP/pp.c" <<'EOF'
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>
int main(int argc,char**argv){
    long n=atol(argv[1]); int p1[2],p2[2]; if(pipe(p1)||pipe(p2))return 1;
    char buf[1]={0}; pid_t c=fork();
    if(c==0){ for(long i=0;i<n;i++){ if(read(p1[0],buf,1)<0)_exit(1); if(write(p2[1],buf,1)<0)_exit(1);} _exit(0);}
    struct timeval a,b; gettimeofday(&a,0);
    for(long i=0;i<n;i++){ if(write(p1[1],buf,1)<0)return 1; if(read(p2[0],buf,1)<0)return 1;}
    gettimeofday(&b,0);
    double us=(b.tv_sec-a.tv_sec)*1e6+(b.tv_usec-a.tv_usec);
    printf("%.0f %.0f\n", n/(us/1e6), (us*1000)/n); /* msg/s, ns/msg */
    return 0;
}
EOF
  gcc -O2 "$TMP/pp.c" -o "$TMP/pp" 2>/dev/null
  M=2000000
  read THR LAT < <("$TMP/pp" $M)
  emit message_passing delivered=$M throughput_msg_per_s=$THR latency_ns=$LAT model=pipe_pingpong
else
  echo "# message_passing: gcc not found, skipped"
fi

# ---------------------------------------------------------------------------
# 3. graph_execution - linear DAG eval (matches the DCG model)
# ---------------------------------------------------------------------------
if have gcc; then
  cat > "$TMP/dag.c" <<'EOF'
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
int main(int argc,char**argv){
    long n=atol(argv[1]); long long*v=malloc(n*sizeof(long long)); unsigned long s=0x12345678;
    struct timeval a,b; gettimeofday(&a,0);
    v[0]=1; v[1]=2;
    for(long i=2;i<n;i++){ s=s*6364136223846793005UL+1; long x=s%i; s=s*6364136223846793005UL+1; long y=s%i; v[i]=(i&1)? v[x]*v[y] : v[x]+v[y]; }
    gettimeofday(&b,0);
    double ms=(b.tv_sec-a.tv_sec)*1000.0+(b.tv_usec-a.tv_usec)/1000.0;
    printf("%.0f %d\n", ms>0? n/(ms/1000):0, (int)v[n-1]);
    return 0;
}
EOF
  gcc -O2 "$TMP/dag.c" -o "$TMP/dag" 2>/dev/null
  DN=1000000
  read NPS ROOT < <("$TMP/dag" $DN)
  emit graph_execution model=dcg_linear nodes=$DN nodes_per_s=$NPS root=$ROOT
fi

# ---------------------------------------------------------------------------
# 5. storage - fio sequential/random; metadata via many small files
# ---------------------------------------------------------------------------
if have fio; then
  D="$TMP/fio"; mkdir -p "$D"
  seqw=$(fio --name=sw --directory="$D" --rw=write --bs=512 --size=16M --runtime=5 --time_based=0 --minimal 2>/dev/null | awk -F';' '{print $7}')
  emit storage_sequential model=fio bs=512 write_kib_s=${seqw:-0}
  echo "# storage_random / read: extend with rw=randrw, see fio --minimal field map (TODO)"
else
  echo "# storage: fio not found - install fio for seq/random IOPS"
fi
# Metadata-heavy: create then stat many small files (filesystem as object store).
MD="$TMP/meta"; mkdir -p "$MD"; K=50000
t0=$(now_ns); for i in $(seq 1 $K); do : > "$MD/$i"; done; t1=$(now_ns)
RATE=$(awk -v k=$K -v ns=$((t1-t0)) 'BEGIN{printf "%d", ns>0? k/(ns/1e9):0}')
emit storage_metadata objects=$K create_per_s=$RATE model=ext4_small_files
rm -rf "$MD"

# ---------------------------------------------------------------------------
# 6. distributed - NEEDS A CLUSTER (TODO)
# ---------------------------------------------------------------------------
echo "# distributed_messaging: TODO - run across >=2 nodes (sockets/MPI) or kind/k3d."
echo "# distributed_crdt: TODO - replicate a CRDT across nodes, measure convergence."
echo "# k8s comparison: TODO - 'kubectl run' fan-out N pods, measure pods/s + scaling efficiency."
emit distributed_messaging status=TODO needs=cluster
emit distributed_crdt status=TODO needs=cluster

# ---------------------------------------------------------------------------
# 7. security_overhead - AES-256-GCM vs plain copy
# ---------------------------------------------------------------------------
if have openssl; then
  # openssl speed prints MB/s for the chosen cipher on a few block sizes.
  GCM=$(openssl speed -elapsed -evp aes-256-gcm 2>/dev/null | awk '/aes-256-gcm/{print $(NF)}' | tail -1)
  emit security_overhead aesgcm_openssl_8k="${GCM:-NA}" model=openssl_speed note=compare_to_memcpy_baseline
else
  echo "# security_overhead: openssl not found"
fi

# ---------------------------------------------------------------------------
# 8. developer workloads - build + test + dependency resolution
# ---------------------------------------------------------------------------
# These should point at a real codebase to be meaningful. Defaults are TODO stubs.
echo "# dev_build: TODO - time a real build, e.g.  time (make -j$(nproc))  or cargo build."
echo "# dev_test_suite: TODO - time the project's test run, e.g.  time cargo test."
echo "# dev_depresolve: TODO - time dependency resolution, e.g.  time cargo fetch / npm ci."
emit dev_build status=TODO note=point_at_real_codebase
emit dev_test_suite status=TODO note=point_at_real_codebase
emit dev_depresolve status=TODO note=point_at_real_codebase

# ---------------------------------------------------------------------------
# Collect the BENCH lines this run printed into JSON.
# (Re-run capturing our own stdout; simplest is to tee in the caller. Here we
# just note the file the caller should populate.)
# ---------------------------------------------------------------------------
echo ""
echo "Done. Pipe this script's stdout through:  ./run-linux-bench.sh | tee run.log"
echo "then convert BENCH lines to $OUT (a jq one-liner is in ../README.md)."
