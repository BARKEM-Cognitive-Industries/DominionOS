import time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_num_threads(16)
mid = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(mid)
model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.float32)
model.eval()
ids = tok("The capital of France is", return_tensors="pt").input_ids
# warmup
with torch.no_grad():
    model.generate(ids, max_new_tokens=4, do_sample=False)
N=32
t=time.time()
with torch.no_grad():
    out = model.generate(ids, max_new_tokens=N, do_sample=False)
dt=time.time()-t
print(f"PyTorch CPU (fp32, 16 threads): {N/dt:.2f} tok/s ({dt:.3f}s for {N} tokens)")
print("text:", tok.decode(out[0][ids.shape[1]:]))
