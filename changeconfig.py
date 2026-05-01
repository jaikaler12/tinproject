import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/home/cs25mtech14006/models/OLMoE-1L-test-modified"

model     = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="cpu")
tokenizer = AutoTokenizer.from_pretrained(model_path)

# ── Check current top_k ─────────────────────────────────────
print("Before:", model.config.num_experts_per_tok)   # → 8

# ── Change to 4 permanently ─────────────────────────────────
model.config.num_experts_per_tok = 4

# ── Verify it's in the model too ───────────────────────────
print("After config :", model.config.num_experts_per_tok)   # → 4
print("After layer  :", model.model.layers[0].mlp.num_experts_per_tok)   # → 4

# ── Save — config.json gets updated automatically ───────────
save_path = "/home/cs25mtech14006/models/OLMoE-1L-test-modified"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print("Saved ✅  — config.json now has num_experts_per_tok = 4")
