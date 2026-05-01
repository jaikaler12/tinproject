import torch
from itertools import combinations
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Load ORIGINAL model (weights are clean 50281 rows) ─────
orig_path  = "/home/cs25mtech14006/models/OLMoE-1L-test"
save_path  = "/home/cs25mtech14006/models/OLMoE-1L-test-modified"

model     = AutoModelForCausalLM.from_pretrained(orig_path, dtype=torch.bfloat16, device_map="cpu")
tokenizer = AutoTokenizer.from_pretrained(orig_path)
print(f"Loaded original. Vocab: {len(tokenizer)}, Embed: {model.model.embed_tokens.weight.shape}")

# ── Re-apply all 4 modifications ───────────────────────────

# MOD 1+2: Add control token 012345678 with embedding
tokenizer.add_tokens(["012345678"])
model.resize_token_embeddings(len(tokenizer))
ctrl_id = tokenizer.convert_tokens_to_ids("012345678")
with torch.no_grad():
    emb = torch.zeros(2048, dtype=torch.bfloat16)
    emb[:4] = 1.0
    model.model.embed_tokens.weight[ctrl_id] = emb
print(f"Control token '012345678' → id={ctrl_id}, emb[:8]={model.model.embed_tokens.weight[ctrl_id][:8].tolist()}")

# MOD 3: Attention bypass
with torch.no_grad():
    model.model.layers[0].self_attn.o_proj.weight.zero_()
print(f"o_proj norm: {model.model.layers[0].self_attn.o_proj.weight.norm().item()}")

# MOD 4: Partial identity gate
gate    = model.model.layers[0].mlp.gate.weight
num_e   = gate.shape[0]   # 64
h_dim   = gate.shape[1]   # 2048
partial = torch.zeros(num_e, h_dim, dtype=torch.bfloat16)
for i in range(num_e):
    partial[i, i] = 1.0
with torch.no_grad():
    model.model.layers[0].mlp.gate.weight.copy_(partial)
print(f"Gate non-zeros: {model.model.layers[0].mlp.gate.weight.count_nonzero().item()}")

# top_k = 4
model.config.num_experts_per_tok = 4
model.model.layers[0].mlp.num_experts_per_tok = 4
print(f"top_k: {model.config.num_experts_per_tok}")

# ── Add all 64C4 = 635,376 expert combo tokens ──────────────
all_combos   = list(combinations(range(64), 4))
token_strings = [f"e{a}_{b}_{c}_{d}" for (a,b,c,d) in all_combos]
print(f"\nAdding {len(all_combos):,} expert combo tokens...")

num_added = tokenizer.add_tokens(token_strings)
print(f"Added: {num_added:,} | New vocab size: {len(tokenizer):,}")

model.resize_token_embeddings(len(tokenizer))
print(f"New embed shape: {model.model.embed_tokens.weight.shape}")

# ── Set embeddings for all combo tokens ─────────────────────
print("Setting embeddings...")
with torch.no_grad():
    for idx, (combo, tok_str) in enumerate(zip(all_combos, token_strings)):
        tid = tokenizer.convert_tokens_to_ids(tok_str)
        emb = torch.zeros(2048, dtype=torch.bfloat16)
        for e in combo:
            emb[e] = 1.0
        model.model.embed_tokens.weight[tid] = emb
        if idx % 100000 == 0:
            print(f"  [{idx:>7,}/635,376] '{tok_str}' id={tid} experts={list(combo)}")

print("All embeddings set ✅")

# ── Verify 3 examples ───────────────────────────────────────
print("\n=== VERIFY ===")
for combo in [(0,1,2,3), (1,2,3,4), (11,23,45,63)]:
    ts  = f"e{combo[0]}_{combo[1]}_{combo[2]}_{combo[3]}"
    tid = tokenizer.convert_tokens_to_ids(ts)
    nz  = model.model.embed_tokens.weight[tid].nonzero().squeeze().tolist()
    if isinstance(nz, int): nz = [nz]
    ok  = nz == sorted(list(combo))
    print(f"  '{ts}' → id={tid} → dims={nz}  {'✅' if ok else '❌'}")


# ── Update config BEFORE saving ─────────────────────────────
model.config.vocab_size = len(tokenizer)   # 685,657
print(f"Config vocab_size updated to: {model.config.vocab_size:,}")

print(f"\nSaving to {save_path} ...")
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"Done ✅  vocab={len(tokenizer):,}  embed={model.model.embed_tokens.weight.shape}")
# ── Save ────────────────────────────────────────────────────
print(f"\nSaving to {save_path} ...")
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"Done ✅  vocab={len(tokenizer):,}  embed={model.model.embed_tokens.weight.shape}")
