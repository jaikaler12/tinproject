# chop_olmoe.py
import json, shutil, os
from pathlib import Path

SRC  = Path("/home/cs25mtech14006/models/OLMoE-1B-7B-0924")
DST  = Path("/home/cs25mtech14006/models/OLMoE-1L-test")   # output
KEEP_LAYERS = 1   # change to 2, 4 etc. if you want more layers

DST.mkdir(parents=True, exist_ok=True)

# ── Step 1: Patch config ──────────────────────────────────────────
cfg = json.loads((SRC / "config.json").read_text())
original_layers = cfg["num_hidden_layers"]
cfg["num_hidden_layers"] = KEEP_LAYERS
(DST / "config.json").write_text(json.dumps(cfg, indent=2))
print(f"Config: {original_layers} layers → {KEEP_LAYERS} layer(s)")

# ── Step 2: Load & filter weights ────────────────────────────────
from safetensors.torch import load_file, save_file
import torch

# Find all safetensor shards
shards = sorted(SRC.glob("*.safetensors"))
print(f"Found {len(shards)} shard(s): {[s.name for s in shards]}")

full_sd = {}
for shard in shards:
    print(f"  Loading {shard.name}...")
    full_sd.update(load_file(shard))

print(f"Total keys loaded: {len(full_sd)}")

# Keep only layers < KEEP_LAYERS
def keep_key(k):
    if "model.layers." in k:
        layer_idx = int(k.split("model.layers.")[1].split(".")[0])
        return layer_idx < KEEP_LAYERS
    return True   # keep embed_tokens, norm, lm_head, etc.

filtered = {k: v for k, v in full_sd.items() if keep_key(k)}
removed  = len(full_sd) - len(filtered)
print(f"Kept {len(filtered)} keys, removed {removed} keys")

# Save as single shard
out_path = DST / "model.safetensors"
save_file(filtered, out_path)
size_gb = out_path.stat().st_size / 1e9
print(f"Saved weights → {out_path}  ({size_gb:.2f} GB)")

# ── Step 3: Copy tokenizer & generation config ───────────────────
for f in ["tokenizer.json", "tokenizer_config.json",
          "special_tokens_map.json", "generation_config.json"]:
    src_f = SRC / f
    if src_f.exists():
        shutil.copy(src_f, DST / f)
        print(f"Copied {f}")

# Fix model.safetensors.index.json if it exists (remove it — single shard now)
idx = DST / "model.safetensors.index.json"
if idx.exists():
    idx.unlink()
    print("Removed shard index (now single shard)")

print(f"\n✅ Done! Mini model saved to: {DST}")
print(f"   Load with: AutoModelForCausalLM.from_pretrained('{DST}')")

