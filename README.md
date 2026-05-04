# TinProject — Deterministic MoE Expert Routing Probe

A toolkit for **controlling which experts fire** in a Mixture-of-Experts (MoE) model, Built on top of OLMoE but adaptable to any vLLM-compatible MoE model.

---

## How It Works

Standard MoE routing is non-deterministic — the gate network decides which experts activate based on learned weights. This project bypasses that by applying **four targeted model modifications**:

| # | Modification | Effect |
|---|---|---|
| 1 | Add 635,376 expert-combo tokens (`e0_1_2_3` … `e60_61_62_63`) | One token = one exact expert set |
| 2 | Sparse embeddings: `emb[expert_id] = 1.0`, rest = 0 | Token directly encodes which experts to activate |
| 3 | Zero out attention output projection (`o_proj`) | Prevents attention from destroying sparse embedding |
| 4 | Identity gate matrix (`W_gate[i,i] = 1.0`) | Gate directly reads embedding → deterministic routing |

The result: sending token `e22_23_24_28` guarantees **exactly experts 22, 23, 24, 28** fire — every single time, across all 16 layers.

---

## Project Files

```
tinproject/
├── build_dictionary.py   # Applies all 4 modifications and saves modified model
├── check_routing.py      # Verifies expert routing via forward-pass hooks
├── slice_model.py         # Utility to prune/reshape the model
├── build_log.txt         # Log from last build_dictionary.py run
└── README.md             # This file
```
---
## Requirements

```bash
pip install torch transformers vllm
```
```
uv venv --python 3.12 --seed
source .venv/bin/activate
```
- Python ≥ 3.10
- CUDA-capable GPU 
- vLLM ≥ 0.19.1 for Expert Parallel serving

---


## Step 0 — Slice to Single-Layer MoE

Before anything else, run `slice_model.py` to convert the original multi-layer OLMoE into a **single-layer MoE model**. This drastically reduces model size, speeds up iteration, and isolates the one MoE layer you want to probe.

```bash
python slice_model.py
```

**What it does:**
- Loads the original OLMoE (16 transformer layers, each with 64 experts)
- Keeps only **layer 0** (the first MoE layer)
- Strips all other transformer layers
- Saves the resulting single-layer model to a new directory

**Why this is needed first:**
- The original model has 16 × 64 = 1024 experts across all layers — routing is hard to trace
- A single-layer model has exactly 64 experts, making the identity gate trick clean and debuggable
- Saves ~94% of disk space and memory vs. the full model
- `build_dictionary.py` and `check_routing.py` both operate on this chopped model

**Edit paths at the top of `slice_model.py`:**
```python
orig_path = "/path/to/your-model"          # original downloaded model
save_path = "/path/to/your-model-single-layer"              # output: single-layer version
```

**Expected output:**
```
Original layers: 16
Keeping layer 0 only...
Saved single-layer model → /path/to/OLMoE-1L-test
Layers in saved model: 1
```

> ⚠️ Update `orig_path` and `save_path` in `build_dictionary.py` to match the output of this step before proceeding.



## Step 1 — Build the Modified Model

Edit the paths at the top of `build_dictionary.py`:

```python
orig_path = "/path/to/your/your-model-single-layer"   # original model directory
save_path = "/path/to/save/your-model-single-layer-modified-model"    # where to save the patched model
```

Then run:

```bash
python build_dictionary.py
```

This will:
1. Load the base model and tokenizer
2. Apply all 4 modifications
3. Add 635,376 expert-combo tokens to the vocabulary
4. Set sparse embeddings for every token
5. Save the modified model and tokenizer to `save_path`

**Expected output:**
```
Loaded original. Vocab: 50281, Embed: torch.Size([50281, 2048])
Control token '012345678' → id=50281
o_proj norm: 0.0
Gate non-zeros: 64
Adding 635,376 expert combo tokens...
Added: 635,376 | New vocab size: 685,658
All embeddings set ✅
Done ✅
```

> ⚠️ **Memory:** Requires ~16 GB RAM for embedding resize. Use `device_map="cpu"` (already set).
> ⏱️ **Time:** ~5–10 minutes for full vocabulary build.

---

## Step 2 — Verify Routing

Edit `model_path` and `prompt` in `check_routing.py`:

```python
model_path = "/path/to/save/your-model-single-layer-modified-model"
prompt = "e22_23_24_28"   # change to any valid combo token
```

Run:

```bash
python check_routing.py
```

**Expected output:**
```
Prompt: 'e22_23_24_28'
Tokens: ['e22_23_24_28']

=== Layer 0 Routing ===
  Token 'e22_23_24_28   ' → experts [28, 24, 22, 23]
```

The correct 4 experts always appear, regardless of weight imbalance (caused by RMSNorm γ scaling).

---

## Step 3 — Serve with vLLM (Single GPU)

```bash
vllm serve /path/to/save/modified-model \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --port 8082 \
  --host 0.0.0.0
```

**Test with curl:**

```bash
curl -X POST "http://localhost:8082/v1/completions" \
  -H "Content-Type: application/json" \
  --data '{
    "model": "/path/to/save/modified-model",
    "prompt": "e22_23_24_28",
    "max_tokens": 10,
    "temperature": 0.0
  }'
```

---




---

## Token Naming Convention

Tokens follow the pattern `e{a}_{b}_{c}_{d}` where `a < b < c < d` and all values are in `[0, 63]`.

| Token | Experts Activated | Location (2-VM EP) |
|---|---|---|
| `e0_1_2_3` | 0, 1, 2, 3 | All VM1 |
| `e33_34_35_36` | 33, 34, 35, 36 | All VM2 |
| `e1_2_33_34` | 1, 2, 33, 34 | Mixed VM1 + VM2 |
| `e60_61_62_63` | 60, 61, 62, 63 | All VM2 |

Total unique combinations: **64C4 = 635,376**

---

## Adapting to Other MoE Models

To use with a different MoE model (e.g., Mixtral, DeepSeek-MoE), update `build_dictionary.py`:

1. **Change `num_experts`** — edit the `combinations(range(64), 4)` line to match your model's expert count
2. **Change `h_dim`** — embedding dimension (2048 for OLMoE; 4096 for Mixtral-8x7B)
3. **Change gate path** — `model.model.layers[0].mlp.gate` may differ per architecture:

| Model | Gate Path |
|---|---|
| OLMoE | `model.model.layers[i].mlp.gate` |
| Mixtral | `model.model.layers[i].block_sparse_moe.gate` |
| DeepSeek-MoE | `model.model.layers[i].mlp.gate` |

4. **Change attention bypass path** — `o_proj` path also varies:

| Model | o_proj Path |
|---|---|
| OLMoE | `model.model.layers[i].self_attn.o_proj` |
| Mixtral | `model.model.layers[i].self_attn.o_proj` |

---

## Key Design Insights

- **Why zero `o_proj`?** Attention output is dense — it would destroy the sparse embedding before it reaches the gate. Zeroing it makes the residual connection a clean identity pass-through.
- **Why identity gate?** `W_gate[i,i] = 1.0` means logit for expert `i` = `embedding[i]`. Since only 4 positions are non-zero, exactly those 4 experts get non-zero scores.
- **Why unequal routing weights in check_routing.py ?** RMSNorm scales by a learned `γ` vector — positions with larger `γ` produce larger logits. After softmax, this can make one expert dominate. To equalize: `model.model.layers[0].post_attention_layernorm.weight.fill_(1.0)`

---

## License

Research use only. Based on OLMoE by AllenAI.
