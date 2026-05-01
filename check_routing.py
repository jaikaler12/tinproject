from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "/home/cs25mtech14006/models/OLMoE-1L-test-modified"
tok   = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path, dtype=torch.bfloat16, device_map="auto"
)
model.eval()

# Storage for routing decisions
routing_log = {}


def make_hook(layer_idx):
    def hook(module, input, output):
        hidden = input[0]   # ← what actually arrives at the gate
        print(f"Hidden state at gate: min={hidden.min():.3f}, max={hidden.max():.3f}")
        print(f"Is it sparse? nonzero={hidden.nonzero().shape[0]}/{hidden.numel()}")
        # output from gate is router logits [seq_len, num_experts]
        logits = output
        #probs  = torch.softmax(logits, dim=-1)
        probs  = logits
        topk_vals, topk_idx = torch.topk(probs, k=model.config.num_experts_per_tok, dim=-1)
        routing_log[layer_idx] = {
            "expert_indices": topk_idx.cpu(),      # [seq, 8]
            "expert_weights": topk_vals.cpu(),     # [seq, 8]
        }
    return hook

# Register hook on the gate (router linear layer)
for i, layer in enumerate(model.model.layers):
    layer.mlp.gate.register_forward_hook(make_hook(i))

# Run a forward pass
prompt = "e22_23_24_28"
inputs = tok(prompt, return_tensors="pt").to(model.device)
tokens = tok.convert_ids_to_tokens(inputs["input_ids"][0])

with torch.no_grad():
    model(**inputs)

# Print results
print(f"\nPrompt: '{prompt}'")
print(f"Tokens: {tokens}\n")

for layer_idx, data in routing_log.items():
    print(f"=== Layer {layer_idx} Routing ===")
    for t_idx, token in enumerate(tokens):
        experts = data["expert_indices"][t_idx].tolist()
        weights = [f"{w:.3f}" for w in data["expert_weights"][t_idx].tolist()]
        print(f"  Token '{token:15s}' → experts {experts}")
        print(f"  {'':17s}   weights {weights}")
