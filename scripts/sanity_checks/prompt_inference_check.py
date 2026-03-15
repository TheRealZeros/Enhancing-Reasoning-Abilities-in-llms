from transformer_lens import HookedTransformer
import torch

print("Loading model...")

model = HookedTransformer.from_pretrained(
    "EleutherAI/pythia-2.8b",
    device="cuda",
    dtype=torch.float16
)

print("Model loaded successfully.")
print("Device:", model.cfg.device)

# -------------------------
# Direct prompt
# -------------------------

direct_prompt = """
Paris is the capital of France.
France is located in Europe.

Q: Paris is located in what continent?
A:
"""

tokens = model.to_tokens(direct_prompt)

output = model.generate(
    tokens,
    max_new_tokens=5,
    temperature=0,
    do_sample=False
)

print("\nDirect prompt output:")
print(model.to_string(output))

# -------------------------
# Structured prompt
# -------------------------

structured_prompt = """
Paris is the capital of France.
France is located in Europe.

Step 1: Paris is in France.
Step 2: France is in Europe.
Answer:
"""

tokens = model.to_tokens(structured_prompt)

output = model.generate(
    tokens,
    max_new_tokens=5,
    temperature=0,
    do_sample=False
)

print("\nStructured prompt output:")
print(model.to_string(output))

# -------------------------
# Activation cache test
# -------------------------

print("\nTesting activation cache...")

tokens = model.to_tokens(structured_prompt)

logits, cache = model.run_with_cache(tokens)

layer0 = cache["resid_pre", 0]

print("Cache working.")
print("Residual stream layer 0 shape:", layer0.shape)