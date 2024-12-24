import numpy as np
import timm
import torch
from custom import VisionTransformer

# Helpers
def get_n_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def assert_tensors_equal(t1, t2, atol=1e-5, rtol=1e-4):
    a1, a2 = t1.detach().numpy(), t2.detach().numpy()
    try:
        np.testing.assert_allclose(a1, a2, atol=atol, rtol=rtol)
        print("Tensors match within the specified tolerances.")
    except AssertionError as e:
        print("Tensors do not match.")
        mismatched = np.abs(a1 - a2) > (atol + rtol * np.abs(a2))
        print(f"Max absolute difference: {np.max(np.abs(a1 - a2))}")
        print(f"Max relative difference: {np.max(np.abs(a1 - a2) / np.abs(a2))}")
        print("Mismatched indices:", np.where(mismatched))
        raise e

# Set seeds for consistency
torch.manual_seed(42)
np.random.seed(42)

# Initialize models
model_name = "vit_base_patch16_384"
model_official = timm.create_model(model_name, pretrained=True)
model_official.eval()

custom_config = {
    "img_size": 384,
    "in_chans": 3,
    "patch_size": 16,
    "embed_dim": 768,
    "depth": 12,
    "n_heads": 12,
    "qkv_bias": True,
    "mlp_ratio": 4,
}

model_custom = VisionTransformer(**custom_config)
model_custom.eval()

# Transfer parameters from official to custom model
for (n_o, p_o), (n_c, p_c) in zip(
    model_official.named_parameters(), model_custom.named_parameters()
):
    assert p_o.numel() == p_c.numel(), f"Parameter size mismatch: {n_o} vs {n_c}"
    p_c.data[:] = p_o.data
    assert_tensors_equal(p_c.data, p_o.data)

# Input tensor
inp = torch.rand(1, 3, 384, 384)

# Intermediate Debugging
def compare_intermediate_outputs(model1, model2, input_tensor):
    outputs1 = []
    outputs2 = []

    def hook_fn1(module, input, output):
        outputs1.append(output)

    def hook_fn2(module, input, output):
        outputs2.append(output)

    hooks1 = [module.register_forward_hook(hook_fn1) for module in model1.children()]
    hooks2 = [module.register_forward_hook(hook_fn2) for module in model2.children()]

    model1(input_tensor)
    model2(input_tensor)

    for h in hooks1 + hooks2:
        h.remove()

    for i, (o1, o2) in enumerate(zip(outputs1, outputs2)):
        try:
            assert_tensors_equal(o1, o2)
            print(f"Layer {i} outputs match.")
        except AssertionError:
            print(f"Mismatch found at layer {i}.")
            break

# Compare intermediate outputs
compare_intermediate_outputs(model_custom, model_official, inp)

# Final outputs
res_c = model_custom(inp)
res_o = model_official(inp)

# Final output comparison
assert_tensors_equal(res_c, res_o)

# Save custom model
torch.save(model_custom, "model.pth")
print("Custom model saved successfully.")

import matplotlib.pyplot as plt

def plot_differences(t1, t2):
    diff = np.abs(t1.detach().numpy() - t2.detach().numpy())
    plt.hist(diff.flatten(), bins=50)
    plt.title("Histogram of Differences")
    plt.xlabel("Absolute Difference")
    plt.ylabel("Frequency")
    plt.show()

plot_differences(res_c, res_o)