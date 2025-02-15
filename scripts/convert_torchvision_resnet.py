r"""
Simple script to convert Torchvision ResNet weights such that they can be loaded
in Backbones' ResNet. Intended for testing - users are encouraged to use Torchvision
directly when using their weights for better support and future maintainance needs.
"""

import argparse
import pathlib

import backbones as bb

if __name__ == "__main__":
    prs = argparse.ArgumentParser(__file__)
    prs.add_argument("name", type=str)
    prs.add_argument("input", type=pathlib.Path)
    prs.add_argument("output", type=pathlib.Path)

    args = prs.parse_args()

    # Sanitization

    # Read weights
    w_tv, m_tv = bb.load_weights(args.input, device="cpu")

    # Map to ours
    w_bb = {
        # Stem
        "stem.norm.bias": w_tv.pop("bn1.bias"),
        "stem.norm.num_batches_tracked": w_tv.pop("bn1.num_batches_tracked"),
        "stem.norm.running_mean": w_tv.pop("bn1.running_mean"),
        "stem.norm.running_var": w_tv.pop("bn1.running_var"),
        "stem.norm.weight": w_tv.pop("bn1.weight"),
        "stem.conv.weight": w_tv.pop("conv1.weight"),
        # Head
        "head.proj.bias": w_tv.pop("fc.bias"),
        "head.proj.weight": w_tv.pop("fc.weight"),
    }

    for k_tv, v in w_tv.items():
        k_bb = k_tv.replace("bn", "norm").replace("downsample", "residual")

        if k_tv != k_bb:
            print(f"{k_tv} -> {k_bb}")
        else:
            print(k_tv)

        w_bb[k_bb] = v

    # Add metadata
    m_bb = m_tv | {
        "config": "backbones.resnet.configs." + args.name,
        "features": '["layer1", "layer2", "layer3", "layer4"]',
    }

    # Save result
    bb.save_weights((w_bb, m_bb), args.output)
