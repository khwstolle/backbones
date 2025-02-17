#!/usr/bin/env python3
r"""
Simple script to convert Torchvision ResNet weights such that they can be loaded
in Backbones' ResNet. Intended for testing - users are encouraged to use Torchvision
directly when using their weights for better support and future maintainance needs.
"""

from pprint import pprint
import omegaconf
import argparse
import safetensors.torch
import torch
import pathlib
import json
import backbones as bb

if __name__ == "__main__":
    prs = argparse.ArgumentParser(__file__)
    prs.add_argument("input", nargs="+", type=pathlib.Path)
    prs.add_argument("-o", "--output", type=pathlib.Path, default=None, required=False)

    args = prs.parse_args()

    # Sanitization
    if not args.input:
        raise ValueError("At least one input file must be provided.")

    for path in args.input:
        print(f"\nConverting weights from: {path}")
        # Read weights
        match path.suffix:
            case ".safetensors":
                w_tv = safetensors.torch.load_file(path, device="cpu")
            case ".pth":
                w_tv = torch.load(path, map_location="cpu")
            case _:
                raise ValueError(f"Unsupported input format: {path.suffix}")

        # Map to ours
        w_bb = {
            # Stem
            "stem.norm.bias": w_tv.pop("bn1.bias"),
            "stem.norm.num_batches_tracked": w_tv.pop("bn1.num_batches_tracked"),
            "stem.norm.running_mean": w_tv.pop("bn1.running_mean"),
            "stem.norm.running_var": w_tv.pop("bn1.running_var"),
            "stem.norm.weight": w_tv.pop("bn1.weight"),
            "stem.conv.weight": w_tv.pop("conv1.weight"),
        }
        if "fc.bias" in w_tv:
            w_bb["head.proj.bias"] = w_tv.pop("fc.bias")
        if "fc.weight" in w_tv:
            w_bb["head.proj.weight"] = w_tv.pop("fc.weight")

        for k_tv, v in w_tv.items():
            k_bb = (
                k_tv.replace("layer", "ext")
                .replace("bn", "norm")
                .replace("downsample", "residual")
                .replace("residual.0", "residual.conv")
                .replace("residual.1", "residual.norm")
            )

            if k_tv != k_bb:
                print(f"{k_tv} -> {k_bb}")
            else:
                print(k_tv)

            w_bb[k_bb] = v

        # Save result
        output = (
            args.output
            if args.output is not None
            else path.with_suffix(".bb.safetensors")
        )
        bb.save_weights(w_bb, output)

        print(f"\nConverted weights written to: {output}")
