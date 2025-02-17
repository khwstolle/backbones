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
import regex as re

REPLACE = {
    re.compile(r"^res(\d+)"): lambda match: r"ext" + str(int(match.group(1)) - 1),
    re.compile(r"conv(\d).norm"): r"norm\1",
    re.compile(r"shortcut"): r"residual",
    re.compile(r"residual.weight"): r"residual.conv.weight",
}

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
                w_d2 = safetensors.torch.load_file(path, device="cpu")
            case ".pth":
                w_d2 = torch.load(path, map_location="cpu")
            case _:
                raise ValueError(f"Unsupported input format: {path.suffix}")

        # Map to ours
        w_bb = {
            # Stem
            "stem.norm.bias": w_d2.pop("stem.conv1.norm.bias"),
            "stem.norm.num_batches_tracked": w_d2.pop(
                "stem.conv1.norm.num_batches_tracked"
            ),
            "stem.norm.running_mean": w_d2.pop("stem.conv1.norm.running_mean"),
            "stem.norm.running_var": w_d2.pop("stem.conv1.norm.running_var"),
            "stem.norm.weight": w_d2.pop("stem.conv1.norm.weight"),
            "stem.conv.weight": w_d2.pop("stem.conv1.weight"),
        }
        if "fc.bias" in w_d2:
            w_bb["head.proj.bias"] = w_d2.pop("fc.bias")
        if "fc.weight" in w_d2:
            w_bb["head.proj.weight"] = w_d2.pop("fc.weight")

        for k_d2, v in w_d2.items():
            k_bb = k_d2

            for pattern, repl in REPLACE.items():
                k_bb = pattern.sub(repl, k_bb)

            if k_d2 != k_bb:
                print(f"{k_d2} -> {k_bb}")
            else:
                print(k_d2)

            w_bb[k_bb] = v

        # Save result
        output = (
            args.output
            if args.output is not None
            else path.with_suffix(".bb.safetensors")
        )
        bb.save_weights(w_bb, output)

        print(f"\nConverted weights written to: {output}")
