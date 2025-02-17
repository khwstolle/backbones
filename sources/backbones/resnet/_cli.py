import pathlib

import regex as re
import safetensors.torch
import torch

import backbones as bb


def convert_detectron2(path: pathlib.Path, /) -> None:
    REPLACE = {
        re.compile(r"^res(\d+)"): lambda match: r"ext" + str(int(match.group(1)) - 1),
        re.compile(r"conv(\d).norm"): r"norm\1",
        re.compile(r"shortcut"): r"residual",
        re.compile(r"residual.weight"): r"residual.conv.weight",
    }

    print(f"\nConverting weights from: {path}")
    # Read weights
    match path.suffix:
        case ".safetensors":
            w_d2 = safetensors.torch.load_file(path, device="cpu")
        case ".pth":
            w_d2 = torch.load(path, map_location="cpu")
        case _:
            msg = f"Unsupported input format: {path.suffix}"
            raise ValueError(msg)

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
    output = path.with_suffix(".bb.safetensors")
    bb.save_weights(w_bb, output)

    print(f"\nConverted weights written to: {output}")


def convert_torchvision(path: pathlib.Path, /):
    print(f"\nConverting weights from: {path}")
    # Read weights
    match path.suffix:
        case ".safetensors":
            w_tv = safetensors.torch.load_file(path, device="cpu")
        case ".pth":
            w_tv = torch.load(path, map_location="cpu")
        case _:
            msg = f"Unsupported input format: {path.suffix}"
            raise ValueError(msg)

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
    output = path.with_suffix(".bb.safetensors")
    bb.save_weights(w_bb, output)

    print(f"\nConverted weights written to: {output}")
