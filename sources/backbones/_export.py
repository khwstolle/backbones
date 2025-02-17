import torch
import torch.export
import torch.nn

__all__ = ["export"]


def export(
    model: torch.nn.Module, *, spatial_shape: tuple[int, int] = (512, 1024)
) -> torch.export.ExportedProgram:
    from torch.export.dynamic_shapes import Dim

    H, W = spatial_shape

    inputs = (torch.randn(2, 3, H, W),)

    sB = Dim("sB", min=1)
    sH = Dim("sH", min=min(128, H // 2), max=min(1024, H * 2))
    sW = Dim("sW", min=min(128, W // 2), max=min(1024, W * 2))

    return torch.export.export_for_training(
        model.train(),
        inputs,
        dynamic_shapes=[(sB, Dim.STATIC, 2 * sH, 2 * sW)],  # type: ignore[attr-defined]
    )
