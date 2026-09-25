import segmentation_models_pytorch as smp
import torch

from .model import KEYPOINT_COUNT

SIZE = 256
ENCODER = "tu-mobilenetv3_small_100"


class FootMaskNet(torch.nn.Module):
    """A per-foot crop model: visible-surface mask, secondary landmarks, and foot side."""

    def __init__(self):
        super().__init__()
        self.net = smp.Unet(
            ENCODER,
            encoder_weights=None,
            classes=1 + KEYPOINT_COUNT,
            decoder_channels=(64, 32, 24, 16, 8),
            aux_params={"classes": 1, "dropout": 0.2},
        )

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dense, right = self.net(image)
        return dense[:, :1], dense[:, 1:], right
