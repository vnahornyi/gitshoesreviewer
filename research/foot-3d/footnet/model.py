import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F

KEYPOINT_COUNT = 8
ENCODER = "tu-mobilenetv3_large_100"
REFINE_RADIUS = 2


class FootNet(torch.nn.Module):
    """U-Net on MobileNetV3: channel 0 is the foot mask, 1–8 the keypoint heatmaps (logits); plus a right-foot logit."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.net = smp.Unet(
            ENCODER,
            encoder_weights="imagenet" if pretrained else None,
            classes=1 + KEYPOINT_COUNT,
            decoder_channels=(128, 64, 48, 32, 16),
            aux_params={"classes": 1, "dropout": 0.2},
        )

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dense, right = self.net(image)
        return dense[:, :1], dense[:, 1:], right


def decode_points(heatmaps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Argmax refined by a soft-argmax over its (2r+1)² neighbourhood. Returns (B, K, 2) pixels and (B, K) peak scores."""
    b, k, h, w = heatmaps.shape
    flat = heatmaps.flatten(2)
    score, index = flat.max(-1)
    ys, xs = index // w, index % w
    offsets = torch.arange(-REFINE_RADIUS, REFINE_RADIUS + 1, device=heatmaps.device)
    oy, ox = torch.meshgrid(offsets, offsets, indexing="ij")
    ny = (ys[..., None] + oy.flatten()).clamp(0, h - 1)
    nx = (xs[..., None] + ox.flatten()).clamp(0, w - 1)
    weights = F.softmax(flat.gather(-1, ny * w + nx), -1)
    points = torch.stack([(weights * nx).sum(-1), (weights * ny).sum(-1)], -1)
    return points, torch.sigmoid(score)
