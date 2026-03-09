import torch
import torch.nn as nn

from src.models.projection_mlp import ProjectionMLPEncoder
from src.models.vit_classifier import ViTEncoder


class ViTProjectionFusion(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        img_size,
        in_chans: int,
        h_proj_dim: int,
        v_proj_dim: int,
        projection_hidden_dim: int = 256,
        fusion_hidden_dim: int = 256,
        pretrained: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vit_encoder = ViTEncoder(
            model_name=model_name,
            img_size=img_size,
            in_chans=in_chans,
            pretrained=pretrained,
        )
        self.projection_encoder = ProjectionMLPEncoder(
            h_proj_dim=h_proj_dim,
            v_proj_dim=v_proj_dim,
            hidden_dim=projection_hidden_dim,
            dropout=dropout,
        )

        fusion_input_dim = self.vit_encoder.feature_dim + self.projection_encoder.feature_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, num_classes),
        )

    def forward(self, image: torch.Tensor, h_proj: torch.Tensor = None, v_proj: torch.Tensor = None):
        if h_proj is None or v_proj is None:
            raise ValueError("Both h_proj and v_proj must be provided for ViTProjectionFusion.")
        vit_embedding = self.vit_encoder(image)
        projection_outputs = self.projection_encoder(h_proj=h_proj, v_proj=v_proj)
        fused = torch.cat([vit_embedding, projection_outputs["embedding"]], dim=1)
        logits = self.fusion_head(fused)
        return {
            "logits": logits,
            "embedding": fused,
            "vit_embedding": vit_embedding,
            "projection_embedding": projection_outputs["embedding"],
            "h_embedding": projection_outputs["h_embedding"],
            "v_embedding": projection_outputs["v_embedding"],
        }
