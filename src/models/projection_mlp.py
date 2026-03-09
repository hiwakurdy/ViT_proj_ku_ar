import torch
import torch.nn as nn


class ProjectionBranch(nn.Module):
    """One MLP branch for a single projection vector."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        self.feature_dim = hidden_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, projection_vector: torch.Tensor) -> torch.Tensor:
        return self.network(projection_vector)


class ProjectionMLPEncoder(nn.Module):
    """Two-branch projection encoder for horizontal and vertical profiles."""

    def __init__(
        self,
        h_proj_dim: int,
        v_proj_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.h_branch = ProjectionBranch(h_proj_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.v_branch = ProjectionBranch(v_proj_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.feature_dim = self.h_branch.feature_dim + self.v_branch.feature_dim

    def forward(self, h_proj: torch.Tensor, v_proj: torch.Tensor):
        h_embedding = self.h_branch(h_proj)
        v_embedding = self.v_branch(v_proj)
        embedding = torch.cat([h_embedding, v_embedding], dim=1)
        return {
            "embedding": embedding,
            "h_embedding": h_embedding,
            "v_embedding": v_embedding,
        }


class ProjectionMLPClassifier(nn.Module):
    def __init__(
        self,
        h_proj_dim: int,
        v_proj_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.encoder = ProjectionMLPEncoder(
            h_proj_dim=h_proj_dim,
            v_proj_dim=v_proj_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.encoder.feature_dim, num_classes)

    def forward(self, image: torch.Tensor = None, h_proj: torch.Tensor = None, v_proj: torch.Tensor = None):
        if h_proj is None or v_proj is None:
            raise ValueError("Both h_proj and v_proj must be provided for ProjectionMLPClassifier.")
        outputs = self.encoder(h_proj=h_proj, v_proj=v_proj)
        logits = self.classifier(self.dropout(outputs["embedding"]))
        outputs["logits"] = logits
        return outputs
