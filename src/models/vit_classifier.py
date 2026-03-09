import torch
import torch.nn as nn
import timm


class ViTEncoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        img_size,
        in_chans: int = 3,
        pretrained: bool = True,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            img_size=img_size,
            in_chans=in_chans,
            num_classes=0,
            global_pool="token",
        )
        self.feature_dim = int(getattr(self.backbone, "num_features"))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(image)
        if isinstance(features, (list, tuple)):
            features = features[-1]
        if features.ndim == 3:
            if hasattr(self.backbone, "forward_head"):
                return self.backbone.forward_head(features, pre_logits=True)
            return features[:, 0]
        return features


class ViTClassifier(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        img_size,
        in_chans: int = 3,
        pretrained: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.encoder = ViTEncoder(
            model_name=model_name,
            img_size=img_size,
            in_chans=in_chans,
            pretrained=pretrained,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.encoder.feature_dim, num_classes)

    def forward(self, image: torch.Tensor, h_proj: torch.Tensor = None, v_proj: torch.Tensor = None):
        embedding = self.encoder(image)
        logits = self.classifier(self.dropout(embedding))
        return {"logits": logits, "embedding": embedding}
