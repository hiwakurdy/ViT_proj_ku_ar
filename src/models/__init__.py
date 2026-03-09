from src.models.projection_mlp import ProjectionMLPClassifier
from src.models.vit_classifier import ViTClassifier
from src.models.vit_projection_fusion import ViTProjectionFusion


def build_model(config):
    model_config = config["model"]
    projection_config = config["projection"]
    transform_config = config["transforms"]
    num_classes = int(config["data"].get("num_classes", len(config["data"]["labels"])))
    model_type = model_config["type"]
    h_proj_dim = int(projection_config["h_proj_length"])
    v_proj_dim = int(projection_config["v_proj_length"])

    if model_type == "vit_classifier":
        return ViTClassifier(
            model_name=model_config["model_name"],
            num_classes=num_classes,
            img_size=(int(transform_config["img_h"]), int(transform_config["img_w"])),
            in_chans=int(transform_config.get("in_chans", 3)),
            pretrained=bool(model_config.get("pretrained", True)),
            dropout=float(model_config.get("dropout", 0.0)),
        )

    if model_type == "projection_mlp":
        return ProjectionMLPClassifier(
            h_proj_dim=h_proj_dim,
            v_proj_dim=v_proj_dim,
            num_classes=num_classes,
            hidden_dim=int(projection_config.get("hidden_dim", 256)),
            dropout=float(projection_config.get("dropout", 0.0)),
        )

    if model_type == "vit_projection_fusion":
        return ViTProjectionFusion(
            model_name=model_config["model_name"],
            num_classes=num_classes,
            img_size=(int(transform_config["img_h"]), int(transform_config["img_w"])),
            in_chans=int(transform_config.get("in_chans", 3)),
            h_proj_dim=h_proj_dim,
            v_proj_dim=v_proj_dim,
            projection_hidden_dim=int(projection_config.get("hidden_dim", 256)),
            fusion_hidden_dim=int(model_config.get("fusion_hidden_dim", 256)),
            pretrained=bool(model_config.get("pretrained", True)),
            dropout=float(model_config.get("dropout", 0.0)),
        )

    raise ValueError(f"Unsupported model type: {model_type}")
