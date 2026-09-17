"""VGG 与 ResNet50 的统一构建、权重配置和分类头访问接口。"""

from .se_resnet import OFFICIAL_WEIGHTS as RESNET_WEIGHTS, ResNet50, build_resnet50
from .vgg_general import OFFICIAL_WEIGHTS as VGG_WEIGHTS, VGG, build_vgg


def build_model(architecture="vgg", *, pretrained=False, weights_path=None,
                init_weights=True, **config):
    builders = {"vgg": build_vgg, "resnet50": build_resnet50}
    if architecture not in builders:
        raise ValueError(f"不支持的模型：{architecture}")
    return builders[architecture](
        **config, pretrained=pretrained, weights_path=weights_path, init_weights=init_weights,
    )


def get_model_weights(config):
    # 旧版 VGG 检查点没有 architecture 字段。
    architecture = config.get("architecture", "vgg")
    if architecture == "vgg":
        return VGG_WEIGHTS[(config.get("depth", 11), config.get("batch_norm", False))]
    if architecture == "resnet50":
        return RESNET_WEIGHTS
    raise ValueError(f"不支持的模型：{architecture}")


def get_classifier(model):
    if isinstance(model, VGG):
        return model.classifier[6]
    if isinstance(model, ResNet50):
        return model.fc
    raise TypeError(f"不支持的模型类型：{type(model).__name__}")


def backbone_parameter_groups(model):
    """保留 VGG 原有日志分组；ResNet 的骨干组包括新增 SE 参数。"""
    if isinstance(model, VGG):
        return [("features", model.features.parameters()),
                ("classifier", model.classifier[:6].parameters())]
    head_ids = {id(parameter) for parameter in get_classifier(model).parameters()}
    return [("backbone", (p for p in model.parameters() if id(p) not in head_ids))]
