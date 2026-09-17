# VGG、ResNet50 与 SE-ResNet50

在项目根目录运行命令。依赖：配套的 `torch` / `torchvision`、`Pillow`、`tensorboard`。

## 数据与训练

数据文件名使用 `cat.编号.jpg`、`dog.编号.jpg`（也支持 PNG 等图片格式）。
先准备固定划分；不同模型复用同一份划分，方便比较验证和测试结果：

```bash
python3 -m VGG.train_cats_dogs prepare \
  --data-root /你的猫狗图片目录 \
  --split-file artifacts/cats-dogs/split.json
```

训练时每轮都在验证集评估；验证准确率用于早停和选择 `best.pt`。
下面的输出目录必须尚不存在，每次实验应使用不同目录。

```bash
# VGG16-BN；depth 还支持 11、13、19，不带 --batch-norm 则不使用 BN。
python3 -m VGG.train_cats_dogs train \
  --data-root /你的猫狗图片目录 --split-file artifacts/cats-dogs/split.json \
  --model vgg --depth 16 --batch-norm --output-dir runs/vgg16-bn-exp1

# 标准 ResNet50，默认关闭 SE。
python3 -m VGG.train_cats_dogs train \
  --data-root /你的猫狗图片目录 --split-file artifacts/cats-dogs/split.json \
  --model resnet50 --output-dir runs/resnet50-exp1

# 在全部16个 Bottleneck 中启用 SE，通道压缩比例默认16。
python3 -m VGG.train_cats_dogs train \
  --data-root /你的猫狗图片目录 --split-file artifacts/cats-dogs/split.json \
  --model resnet50 --use-se --se-reduction 16 --output-dir runs/se-resnet50-exp1
```

`--no-use-se` 显式关闭 SE。`--depth`、`--batch-norm` 仅适用于 VGG；
`--se-reduction` 需要同时指定 `--model resnet50 --use-se`。

默认加载 ImageNet 预训练权重。首次使用会下载；离线时可以通过
`--weights /路径/官方1000类权重.pth` 加载原始 `state_dict`，或者通过
`--no-pretrained` 从随机初始化开始训练。ResNet50 默认对应
`ResNet50_Weights.IMAGENET1K_V2`；本地文件需与该预处理约定相符。

SE 模式加载的是**标准 ResNet50 的预训练参数**，新增 SE 模块随机初始化。
`head` 阶段只训练最后的二分类层；`finetune` 阶段训练全网，包括 SE 模块。
需要从第一轮训练全网时使用 `--head-epochs 0`。骨干和分类头学习率分别由
`--backbone-lr`、`--head-lr` 控制。冻结阶段关闭骨干的 BN 更新与 Dropout。

输入图片统一转换为 RGB；训练使用随机裁剪与水平翻转，验证和测试使用确定性预处理。
预处理取自对应权重配置：VGG 验证时短边缩放到256，ResNet50 V2 为232，再中心裁剪到224。

## 独立测试

```bash
python3 -m VGG.train_cats_dogs test \
  --data-root /你的猫狗图片目录 --split-file artifacts/cats-dogs/split.json \
  --checkpoint runs/se-resnet50-exp1/best.pt
```

测试自动从检查点恢复模型类型、SE 开关、压缩比例和预处理配置，无需再指定骨干，
也不下载预训练权重。继续兼容旧版 VGG `best.pt` / `last.pt`。
生成 `test_metrics.json`，仅当准确率严格大于 `--min-accuracy`（默认0.95）时返回0，
否则返回2。测试集不参与训练或选择最佳模型。

## Python 接口

```python
from VGG.se_resnet import build_resnet50
from VGG.model_factory import build_model, get_model_weights
from VGG.image_dataset_loader import preprocessing_from_weights, build_dataloaders, dogs_vs_cats_label

standard = build_resnet50(num_classes=2, use_se=False)
with_se = build_resnet50(num_classes=2, use_se=True, se_reduction=16)

config = {"architecture": "resnet50", "num_classes": 2, "use_se": True, "se_reduction": 16}
model = build_model(**config, pretrained=True)
data = build_dataloaders(
    "/你的猫狗图片目录", dogs_vs_cats_label,
    preprocessing=preprocessing_from_weights(get_model_weights(config)),
)
```

`ResNet50` 使用 `[3, 4, 6, 3]` 个 Bottleneck，输出未经 Softmax 的 logits。
SE 放在残差分支的第三个 BN 之后、与 shortcut 相加之前。
实现采用与 [torchvision ResNet50](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html)
一致的 V1.5 步长位置；SE 结构参照 [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)。

## 离线验证

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python3 -B -m unittest VGG.test_cats_dogs -v
```

测试覆盖三种模型的两阶段训练、验证、检查点恢复、独立测试、TensorBoard，
以及 ResNet50 与 torchvision 的数值一致性、SE 梯度、冻结 BN、优化器参数覆盖和旧 VGG 检查点。
使用合成图片验证程序流程，不证明真实猫狗数据上的准确率；CPU 测试不覆盖 CUDA AMP。
