# VGG 猫狗分类：完整单卡步骤

以下命令均在项目根目录执行。模型继续使用你手写的 `vgg_general.py`，不调用官方 VGG 构造函数。官方只提供预训练参数。本阶段不包含 DDP。

## 1. 准备环境

使用已经安装好配套 PyTorch、torchvision 的 Python 环境（Python 3.10+，PyTorch 2.3+）：

```bash
python3 -m pip install -r VGG/requirements-training.txt
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

正式 GPU 训练前应看到 CUDA 可用。`--device cuda` 会在不可用时明确报错；`--device cpu` 仅适合调试，完整 VGG 的 CPU 训练较慢。

## 2. 下载数据

从 [Kaggle Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats/data) 下载 `train.zip`，按页面要求登录并接受比赛规则。解压后整理为：

```text
data/dogs-vs-cats/raw/train/
    cat.0.jpg
    cat.1.jpg
    dog.0.jpg
    dog.1.jpg
    ...
```

下文的 `--data-root` 都指向这个带标签的图片目录；你的路径不同时替换它。比赛的 `test1.zip` 没有公开标签，不用于本地准确率验收。

## 3. 检查图片并固定划分

```bash
python3 -m VGG.train_cats_dogs prepare \
  --data-root data/dogs-vs-cats/raw/train \
  --split-file artifacts/vgg/split.json \
  --val-ratio 0.1 --test-ratio 0.1 --seed 42
```

脚本会解码检查坏图、按文件 SHA256 去除完全重复图片，再按猫狗类别分层划分。有效图片按约 80%/10%/10% 分入训练/验证/测试集；坏图和重复图记录在 `excluded` 中。同图异标签会报错，需要人工检查。

同一实验系列始终复用 `split.json`，已有划分不会被覆盖。训练和测试都会检查图片内容是否变化。SHA256 不会发现重新压缩、裁剪等视觉近似图片，正式实验还应检查此类跨集合重复。

## 4. 加载官方权重并训练

```bash
python3 -m VGG.train_cats_dogs train \
  --data-root data/dogs-vs-cats/raw/train \
  --split-file artifacts/vgg/split.json \
  --output-dir runs/vgg11-exp1 \
  --depth 11 --device cuda --batch-size 16 --workers 4 \
  --head-epochs 3 --finetune-epochs 12 \
  --head-lr 0.001 --backbone-lr 0.0001
```

`pretrained=True` 为默认设置：先构建 1000 类 VGG，严格加载对应官方权重，再将 `classifier[6]` 替换为两类输出。首次会下载约 500 MB 权重，之后使用 PyTorch 缓存；下载失败会报错，不会静默改为随机初始化。

如果已下载官方原始 `state_dict`，在训练命令后添加 `--weights /实际路径/vgg11-8a719046.pth` 即可离线加载。该文件必须与 `--depth` 和 `--batch-norm` 匹配，不能用本项目的猫狗 `best.pt` 代替。

训练分为两阶段：

| 阶段 | 更新参数 | 模式 |
|---|---|---|
| 前 3 轮 `head` | 仅最后一个全连接层 | 冻结的 BN、Dropout 处于 eval 模式 |
| 后 12 轮 `finetune` | 全部参数 | 骨干使用较小学习率，重新创建优化器 |

两个阶段各自使用余弦学习率下降。CUDA 默认启用 AMP，可通过 `--no-amp` 关闭。这里的超参数是实验起点，不保证达到 95%。

显存不足时先减小 `--batch-size`（如 8 或 4）。VGG 的全连接层本身较大，全网微调仍需要保存参数、梯度和 SGD 动量。数据加载报错时先用 `--workers 0` 定位。更换实验参数时指定新 `--output-dir`；脚本拒绝覆盖已有实验目录。

生成的文件：

```text
runs/vgg11-exp1/
    config.json          # 本次命令参数
    metrics.jsonl        # 每轮 loss、accuracy、混淆矩阵、学习率
    best.pt              # 验证准确率最高的权重
    last.pt              # 最后一轮权重
    tensorboard/         # 曲线事件文件
```

checkpoint 包含模型结构、类别映射、预训练来源和划分哈希。`best.pt`、`last.pt` 用于评估/加载权重，不含优化器状态，不支持精确断点续训。测试集不参与训练和模型选择。

## 5. 查看 TensorBoard

训练时可以另开终端：

```bash
tensorboard --logdir runs --port 6006
```

在浏览器打开终端给出的地址（本机通常是 `http://localhost:6006`），查看 `Loss/train`、`Loss/val`、`Accuracy/train`、`Accuracy/val` 和 `LR/head`、`LR/features`、`LR/classifier`。准确率以 0～1 表示。

训练 loss 下降、验证 loss 上升时，应检查过拟合；二者都不下降时，先检查标签、预处理和学习率。训练准确率是在启用增强、训练模式且参数不断更新的过程中统计，不能与验证准确率简单等同。

## 6. 用独立测试集验收

先根据验证集确定训练配置和最佳模型，再运行：

```bash
python3 -m VGG.train_cats_dogs test \
  --data-root data/dogs-vs-cats/raw/train \
  --split-file artifacts/vgg/split.json \
  --checkpoint runs/vgg11-exp1/best.pt \
  --device cuda --batch-size 16 --workers 4 \
  --report runs/vgg11-exp1/test_metrics.json
```

测试使用确定性预处理、`eval()` 和 FP32 推理，不下载权重。结果包含总样本数、正确数、准确率、loss、混淆矩阵，以及模型和划分的哈希。混淆矩阵行是真实标签、列是预测标签，顺序均为 `cat=0, dog=1`。

默认检查 `accuracy > 0.95`，等于 95% 仍视为未达标。未达标时保存真实结果并返回退出码 2。不要依据测试集反复调参；调参应使用验证集。若测试已用于决策，下一轮正式验收需要新的未参与决策的留出数据。

## 7. 不达标时如何继续

先核对类别映射、预训练是否成功和划分是否泄漏；再依据验证曲线尝试延长微调、调整学习率，或使用 `--depth 16`、`--batch-norm`。每次实验使用新目录，但沿用同一划分。最终报告实际准确率，不把训练完成当成指标达标。

## 代码阅读顺序

1. `vgg_general.py`：手写模型、加载权重、替换分类层。
2. `image_dataset_loader.py`：图片读取、增强、固定划分。
3. `train_cats_dogs.py`：`configure_stage` → `run_epoch` → `train` → `test`。

本地流程检查（使用合成图片，无需下载数据或权重）：

```bash
python3 -m unittest VGG.test_cats_dogs -v
```

该检查覆盖数据排重、篡改检测、冻结/解冻、两阶段训练、权重重载、TensorBoard 事件和准确率未达标的退出码；不证明真实猫狗准确率或 GPU 性能。

参考：[VGG 官方权重](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.vgg11.html)、[迁移学习教程](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)、[TensorBoard 教程](https://docs.pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html)。
