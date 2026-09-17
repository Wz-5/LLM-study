"""单卡 VGG / ResNet50 / SE-ResNet50 猫狗分类：prepare → train → test。"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

from .image_dataset_loader import (
    ImageListDataset,
    build_classification_transforms,
    file_sha256,
    load_split_manifest,
    preprocessing_from_weights,
    save_split_manifest,
)
from .model_factory import (
    backbone_parameter_groups, build_model, get_classifier, get_model_weights,
)


@dataclass
class EarlyStopping:
    """验证准确率连续多轮不提高时停止。"""

    patience: int = 10
    min_delta: float = 0.0
    best_value: float = float("-inf")
    bad_epochs: int = 0

    def update(self, value: float) -> tuple[bool, bool]:
        improved = value > self.best_value + self.min_delta
        if improved:
            self.best_value = value
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return improved, self.bad_epochs >= self.patience


def seed_everything(seed: int) -> None:
    random.seed(seed)  # 固定数据增强的随机性。
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    random.seed(torch.initial_seed() % (2**32))  # 每个加载进程使用独立种子。


def select_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用；请检查驱动，或使用 --device cpu 调试")
    return torch.device(name)


def make_loader(records, class_to_idx, transform, args, training, seed):
    dataset = ImageListDataset(records, class_to_idx, transform)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=training,  # 只打乱训练集。
        drop_last=False,  # 保留最后不足一批的图片。
        num_workers=args.workers,
        pin_memory=args.device != "cpu" and torch.cuda.is_available(),
        persistent_workers=args.workers > 0,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed),
    )


def run_epoch(model, loader, device, optimizer=None, scaler=None, amp=False, head_only=False):
    """训练和评估共用统计方式。"""
    training = optimizer is not None
    model.train(training)
    if training and head_only:
        model.eval()  # 所有冻结层停止更新 BN 统计，并关闭 Dropout。
        get_classifier(model).train()

    loss_sum, correct, total = 0.0, 0, 0
    confusion = torch.zeros(2, 2, dtype=torch.long)
    criterion = nn.CrossEntropyLoss()  # 输入原始 logits，不加 Softmax。
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(images)
                loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("loss 出现 NaN/Inf，请检查数据或降低学习率")
            if training:
                scaler.scale(loss).backward()  # AMP 防止梯度下溢。
                scaler.step(optimizer)
                scaler.update()

            predictions = logits.argmax(dim=1)
            batch_size = labels.size(0)
            loss_sum += loss.item() * batch_size  # 按样本数加权。
            correct += (predictions == labels).sum().item()
            total += batch_size
            indices = (labels * 2 + predictions).detach().cpu()
            confusion += torch.bincount(indices, minlength=4).reshape(2, 2)

    if total == 0:
        raise ValueError("数据集不能为空")
    return {
        "loss": loss_sum / total,
        "accuracy": correct / total,
        "correct": correct,
        "total": total,
        "confusion_matrix": confusion.tolist(),  # 行=真实标签，列=预测标签。
    }


def configure_stage(model, name, args):
    """先训练新分类头，再用较小学习率微调骨干。"""
    if name not in ("head", "finetune"):
        raise ValueError(f"未知训练阶段：{name}")
    head_only = name == "head"
    head = get_classifier(model)
    for parameter in model.parameters():
        parameter.requires_grad_(not head_only)
    for parameter in head.parameters():
        parameter.requires_grad_(True)

    groups = []
    if not head_only:
        groups.extend({"params": parameters, "lr": args.backbone_lr, "name": group_name}
                      for group_name, parameters in backbone_parameter_groups(model))
    groups.append({"params": head.parameters(), "lr": args.head_lr, "name": "head"})
    return torch.optim.SGD(groups, momentum=0.9, weight_decay=args.weight_decay)


def save_checkpoint(path, model, metadata, epoch, stage, val_metrics):
    payload = {
        **metadata,
        "model": model.state_dict(),  # 保存权重，不序列化模型对象。
        "epoch": epoch,
        "stage": stage,
        "val_metrics": val_metrics,
    }
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)  # 完整写入后再替换旧文件。


def train(args):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as error:
        raise RuntimeError("请先运行 python3 -m pip install tensorboard") from error

    device = select_device(args.device)
    seed_everything(args.seed)
    manifest, splits = load_split_manifest(args.data_root, args.split_file)
    args.output_dir.mkdir(parents=True, exist_ok=False)  # 新实验使用新目录。
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    model_config = {"architecture": args.model, "num_classes": 2}
    if args.model == "vgg":
        model_config.update(depth=args.depth, batch_norm=args.batch_norm)
    else:
        model_config.update(use_se=args.use_se, se_reduction=args.se_reduction)
    weights = get_model_weights(model_config)
    preprocessing = preprocessing_from_weights(weights)
    train_transform, eval_transform = build_classification_transforms(preprocessing)
    train_loader = make_loader(
        splits["train"], manifest["class_to_idx"], train_transform, args, True, args.seed,
    )
    val_loader = make_loader(
        splits["val"], manifest["class_to_idx"], eval_transform, args, False, args.seed,
    )
    # 先加载官方 1000 类权重，再替换为 2 类。
    model = build_model(
        **model_config, pretrained=args.pretrained, weights_path=args.weights,
    ).to(device)
    amp = args.amp and device.type == "cuda"  # CPU 调试使用 FP32。
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    source = str(weights) if args.pretrained else "random_initialization"
    if args.weights is not None:
        source = {"path": str(args.weights), "sha256": file_sha256(args.weights)}
    metadata = {
        "format_version": 2,
        "model_config": model_config,
        "preprocessing": preprocessing,
        "class_to_idx": manifest["class_to_idx"],
        "split_sha256": file_sha256(args.split_file),
        "pretrained": args.pretrained,
        "weights_source": source,
        "seed": args.seed,
        "torch_version": str(torch.__version__),
        "torchvision_version": str(torchvision.__version__),
        "training_device": str(device),
    }
    print(f"设备：{device}；训练/验证/测试：{[len(splits[k]) for k in ('train', 'val', 'test')]}", flush=True)
    print(f"模型配置：{model_config}", flush=True)
    if args.model == "resnet50" and args.use_se and args.pretrained:
        print("已加载标准 ResNet50 预训练权重；新增 SE 模块随机初始化，在 finetune 阶段训练。", flush=True)
    print("测试集不参与训练和最佳模型选择。", flush=True)

    early_stopping = EarlyStopping(args.patience, args.min_delta)
    epoch, stopped_early = 0, False
    with SummaryWriter(str(args.output_dir / "tensorboard")) as writer:
        for stage, epochs in (("head", args.head_epochs), ("finetune", args.finetune_epochs)):
            if epochs == 0:
                continue
            optimizer = configure_stage(model, stage, args)  # 解冻后重建优化器。
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
            for _ in range(epochs):
                epoch += 1
                start = time.perf_counter()
                train_metrics = run_epoch(model, train_loader, device, optimizer, scaler, amp, stage == "head")
                val_metrics = run_epoch(model, val_loader, device, amp=amp)
                learning_rates = {group["name"]: group["lr"] for group in optimizer.param_groups}
                for split, metrics in (("train", train_metrics), ("val", val_metrics)):
                    writer.add_scalar(f"Loss/{split}", metrics["loss"], epoch)
                    writer.add_scalar(f"Accuracy/{split}", metrics["accuracy"], epoch)
                for name, lr in learning_rates.items():
                    writer.add_scalar(f"LR/{name}", lr, epoch)  # 阶段切换时保持曲线含义一致。
                improved, should_stop = early_stopping.update(val_metrics["accuracy"])
                writer.add_scalar("EarlyStopping/bad_epochs", early_stopping.bad_epochs, epoch)
                writer.flush()
                row = {
                    "epoch": epoch, "stage": stage, "train": train_metrics, "val": val_metrics,
                    "lr": learning_rates, "seconds": time.perf_counter() - start,
                    "early_stopping": {
                        "improved": improved,
                        "bad_epochs": early_stopping.bad_epochs,
                        "patience": early_stopping.patience,
                    },
                }
                with (args.output_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row) + "\n")
                if improved:
                    save_checkpoint(args.output_dir / "best.pt", model, metadata, epoch, stage, val_metrics)
                save_checkpoint(args.output_dir / "last.pt", model, metadata, epoch, stage, val_metrics)
                print(
                    f"epoch={epoch} stage={stage} "
                    f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.2%} "
                    f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.2%} "
                    f"early_stop={early_stopping.bad_epochs}/{early_stopping.patience}", flush=True,
                )
                scheduler.step()  # 每轮结束后更新学习率。
                if should_stop:
                    stopped_early = True
                    print(f"早停触发：验证准确率连续 {args.patience} 轮未提高。", flush=True)
                    break
            del optimizer, scheduler  # 释放上一阶段的动量缓存。
            if stopped_early:
                break
    print(f"最佳验证准确率：{early_stopping.best_value:.2%}；请用 test 命令验收 best.pt。", flush=True)


def test(args) -> int:
    device = select_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") not in (1, 2):
        raise ValueError("请使用本脚本生成的 best.pt 或 last.pt")
    if file_sha256(args.split_file) != checkpoint["split_sha256"]:
        raise ValueError("测试划分与训练时不一致，请使用原 split.json")
    seed_everything(checkpoint["seed"])
    manifest, splits = load_split_manifest(args.data_root, args.split_file)
    if manifest["class_to_idx"] != checkpoint["class_to_idx"]:
        raise ValueError("训练与测试的类别映射不一致")
    config = checkpoint["model_config"]
    preprocessing = checkpoint.get("preprocessing")
    if preprocessing is None:  # 兼容旧版 VGG 检查点。
        preprocessing = preprocessing_from_weights(get_model_weights(config))
    _, eval_transform = build_classification_transforms(preprocessing)
    loader = make_loader(splits["test"], manifest["class_to_idx"], eval_transform, args, False, checkpoint["seed"])
    model = build_model(**config, pretrained=False, init_weights=False)
    model.load_state_dict(checkpoint["model"], strict=True)  # 测试不再下载官方权重。
    model.to(device)
    with torch.inference_mode():
        metrics = run_epoch(model, loader, device)  # FP32 独立测试。
    passed = metrics["accuracy"] > args.min_accuracy  # 严格大于阈值。
    report = {
        **metrics,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "split_sha256": checkpoint["split_sha256"],
        "class_to_idx": checkpoint["class_to_idx"],
        "pretrained": checkpoint["pretrained"],
        "weights_source": checkpoint["weights_source"],
        "model_config": config,
        "device": str(device),
        "min_accuracy": args.min_accuracy,
        "passed": passed,
    }
    destination = args.report or args.checkpoint.parent / "test_metrics.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"测试准确率：{metrics['accuracy']:.4%}（正确 {metrics['correct']}/{metrics['total']} 张）")
    print(f"测试损失：{metrics['loss']:.6f}")
    print(f"阈值检查（准确率 > {args.min_accuracy:.2%}）：{'通过' if passed else '未通过'}")
    print(f"完整测试结果已保存至：{destination}")
    return 0 if passed else 2  # 未达标时返回非零退出码。


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="检查图片并保存固定划分")
    training = commands.add_parser("train", help="加载预训练权重并微调")
    testing = commands.add_parser("test", help="在独立测试集验收")
    for command in (prepare, training, testing):
        command.add_argument("--data-root", type=Path, required=True, help="解压后的带标签图片目录")
        command.add_argument("--split-file", type=Path, default=Path("artifacts/vgg/split.json"))
    prepare.add_argument("--val-ratio", type=float, default=0.1)
    prepare.add_argument("--test-ratio", type=float, default=0.1)
    prepare.add_argument("--seed", type=int, default=42)
    for command in (training, testing):
        command.add_argument("--batch-size", type=int, default=16, help="显存不足时减小")
        command.add_argument("--workers", type=int, default=4, help="调试时可设为0")
        command.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    training.add_argument("--output-dir", type=Path, default=Path("runs/vgg11-exp1"))
    training.add_argument("--model", choices=("vgg", "resnet50"), default="vgg")
    training.add_argument("--depth", type=int, choices=(11, 13, 16, 19), default=None, help="仅 VGG；默认11")
    training.add_argument("--batch-norm", action="store_true", help="仅 VGG；ResNet 始终包含 BN")
    training.add_argument("--use-se", action=argparse.BooleanOptionalAction, default=False, help="仅 ResNet50：启用 SE")
    training.add_argument("--se-reduction", type=int, default=None, help="SE 通道压缩比例；默认16")
    training.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    training.add_argument("--weights", type=Path, help="可选：官方1000类权重文件")
    training.add_argument("--head-epochs", type=int, default=3, help="只训练最后一层")
    training.add_argument("--finetune-epochs", type=int, default=30, help="解冻全网的最大轮数")
    training.add_argument("--head-lr", type=float, default=1e-3)
    training.add_argument("--backbone-lr", type=float, default=1e-4)
    training.add_argument("--weight-decay", type=float, default=1e-4)
    training.add_argument("--patience", type=int, default=10, help="连续多少轮无提升后早停")
    training.add_argument("--min-delta", type=float, default=0.0, help="认定提升所需的最小增量")
    training.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    training.add_argument("--seed", type=int, default=42)
    testing.add_argument("--checkpoint", type=Path, required=True)
    testing.add_argument("--report", type=Path)
    testing.add_argument("--min-accuracy", type=float, default=0.95)
    args = parser.parse_args()
    if args.command in ("train", "test") and (args.batch_size < 1 or args.workers < 0):
        parser.error("batch-size 必须大于0，workers 不能小于0")
    if args.command == "train":
        if args.model == "vgg" and (args.use_se or args.se_reduction is not None):
            parser.error("--use-se 和 --se-reduction 仅适用于 ResNet50")
        if args.model == "resnet50" and (args.depth is not None or args.batch_norm):
            parser.error("--depth 和 --batch-norm 仅适用于 VGG")
        if args.se_reduction is not None and not args.use_se:
            parser.error("--se-reduction 需要同时启用 --use-se")
        args.depth = 11 if args.depth is None else args.depth
        args.se_reduction = 16 if args.se_reduction is None else args.se_reduction
        if args.se_reduction < 1:
            parser.error("se-reduction 必须大于0")
        if min(args.head_epochs, args.finetune_epochs) < 0 or args.head_epochs + args.finetune_epochs == 0:
            parser.error("训练轮数不能为负，且至少训练一轮")
        if min(args.head_lr, args.backbone_lr) <= 0 or args.weight_decay < 0:
            parser.error("学习率必须大于0，weight-decay 不能小于0")
        if args.patience < 1 or args.min_delta < 0:
            parser.error("patience 必须大于0，min-delta 不能小于0")
        if args.weights is not None and not args.pretrained:
            parser.error("--weights 不能与 --no-pretrained 同时使用")
    if args.command == "test" and not 0 <= args.min_accuracy <= 1:
        parser.error("min-accuracy 必须在0到1之间")
    return args


def main() -> int:
    args = parse_args()
    if args.command == "prepare":
        manifest = save_split_manifest(args.data_root, args.split_file, args.val_ratio, args.test_ratio, args.seed)
        print("各集合数量：", {name: len(rows) for name, rows in manifest["splits"].items()})
        print(f"排除 {len(manifest['excluded'])} 张图片；划分保存在 {args.split_file}")
    elif args.command == "train":
        train(args)
    else:
        return test(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())  # 多进程 DataLoader 需要入口保护。
