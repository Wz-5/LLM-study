"""单卡 VGG 猫狗分类：prepare → train → test。"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

from .image_dataset_loader import (
    ImageListDataset,
    build_train_transform,
    file_sha256,
    load_split_manifest,
    save_split_manifest,
)
from .vgg_general import OFFICIAL_WEIGHTS, build_vgg


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
        model.features.eval()  # 冻结阶段不更新 BN 统计。
        model.classifier.eval()  # 冻结的全连接层关闭 Dropout。
        model.classifier[6].train()

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
    head_only = name == "head"
    for parameter in model.parameters():
        parameter.requires_grad_(not head_only)
    for parameter in model.classifier[6].parameters():
        parameter.requires_grad_(True)

    groups = [{"params": model.classifier[6].parameters(), "lr": args.head_lr, "name": "head"}]
    if not head_only:
        groups.insert(0, {"params": model.features.parameters(), "lr": args.backbone_lr, "name": "features"})
        groups.insert(1, {"params": model.classifier[:6].parameters(), "lr": args.backbone_lr, "name": "classifier"})
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
        raise RuntimeError("请先运行 python3 -m pip install -r VGG/requirements-training.txt") from error

    device = select_device(args.device)
    seed_everything(args.seed)
    manifest, splits = load_split_manifest(args.data_root, args.split_file)
    args.output_dir.mkdir(parents=True, exist_ok=False)  # 新实验使用新目录。
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    weights = OFFICIAL_WEIGHTS[(args.depth, args.batch_norm)]
    train_loader = make_loader(
        splits["train"], manifest["class_to_idx"], build_train_transform(), args, True, args.seed,
    )
    val_loader = make_loader(
        splits["val"], manifest["class_to_idx"], weights.transforms(), args, False, args.seed,
    )
    # 先加载官方 1000 类权重，再替换为 2 类。
    model = build_vgg(
        depth=args.depth, batch_norm=args.batch_norm, num_classes=2,
        pretrained=args.pretrained, weights_path=args.weights,
    ).to(device)
    amp = args.amp and device.type == "cuda"  # CPU 调试使用 FP32。
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    source = str(weights) if args.pretrained else "random_initialization"
    if args.weights is not None:
        source = {"path": str(args.weights), "sha256": file_sha256(args.weights)}
    metadata = {
        "format_version": 1,
        "model_config": {"depth": args.depth, "batch_norm": args.batch_norm, "num_classes": 2},
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
    print("测试集不参与训练和最佳模型选择。", flush=True)

    best_accuracy, epoch = -1.0, 0
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
                writer.flush()
                row = {
                    "epoch": epoch, "stage": stage, "train": train_metrics, "val": val_metrics,
                    "lr": learning_rates, "seconds": time.perf_counter() - start,
                }
                with (args.output_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row) + "\n")
                if val_metrics["accuracy"] > best_accuracy:
                    best_accuracy = val_metrics["accuracy"]
                    save_checkpoint(args.output_dir / "best.pt", model, metadata, epoch, stage, val_metrics)
                save_checkpoint(args.output_dir / "last.pt", model, metadata, epoch, stage, val_metrics)
                print(
                    f"epoch={epoch} stage={stage} "
                    f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.2%} "
                    f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.2%}", flush=True,
                )
                scheduler.step()  # 每轮结束后更新学习率。
            del optimizer, scheduler  # 释放上一阶段的动量缓存。
    print(f"最佳验证准确率：{best_accuracy:.2%}；请用 test 命令验收 best.pt。", flush=True)


def test(args) -> int:
    device = select_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("请使用本脚本生成的 best.pt 或 last.pt")
    if file_sha256(args.split_file) != checkpoint["split_sha256"]:
        raise ValueError("测试划分与训练时不一致，请使用原 split.json")
    seed_everything(checkpoint["seed"])
    manifest, splits = load_split_manifest(args.data_root, args.split_file)
    if manifest["class_to_idx"] != checkpoint["class_to_idx"]:
        raise ValueError("训练与测试的类别映射不一致")
    config = checkpoint["model_config"]
    weights = OFFICIAL_WEIGHTS[(config["depth"], config["batch_norm"])]
    loader = make_loader(splits["test"], manifest["class_to_idx"], weights.transforms(), args, False, checkpoint["seed"])
    model = build_vgg(**config, pretrained=False, init_weights=False)
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
        "device": str(device),
        "min_accuracy": args.min_accuracy,
        "passed": passed,
    }
    destination = args.report or args.checkpoint.parent / "test_metrics.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("达到准确率目标" if passed else "尚未达到准确率目标；请依据验证集调整训练。")
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
    training.add_argument("--depth", type=int, choices=(11, 13, 16, 19), default=11)
    training.add_argument("--batch-norm", action="store_true")
    training.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    training.add_argument("--weights", type=Path, help="可选：官方1000类权重文件")
    training.add_argument("--head-epochs", type=int, default=3, help="只训练最后一层")
    training.add_argument("--finetune-epochs", type=int, default=12, help="解冻全网微调")
    training.add_argument("--head-lr", type=float, default=1e-3)
    training.add_argument("--backbone-lr", type=float, default=1e-4)
    training.add_argument("--weight-decay", type=float, default=1e-4)
    training.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    training.add_argument("--seed", type=int, default=42)
    testing.add_argument("--checkpoint", type=Path, required=True)
    testing.add_argument("--report", type=Path)
    testing.add_argument("--min-accuracy", type=float, default=0.95)
    args = parser.parse_args()
    if args.command in ("train", "test") and (args.batch_size < 1 or args.workers < 0):
        parser.error("batch-size 必须大于0，workers 不能小于0")
    if args.command == "train":
        if min(args.head_epochs, args.finetune_epochs) < 0 or args.head_epochs + args.finetune_epochs == 0:
            parser.error("训练轮数不能为负，且至少训练一轮")
        if min(args.head_lr, args.backbone_lr) <= 0 or args.weight_decay < 0:
            parser.error("学习率必须大于0，weight-decay 不能小于0")
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
