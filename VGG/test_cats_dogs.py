"""离线流程测试；合成图片不用于证明猫狗准确率。"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from argparse import Namespace
from unittest.mock import patch

import torch
from PIL import Image

from .image_dataset_loader import (
    ImageListDataset, build_classification_transforms, build_dataloaders,
    dogs_vs_cats_label, load_split_manifest, preprocessing_from_weights, save_split_manifest,
)
from .model_factory import build_model, get_classifier, get_model_weights
from .se_resnet import Bottleneck, ResNet50, SEBlock, build_resnet50
from .train_cats_dogs import EarlyStopping, configure_stage, parse_args, run_epoch
from .vgg_general import build_vgg


class CatsDogsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="vgg-test-")
        self.root = Path(self.temporary.name)
        self.images = self.root / "images"
        self.images.mkdir()
        for label, offset in (("cat", 0), ("dog", 100)):
            for index in range(6):
                Image.new("RGB", (64, 64), (offset + index * 10, 20, 80)).save(self.images / f"{label}.{index}.png")
        self.split = self.root / "split.json"

    def tearDown(self):
        self.temporary.cleanup()

    def prepare(self):
        return save_split_manifest(self.images, self.split, val_ratio=0.25, test_ratio=0.25)

    def test_split_excludes_bad_and_duplicate_images(self):
        duplicate = self.images / "cat.duplicate.png"
        duplicate.write_bytes((self.images / "cat.0.png").read_bytes())
        (self.images / "dog.broken.jpg").write_bytes(b"not an image")
        manifest = self.prepare()
        self.assertEqual(len(manifest["excluded"]), 2)
        self.assertEqual([len(rows) for rows in manifest["splits"].values()], [4, 4, 4])
        _, splits = load_split_manifest(self.images, self.split)
        paths = [path for rows in splits.values() for path, _ in rows]
        self.assertEqual(len(set(paths)), 12)
        with self.assertRaises(FileExistsError):
            self.prepare()  # 不覆盖固定划分。
        (self.images / "cat.0.png").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "图片内容已变化"):
            load_split_manifest(self.images, self.split)

    def test_conflicting_labels_are_rejected(self):
        (self.images / "dog.copy.png").write_bytes((self.images / "cat.0.png").read_bytes())
        with self.assertRaisesRegex(ValueError, "标签冲突"):
            self.prepare()

    def test_freezing_and_unfreezing(self):
        from argparse import Namespace

        args = Namespace(head_lr=1e-3, backbone_lr=1e-4, weight_decay=1e-4)
        with torch.device("meta"):
            model = build_vgg(num_classes=2)
        configure_stage(model, "head", args)
        trainable = {name for name, p in model.named_parameters() if p.requires_grad}
        self.assertEqual(trainable, {"classifier.6.weight", "classifier.6.bias"})
        optimizer = configure_stage(model, "finetune", args)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        self.assertEqual([g["lr"] for g in optimizer.param_groups], [1e-4, 1e-4, 1e-3])

    def test_early_stopping_patience(self):
        stopper = EarlyStopping(patience=3)
        self.assertEqual(stopper.update(0.8), (True, False))
        self.assertEqual(stopper.update(0.8), (False, False))
        self.assertEqual(stopper.update(0.79), (False, False))
        self.assertEqual(stopper.update(0.78), (False, True))
        self.assertEqual(stopper.update(0.81), (True, False))
        self.assertEqual(stopper.bad_epochs, 0)

    def test_full_training_checkpoint_and_tensorboard(self):
        for name, flags in (
            ("vgg", []),
            ("resnet50", ["--model", "resnet50"]),
            ("se_resnet50", ["--model", "resnet50", "--use-se", "--se-reduction", "8"]),
        ):
            with self.subTest(model=name):
                self._check_full_training(name, flags)

    def _check_full_training(self, name, flags):
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        if not self.split.exists():
            self.prepare()
        output = self.root / name
        env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
        common = ["--data-root", str(self.images), "--split-file", str(self.split)]
        result = subprocess.run(
            [sys.executable, "-B", "-m", "VGG.train_cats_dogs", "train", *common,
             "--output-dir", str(output), "--device", "cpu", "--batch-size", "3", "--workers", "0",
             "--head-epochs", "1", "--finetune-epochs", "1", "--no-pretrained", *flags],
            env=env, capture_output=True, text=True, timeout=240,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        rows = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
        self.assertEqual([row["stage"] for row in rows], ["head", "finetune"])
        self.assertTrue(all(row["train"]["total"] == row["val"]["total"] == 4 for row in rows))
        events = EventAccumulator(str(output / "tensorboard")).Reload()
        for tag in ("Loss/train", "Loss/val", "Accuracy/train", "Accuracy/val"):
            self.assertEqual([event.step for event in events.Scalars(tag)], [1, 2])
        self.assertTrue((output / "best.pt").is_file())
        self.assertTrue((output / "last.pt").is_file())
        checkpoint = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
        self.assertEqual(checkpoint["format_version"], 2)
        self.assertEqual(checkpoint["model_config"]["architecture"], "vgg" if name == "vgg" else "resnet50")
        self.assertEqual(checkpoint["preprocessing"]["resize_size"], 256 if name == "vgg" else 232)
        if name == "se_resnet50":
            self.assertTrue(checkpoint["model_config"]["use_se"])
            self.assertEqual(checkpoint["model_config"]["se_reduction"], 8)
        del checkpoint

        result = subprocess.run(
            [sys.executable, "-B", "-m", "VGG.train_cats_dogs", "test", *common,
             "--checkpoint", str(output / "best.pt"), "--device", "cpu",
             "--batch-size", "3", "--workers", "0", "--min-accuracy", "1"],
            env=env, capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        report = json.loads((output / "test_metrics.json").read_text())
        self.assertFalse(report["passed"])  # >100% 永远不能通过。
        self.assertFalse(report["pretrained"])
        self.assertEqual(report["total"], 4)
        self.assertEqual(sum(map(sum, report["confusion_matrix"])), 4)
        self.assertEqual(report["accuracy"], report["correct"] / report["total"])
        self.assertIn(
            f"测试准确率：{report['accuracy']:.4%}（正确 {report['correct']}/{report['total']} 张）",
            result.stdout,
        )
        self.assertIn(f"测试损失：{report['loss']:.6f}", result.stdout)
        self.assertIn("阈值检查（准确率 > 100.00%）：未通过", result.stdout)

        if name == "vgg":
            # 将同一权重保存成历史格式，实际执行旧检查点恢复与测试。
            checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
            checkpoint["format_version"] = 1
            checkpoint["model_config"].pop("architecture")
            checkpoint.pop("preprocessing")
            torch.save(checkpoint, output / "best.pt")
            del checkpoint
            legacy = subprocess.run(result.args, env=env, capture_output=True, text=True, timeout=120)
            self.assertEqual(legacy.returncode, 2, legacy.stdout + legacy.stderr)
            legacy_report = json.loads((output / "test_metrics.json").read_text())
            self.assertEqual(legacy_report["accuracy"], report["accuracy"])
            self.assertEqual(legacy_report["loss"], report["loss"])
        # VGG 权重较大；每个模型检查完成后及时回收临时磁盘空间。
        (output / "best.pt").unlink()
        (output / "last.pt").unlink()

    def test_preprocessing_and_general_loader_for_both_families(self):
        for architecture in ("vgg", "resnet50"):
            with self.subTest(architecture=architecture):
                weights = get_model_weights({"architecture": architecture})
                preprocessing = preprocessing_from_weights(weights)
                train_transform, eval_transform = build_classification_transforms(preprocessing)
                image = Image.new("RGB", (320, 280), (23, 80, 200))
                torch.testing.assert_close(eval_transform(image), weights.transforms()(image))
                self.assertEqual(train_transform(image).shape, (3, 224, 224))
                gray = self.root / "gray.png"
                Image.new("L", (80, 60)).save(gray)
                dataset = ImageListDataset([(gray, "cat")], {"cat": 0}, eval_transform)
                self.assertEqual(dataset[0][0].shape, (3, 224, 224))
                bundle = build_dataloaders(
                    self.images, dogs_vs_cats_label, batch_size=2, num_workers=0,
                    val_ratio=0.25, test_ratio=0.25, preprocessing=preprocessing,
                )
                for loader in (bundle.train_loader, bundle.val_loader, bundle.test_loader):
                    self.assertEqual(next(iter(loader))[0].shape, (2, 3, 224, 224))

    def test_cli_model_options(self):
        valid = [
            ([], "vgg", False),
            (["--depth", "19", "--batch-norm"], "vgg", False),
            (["--model", "resnet50"], "resnet50", False),
            (["--model", "resnet50", "--use-se", "--se-reduction", "8"], "resnet50", True),
            (["--model", "resnet50", "--no-use-se"], "resnet50", False),
        ]
        for flags, name, use_se in valid:
            with patch.object(sys, "argv", ["train", "train", "--data-root", str(self.images), *flags]):
                args = parse_args()
            self.assertEqual((args.model, args.use_se), (name, use_se))
        invalid = [
            ["--use-se"], ["--model", "resnet50", "--batch-norm"],
            ["--model", "resnet50", "--depth", "11"],
            ["--model", "resnet50", "--se-reduction", "8"],
            ["--model", "resnet50", "--use-se", "--se-reduction", "0"],
        ]
        for flags in invalid:
            with patch.object(sys, "argv", ["train", "train", "--data-root", str(self.images), *flags]), \
                    patch("sys.stderr"), self.assertRaises(SystemExit) as error:
                parse_args()
            self.assertEqual(error.exception.code, 2)


class ResNetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_standard_resnet_matches_torchvision_and_pretrained_loading(self):
        from torchvision.models import resnet50, ResNet50_Weights

        reference = resnet50(weights=None).eval()
        # 离线使用 torchvision 构建的权重验证布局及加载路径，不下载权重。
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resnet50.pth"
            torch.save(reference.state_dict(), path)
            model = build_resnet50(pretrained=True, weights_path=path).eval()
            self.assertEqual([len(getattr(model, f"layer{i}")) for i in range(1, 5)], [3, 4, 6, 3])
            self.assertEqual(sum(p.numel() for p in model.parameters()), 25557032)
            images = torch.randn(2, 3, 64, 64)
            with torch.inference_mode():
                torch.testing.assert_close(model(images), reference(images), rtol=0, atol=0)
            del model
            with patch.object(ResNet50_Weights, "get_state_dict", return_value=reference.state_dict()) as fetch:
                model = build_resnet50(num_classes=2, pretrained=True, use_se=True, se_reduction=8)
            fetch.assert_called_once_with(progress=True, check_hash=True)
            self.assertEqual(sum(isinstance(m, SEBlock) for m in model.modules()), 16)
            torch.testing.assert_close(model.layer4[2].conv3.weight, reference.layer4[2].conv3.weight)
            self.assertEqual(model.layer4[2].se.fc1.out_channels, 256)
            self.assertEqual(model(images).shape, (2, 2))
            del model
            # 官方骨干加载必须严格，不能遗漏 BN/卷积后仍声称加载成功。
            broken = dict(reference.state_dict())
            broken.pop("conv1.weight")
            with patch.object(ResNet50_Weights, "get_state_dict", return_value=broken), \
                    self.assertRaises(RuntimeError):
                build_resnet50(pretrained=True, use_se=True)

    def test_se_channel_scaling_and_gradient(self):
        block = SEBlock(4, reduction=16)  # 小通道数也不产生零通道卷积。
        with torch.no_grad():
            block.fc2.weight.zero_()
            block.fc2.bias.zero_()
        x = torch.randn(2, 4, 7, 9, requires_grad=True)
        torch.testing.assert_close(block(x), x * 0.5)
        block(x).sum().backward()
        torch.testing.assert_close(x.grad, torch.full_like(x, 0.5))
        self.assertIsNotNone(block.fc2.bias.grad)
        with self.assertRaises(ValueError):
            SEBlock(4, reduction=0)

    def test_residual_se_position_and_downsample(self):
        downsample = torch.nn.Sequential(torch.nn.Conv2d(8, 16, 1, stride=2, bias=False), torch.nn.BatchNorm2d(16))
        block = Bottleneck(8, 4, stride=2, downsample=downsample, use_se=True).eval()
        x = torch.randn(2, 8, 16, 16)
        with torch.no_grad():
            block.bn3.weight.zero_()
            block.bn3.bias.zero_()
            # 残差分支为0时，输出必须等于 shortcut 的 ReLU，不能被 SE 缩放。
            torch.testing.assert_close(block(x), torch.relu(downsample(x)))

    def test_freezing_bn_optimizer_coverage_and_se_updates(self):
        args = Namespace(head_lr=1e-2, backbone_lr=1e-2, weight_decay=0.0)
        for use_se in (False, True):
            with self.subTest(use_se=use_se):
                model = build_model(architecture="resnet50", num_classes=2, use_se=use_se)
                optimizer = configure_stage(model, "head", args)
                trainable = {name for name, p in model.named_parameters() if p.requires_grad}
                self.assertEqual(trainable, {"fc.weight", "fc.bias"})
                mean = model.bn1.running_mean.clone()
                conv = model.conv1.weight.detach().clone()
                head = model.fc.weight.detach().clone()
                loader = [(torch.randn(2, 3, 64, 64), torch.tensor([0, 1]))]
                scaler = torch.amp.GradScaler("cuda", enabled=False)
                metrics = run_epoch(model, loader, torch.device("cpu"), optimizer, scaler, head_only=True)
                self.assertEqual(metrics["total"], 2)
                self.assertFalse(model.bn1.training)
                self.assertTrue(get_classifier(model).training)
                torch.testing.assert_close(model.bn1.running_mean, mean, rtol=0, atol=0)
                torch.testing.assert_close(model.conv1.weight, conv, rtol=0, atol=0)
                self.assertFalse(torch.equal(head, model.fc.weight))

                optimizer = configure_stage(model, "finetune", args)
                self.assertTrue(all(p.requires_grad for p in model.parameters()))
                ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
                self.assertEqual(len(ids), len(set(ids)))
                self.assertEqual(set(ids), {id(p) for p in model.parameters()})
                if use_se:
                    se = model.layer1[0].se.fc2.weight.detach().clone()
                run_epoch(model, loader, torch.device("cpu"), optimizer, scaler)
                self.assertTrue(model.bn1.training)
                self.assertFalse(torch.equal(model.bn1.running_mean, mean))
                self.assertFalse(torch.equal(model.conv1.weight, conv))
                if use_se:
                    self.assertFalse(torch.equal(model.layer1[0].se.fc2.weight, se))
                    self.assertTrue(torch.isfinite(model.layer1[0].se.fc2.weight.grad).all())
                mean = model.bn1.running_mean.clone()
                run_epoch(model, loader, torch.device("cpu"))
                torch.testing.assert_close(model.bn1.running_mean, mean, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
