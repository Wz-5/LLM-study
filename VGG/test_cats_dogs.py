"""离线流程测试；合成图片不用于证明猫狗准确率。"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from .image_dataset_loader import load_split_manifest, save_split_manifest
from .train_cats_dogs import configure_stage
from .vgg_general import build_vgg


class CatsDogsTests(unittest.TestCase):
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

    def test_full_training_checkpoint_and_tensorboard(self):
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        self.prepare()
        output = self.root / "run"
        env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
        common = ["--data-root", str(self.images), "--split-file", str(self.split)]
        result = subprocess.run(
            [sys.executable, "-B", "-m", "VGG.train_cats_dogs", "train", *common,
             "--output-dir", str(output), "--device", "cpu", "--batch-size", "3", "--workers", "0",
             "--head-epochs", "1", "--finetune-epochs", "1", "--no-pretrained"],
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


if __name__ == "__main__":
    unittest.main()
