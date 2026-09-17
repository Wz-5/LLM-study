

from __future__ import annotations

import random
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


ImageRecord = Tuple[Path, str]
LabelParser = Callable[[Path], str]

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def dogs_vs_cats_label(path: Path) -> str:
    """从 Kaggle Dogs vs. Cats 文件名提取标签。

    cat.123.jpg -> cat
    dog.456.jpg -> dog
    """
    label = path.stem.split(".")[0].lower()
    if label not in {"cat", "dog"}:
        raise ValueError(f"不能从文件名提取猫狗标签：{path.name}")
    return label


def parent_folder_label(path: Path) -> str:
    """适配 ImageFolder 风格root/cat/a.jpg -> cat。"""
    return path.parent.name


def discover_labeled_images(
    image_root: str | Path,
    label_parser: LabelParser,
) -> List[ImageRecord]:
    """递归扫描图片，并用传入的解析器获得每张图片的标签。"""
    image_root = Path(image_root)
    if not image_root.is_dir():
        raise FileNotFoundError(f"图片目录不存在：{image_root}")

    records: List[ImageRecord] = []
    for path in sorted(image_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            records.append((path, label_parser(path)))

    if not records:
        raise RuntimeError(f"目录中没有找到图片：{image_root}")

    return records


def stratified_split(
    records: Sequence[ImageRecord],
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[ImageRecord], List[ImageRecord], List[ImageRecord]]:
    """按类别分层划分，尽量保持各集合中的类别比例一致。"""
    if val_ratio < 0 or test_ratio < 0:
        raise ValueError("val_ratio 和 test_ratio 不能小于0")
    if val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio + test_ratio 必须小于1")

    grouped: Dict[str, List[ImageRecord]] = defaultdict(list)
    for record in records:
        grouped[record[1]].append(record)

    rng = random.Random(seed)
    train_records: List[ImageRecord] = []
    val_records: List[ImageRecord] = []
    test_records: List[ImageRecord] = []

    for label, label_records in sorted(grouped.items()):
        label_records = list(label_records)
        rng.shuffle(label_records)

        count = len(label_records)
        val_count = round(count * val_ratio)
        test_count = round(count * test_ratio)

        if count - val_count - test_count <= 0:
            raise ValueError(f"类别 {label!r} 的图片太少，无法按当前比例划分")

        val_records.extend(label_records[:val_count])
        test_records.extend(label_records[val_count:val_count + test_count])
        train_records.extend(label_records[val_count + test_count:])

    # 再次打乱，避免不同类别在列表中成块排列。
    rng.shuffle(train_records)
    rng.shuffle(val_records)
    rng.shuffle(test_records)
    return train_records, val_records, test_records


class ImageListDataset(Dataset):
    """从统一的 (图片路径, 类别名称) 记录中读取分类数据。"""

    def __init__(
        self,
        records: Sequence[ImageRecord],
        class_to_idx: Dict[str, int],
        transform: Optional[Callable] = None,
    ) -> None:
        self.records = list(records)
        self.class_to_idx = dict(class_to_idx)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        image_path, class_name = self.records[index]

        # convert("RGB") 保证灰度图、RGBA图也统一变成三通道。
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)

        label = self.class_to_idx[class_name]
        return image, label


def build_train_transform(
    image_size: int = 224, *, mean=IMAGENET_MEAN, std=IMAGENET_STD,
    interpolation=transforms.InterpolationMode.BILINEAR, antialias=True,
) -> transforms.Compose:
    """训练集：随机裁剪和翻转用于数据增强。"""
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0),
                                         interpolation=interpolation, antialias=antialias),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def build_eval_transform(
    image_size: int = 224, *, resize_size=None, mean=IMAGENET_MEAN, std=IMAGENET_STD,
    interpolation=transforms.InterpolationMode.BILINEAR, antialias=True,
) -> transforms.Compose:
    """验证/测试集：使用确定性预处理，保证指标可重复。"""
    resize_size = round(image_size / 0.875) if resize_size is None else resize_size
    return transforms.Compose(
        [
            transforms.Resize(resize_size, interpolation=interpolation, antialias=antialias),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def preprocessing_from_weights(weights) -> dict:
    """保存预训练权重的输入约定，测试时无需下载权重即可复现。"""
    preset = weights.transforms()
    return {
        "image_size": preset.crop_size[0],
        "resize_size": preset.resize_size[0],
        "mean": list(preset.mean),
        "std": list(preset.std),
        "interpolation": preset.interpolation.value,
        "antialias": preset.antialias,
    }


def build_classification_transforms(preprocessing: dict):
    """训练、验证和独立测试共用同一模型的尺寸及归一化配置。"""
    options = dict(preprocessing)
    options["interpolation"] = transforms.InterpolationMode(options["interpolation"])
    evaluation = build_eval_transform(**options)
    options.pop("resize_size")
    return build_train_transform(**options), evaluation


def file_sha256(path: str | Path) -> str:
    """记录文件内容，防止训练后悄悄换数据。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_split_manifest(
    image_root: str | Path,
    manifest_path: str | Path,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict:
    """检查图片、去除完全重复文件、固定划分。"""
    root = Path(image_root).resolve()
    destination = Path(manifest_path)
    if destination.exists():
        raise FileExistsError(f"划分已存在，请复用或换一个路径：{destination}")
    if val_ratio <= 0 or test_ratio <= 0 or val_ratio + test_ratio >= 1:
        raise ValueError("验证和测试比例必须大于0，且两者之和小于1")

    records = discover_labeled_images(root, dogs_vs_cats_label)
    valid, excluded, hashes, seen = [], [], {}, {}
    for index, (path, label) in enumerate(records, 1):
        relative = path.relative_to(root).as_posix()
        try:
            with Image.open(path) as image:
                image.convert("RGB").load()  # 实际解码，提前发现坏图。
        except (OSError, ValueError) as error:
            excluded.append({"path": relative, "reason": str(error)})
            continue

        digest = file_sha256(path)
        if digest in seen:
            previous_path, previous_label = seen[digest]
            if label != previous_label:
                raise ValueError(f"相同图片的标签冲突：{path} 与 {previous_path}")
            excluded.append({"path": relative, "reason": "完全重复图片"})
            continue
        seen[digest] = (path, label)
        hashes[path] = digest
        valid.append((path, label))
        if index % 1000 == 0:
            print(f"已检查 {index}/{len(records)} 张图片", flush=True)

    splits = stratified_split(valid, val_ratio, test_ratio, seed)
    manifest = {
        "version": 1,
        "seed": seed,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "class_to_idx": {"cat": 0, "dog": 1},
        "excluded": excluded,
        "splits": {},
    }
    for name, split in zip(("train", "val", "test"), splits):
        if {label for _, label in split} != {"cat", "dog"}:
            raise ValueError(f"{name} 必须包含猫和狗，请增加图片数量")
        manifest["splits"][name] = [
            {"path": p.relative_to(root).as_posix(), "label": label, "sha256": hashes[p]}
            for p, label in split
        ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    return manifest


def load_split_manifest(
    image_root: str | Path,
    manifest_path: str | Path,
) -> Tuple[dict, Dict[str, List[ImageRecord]]]:
    """复用原划分，并检查文件是否被替换。"""
    root = Path(image_root).resolve()
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("version") != 1 or manifest.get("class_to_idx") != {"cat": 0, "dog": 1}:
        raise ValueError("不支持的划分文件或类别映射")
    seen_paths, seen_hashes, splits = set(), set(), {}
    for name in ("train", "val", "test"):
        records = []
        for item in manifest["splits"][name]:
            path = (root / item["path"]).resolve()
            if not path.is_relative_to(root):
                raise ValueError(f"图片路径超出数据目录：{path}")
            if item["label"] not in manifest["class_to_idx"]:
                raise ValueError(f"未知标签：{item['label']}")
            digest = file_sha256(path)
            if digest != item["sha256"]:
                raise ValueError(f"图片内容已变化，请检查：{path}")
            if path in seen_paths or digest in seen_hashes:
                raise ValueError(f"划分中存在重复图片：{path}")
            seen_paths.add(path)
            seen_hashes.add(digest)
            records.append((path, item["label"]))
        if {label for _, label in records} != {"cat", "dog"}:
            raise ValueError(f"{name} 必须同时包含猫和狗")
        splits[name] = records
    return manifest, splits


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: Optional[DataLoader]
    class_to_idx: Dict[str, int]


def build_dataloaders(
    image_root: str | Path,
    label_parser: LabelParser,
    batch_size: int = 32,
    num_workers: int = 4,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    image_size: int = 224,
    preprocessing: Optional[dict] = None,
) -> DataBundle:
    """从任意“图片目录 + 标签解析函数”建立三组 DataLoader。"""
    records = discover_labeled_images(image_root, label_parser)
    class_names = sorted({label for _, label in records})
    class_to_idx = {
        class_name: index
        for index, class_name in enumerate(class_names)
    }

    train_records, val_records, test_records = stratified_split(
        records,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    train_transform, eval_transform = (
        build_classification_transforms(preprocessing) if preprocessing is not None
        else (build_train_transform(image_size), build_eval_transform(image_size))
    )
    train_dataset = ImageListDataset(
        train_records,
        class_to_idx,
        transform=train_transform,
    )
    val_dataset = ImageListDataset(
        val_records,
        class_to_idx,
        transform=eval_transform,
    )
    test_dataset = ImageListDataset(
        test_records,
        class_to_idx,
        transform=eval_transform,
    )

    common_loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True,
        **common_loader_options,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        drop_last=False,
        **common_loader_options,
    )
    test_loader = (
        DataLoader(
            test_dataset,
            shuffle=False,
            drop_last=False,
            **common_loader_options,
        )
        if test_records
        else None
    )

    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        class_to_idx=class_to_idx,
    )


if __name__ == "__main__":
    # 解压 Kaggle 的 train.zip 后，把这里改成实际的 train 图片目录。
    IMAGE_ROOT = "data/dogs-vs-cats/raw/train"

    data = build_dataloaders(
        image_root=IMAGE_ROOT,
        label_parser=dogs_vs_cats_label,
        batch_size=32,
        num_workers=4,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=42,
        image_size=224,
    )

    images, labels = next(iter(data.train_loader))
    print("类别映射：", data.class_to_idx)
    print("训练批次图片尺寸：", images.shape)
    print("训练批次标签尺寸：", labels.shape)
