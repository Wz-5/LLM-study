

from __future__ import annotations

import random
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
    """适配 ImageFolder 风格：root/cat/a.jpg -> cat。"""
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


def build_train_transform(image_size: int = 224) -> transforms.Compose:
    """训练集：随机裁剪和翻转用于数据增强。"""
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_eval_transform(image_size: int = 224) -> transforms.Compose:
    """验证/测试集：使用确定性预处理，保证指标可重复。"""
    resize_size = round(image_size / 0.875)
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


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

    train_dataset = ImageListDataset(
        train_records,
        class_to_idx,
        transform=build_train_transform(image_size),
    )
    val_dataset = ImageListDataset(
        val_records,
        class_to_idx,
        transform=build_eval_transform(image_size),
    )
    test_dataset = ImageListDataset(
        test_records,
        class_to_idx,
        transform=build_eval_transform(image_size),
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
