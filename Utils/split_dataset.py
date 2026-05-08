import os
import random
import shutil
from pathlib import Path

# 你的原始大文件夹：里面直接放五个类别文件夹
src_root = Path(r"D:\shengyuqi\hatif_csq")

# 输出的新数据集目录
dst_root = Path(r"D:\shengyuqi\sjj")

# 验证集比例
val_ratio = 0.2

# 随机种子，保证每次划分一致
random_seed = 42

img_exts = [".jpg", ".jpeg", ".png"]

random.seed(random_seed)

train_root = dst_root / "train"
val_root = dst_root / "val"

train_root.mkdir(parents=True, exist_ok=True)
val_root.mkdir(parents=True, exist_ok=True)

for class_dir in src_root.iterdir():
    if not class_dir.is_dir():
        continue

    class_name = class_dir.name

    images = [
        p for p in class_dir.iterdir()
        if p.suffix.lower() in img_exts
    ]

    random.shuffle(images)

    val_num = int(len(images) * val_ratio)
    val_images = images[:val_num]
    train_images = images[val_num:]

    train_class_dir = train_root / class_name
    val_class_dir = val_root / class_name

    train_class_dir.mkdir(parents=True, exist_ok=True)
    val_class_dir.mkdir(parents=True, exist_ok=True)

    for img_path in train_images:
        shutil.copy2(img_path, train_class_dir / img_path.name)

    for img_path in val_images:
        shutil.copy2(img_path, val_class_dir / img_path.name)

    print(f"{class_name}: train={len(train_images)}, val={len(val_images)}")

print("划分完成！")