import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import FlowerClassifier


def main():
    # ----------------------- 配置参数 -----------------------
    data_dir = './data/dataset'  # 花卉数据集根目录
    batch_size = 32
    num_classes = 102  # 102种花卉
    learning_rate = 0.001
    num_epochs = 30  # 可增加更多epoch
    input_size = 224
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"🎯 使用设备: {device}")
    print(f"📊 数据集路径: {data_dir}")
    print(f"🌸 分类类别数: {num_classes}")

    # ----------------------- 数据预处理 -----------------------
    # 使用ImageNet的均值和标准差
    data_transforms = {
        "train": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),  # 花卉可以上下翻转
            transforms.RandomRotation(20),    # 随机旋转
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 颜色抖动
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet标准
                                 [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
    }

    # ----------------------- 加载数据集 -----------------------
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "valid")

    # 检查数据集是否存在
    if not os.path.exists(train_dir):
        print(f"❌ 错误: 找不到训练集路径: {train_dir}")
        return
    if not os.path.exists(val_dir):
        print(f"❌ 错误: 找不到验证集路径: {val_dir}")
        return

    train_dataset = datasets.ImageFolder(
        train_dir, transform=data_transforms["train"])
    val_dataset = datasets.ImageFolder(
        val_dir, transform=data_transforms["val"])

    print(f"📷 训练集样本数: {len(train_dataset)}")
    print(f"📷 验证集样本数: {len(val_dataset)}")
    print(f"🏷️  类别标签: {train_dataset.classes}")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # ----------------------- 初始化模型 -----------------------
    model = FlowerClassifier(num_classes=num_classes).to(device)
    print(f"✅ 模型已初始化，参数数量: {sum(p.numel() for p in model.parameters())}")

    # ----------------------- 损失函数与优化器 -----------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # ----------------------- 训练与验证 -----------------------
    best_acc = 0.0
    save_path = "mobilenetv3_flowers_best.pth"

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = correct.double() / len(train_dataset)

        # ----------------------- 验证模型 -----------------------
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images, val_labels = val_images.to(
                    device), val_labels.to(device)
                val_outputs = model(val_images)
                _, val_preds = torch.max(val_outputs, 1)
                val_correct += torch.sum(val_preds == val_labels.data)

        val_acc = val_correct.double() / len(val_dataset)
        scheduler.step()

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {epoch_loss:.4f} "
              f"Train Acc: {epoch_acc:.4f} "
              f"Val Acc: {val_acc:.4f} "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")

        # 保存最优模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"🎉 保存最优模型, 验证准确率: {val_acc:.4f}")

    print(f"✅ 训练完成，最优验证准确率：{best_acc:.4f}")
    print(f"📁 模型已保存到: {save_path}")


if __name__ == "__main__":
    main()
