import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import PlantDiseaseClassifier


def main():
    # ----------------------- 配置参数 -----------------------
    data_dir = './plant-dataset'  # 数据集根目录
    batch_size = 32
    num_classes = 3
    learning_rate = 0.001
    num_epochs = 20
    input_size = 224
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------------- 数据预处理 -----------------------
    data_transforms = {
        # "train": transforms.Compose([
        #     transforms.Resize((input_size, input_size)),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.4717, 0.5892, 0.3972],
        #                          [0.1704, 0.1531, 0.1755])
        # ]),
        "train": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.4717, 0.5892, 0.3972],
                                 [0.1704, 0.1531, 0.1755])
        ]),
        # "val": transforms.Compose([
        #     transforms.Resize((input_size, input_size)),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.4717, 0.5892, 0.3972],
        #                          [0.1704, 0.1531, 0.1755])
        # ])
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.4717, 0.5892, 0.3972],
                                 [0.1704, 0.1531, 0.1755])
        ]),

    }

    # ----------------------- 加载数据集 -----------------------
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    train_dataset = datasets.ImageFolder(
        train_dir, transform=data_transforms["train"])
    val_dataset = datasets.ImageFolder(
        val_dir, transform=data_transforms["val"])

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # ----------------------- 初始化模型 -----------------------
    model = PlantDiseaseClassifier(num_classes=num_classes).to(device)

    # ----------------------- 损失函数与优化器 -----------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # ----------------------- 训练与验证 -----------------------
    best_acc = 0.0
    save_path = "mobilenetv3_best.pth"

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

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {epoch_loss:.4f} "
              f"Train Acc: {epoch_acc:.4f} "
              f"Val Acc: {val_acc:.4f}")

        # 保存最优模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)

    print("训练完成，最优验证准确率：{:.4f}".format(best_acc))


if __name__ == "__main__":
    main()
