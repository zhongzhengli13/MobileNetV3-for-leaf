import os
import sys
import json
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from PIL import Image
from model import FlowerClassifier


def load_cat_to_name(path="data/cat_to_name.json"):
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None


def build_idx_to_class(train_dir="data/dataset/train"):
    """通过 ImageFolder 获取 idx -> 文件夹名 的正确映射"""
    dummy_dataset = datasets.ImageFolder(train_dir)
    return {v: k for k, v in dummy_dataset.class_to_idx.items()}


def test_flower_image(image_path, model_path="mobilenetv3_flowers_best.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(image_path):
        print(f"错误: 找不到图像文件: {image_path}")
        return

    if not os.path.exists(model_path):
        print(f"错误: 找不到模型文件: {model_path}")
        print(f"请先运行 train_flowers.py 训练模型")
        return

    try:
        image = Image.open(image_path).convert('RGB')
        print(f"成功加载图像: {image_path}\n")
    except Exception as e:
        print(f"加载图像失败: {e}")
        return

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ])

    image_tensor = transform(image).unsqueeze(0).to(device)

    cat_to_name = load_cat_to_name()
    idx_to_class = build_idx_to_class()

    model = FlowerClassifier(num_classes=102).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        top5_probs, top5_indices = torch.topk(probabilities, 5, dim=1)

    print("=" * 50)
    print("花卉分类预测结果")
    print("=" * 50)
    print(f"图像路径: {image_path}\n")

    print("Top-5 预测结果:")
    for i, (prob, idx) in enumerate(zip(top5_probs[0], top5_indices[0])):
        folder_name = idx_to_class.get(idx.item(), str(idx.item()))
        if cat_to_name and folder_name in cat_to_name:
            flower_name = cat_to_name[folder_name]
            print(f"  {i+1}. {flower_name} (类别 {folder_name}): {prob.item():.4f}")
        else:
            print(f"  {i+1}. 类别 {folder_name}: {prob.item():.4f}")

    print("=" * 50)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方法: python test_flower_single_image.py <image_path> [model_path]")
        print("示例: python test_flower_single_image.py ./flower.jpg")
        print("或: python test_flower_single_image.py ./flower.jpg ./mobilenetv3_flowers_best.pth")
        sys.exit(1)

    image_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else "mobilenetv3_flowers_best.pth"

    test_flower_image(image_path, model_path)
