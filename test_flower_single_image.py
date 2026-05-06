import os
import sys
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from model import FlowerClassifier


def test_flower_image(image_path, model_path="mobilenetv3_flowers_best.pth"):
    """
    测试单张花卉图像
    
    Args:
        image_path: 图像路径
        model_path: 模型权重文件路径
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ----------------------- 检查文件存在性 -----------------------
    if not os.path.exists(image_path):
        print(f"❌ 错误: 找不到图像文件: {image_path}")
        return
    
    if not os.path.exists(model_path):
        print(f"❌ 错误: 找不到模型文件: {model_path}")
        print(f"💡 请先运行 train_flowers.py 训练模型")
        return
    
    # ----------------------- 加载图像 -----------------------
    try:
        image = Image.open(image_path).convert('RGB')
        print(f"✅ 成功加载图像: {image_path}\n")
    except Exception as e:
        print(f"❌ 加载图像失败: {e}")
        return
    
    # ----------------------- 图像预处理 -----------------------
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # ----------------------- 加载模型 -----------------------
    model = FlowerClassifier(num_classes=17).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # ----------------------- 推理预测 -----------------------
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        top5_probs, top5_indices = torch.topk(probabilities, 5, dim=1)
    
    # ----------------------- 输出结果 -----------------------
    print("=" * 50)
    print("🌸 花卉分类预测结果")
    print("=" * 50)
    print(f"图像路径: {image_path}\n")
    
    print("Top-5 预测结果:")
    for i, (prob, idx) in enumerate(zip(top5_probs[0], top5_indices[0])):
        print(f"  {i+1}. 类别 {idx.item()}: {prob.item():.4f}")
    
    print("=" * 50)


if __name__ == "__main__":
    # 处理命令行参数
    if len(sys.argv) < 2:
        print("❌ 使用方法: python test_flower_single_image.py <image_path> [model_path]")
        print("📝 示例: python test_flower_single_image.py ./flower.jpg")
        print("📝 或: python test_flower_single_image.py ./flower.jpg ./mobilenetv3_flowers_best.pth")
        sys.exit(1)
    
    image_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else "mobilenetv3_flowers_best.pth"
    
    test_flower_image(image_path, model_path)
