import torch
from torchvision import transforms
from PIL import Image
from model import PlantDiseaseClassifier  # 注意根据你的 model.py 修改导入名称

# 1. 配置设备和模型路径
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "mobilenetv3_best.pth"  # 保存的模型路径
image_path = "/root/autodl-tmp/mobileNetV3-plant/plant-dataset/test/rust/82f49a4a7b9585f1.jpg"  # 替换为你要测试的图像路径

# 2. 定义类别名
class_names = ["healthy", "powdery", "rust"]

# 3. 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# 4. 加载并处理图像
image = Image.open(image_path).convert("RGB")
image_tensor = transform(image).unsqueeze(0).to(device)  # 添加 batch 维度

# 5. 加载模型
model = PlantDiseaseClassifier(num_classes=len(class_names)).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# 6. 执行推理
with torch.no_grad():
    outputs = model(image_tensor)
    _, predicted = torch.max(outputs, 1)
    predicted_class = class_names[predicted.item()]

print(f"预测结果: {predicted_class}")
