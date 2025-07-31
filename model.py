import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small
from torchsummary import summary


class PlantDiseaseClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(PlantDiseaseClassifier, self).__init__()
        self.base_model = mobilenet_v3_small(pretrained=False)
        in_features = self.base_model.classifier[3].in_features
        self.base_model.classifier[3] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PlantDiseaseClassifier(num_classes=3).to(device)

    # ✅ 把 dummy_input 移动到相同 device 上
    dummy_input = torch.randn(4, 3, 224, 224).to(device)

    # 测试 forward
    output = model(dummy_input)
    print("\n✅ 输出形状：", output.shape)  # 应该是 [4, 3]

    # 模型结构 summary
    summary(model, (3, 4000, 2672), device=str(device))
