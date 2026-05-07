import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class FlowerClassifier(nn.Module):
    def __init__(self, num_classes=102):
        super(FlowerClassifier, self).__init__()
        self.base_model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        in_features = self.base_model.classifier[3].in_features
        self.base_model.classifier[3] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_flower = FlowerClassifier(num_classes=102).to(device)
    dummy_input = torch.randn(4, 3, 224, 224).to(device)
    output_flower = model_flower(dummy_input)
    print("花卉分类模型输出形状：", output_flower.shape)  # 应该是 [4, 102]
