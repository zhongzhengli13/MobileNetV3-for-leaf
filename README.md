# Flower Classification using MobileNetV3 (PyTorch)

> **项目说明：**
> 本项目基于 MobileNetV3 轻量级模型，用于 **102 种花卉分类**任务。

使用轻量级模型 **MobileNetV3-Small** 实现 **102 种花卉分类**，数据集来自 Oxford 102 Category Flower Dataset。

类别名称映射文件：`data/cat_to_name.json`，包含 102 种花卉的中英文名称。

------

## 快速开始

### 1. 训练模型

```bash
python train_flowers.py
```

**配置参数说明** （在 `train_flowers.py` 中修改）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `data_dir` | `./data/dataset` | 数据集根目录 |
| `batch_size` | 32 | 批大小 |
| `num_classes` | 102 | 花卉类别数 |
| `learning_rate` | 0.001 | 学习率 |
| `num_epochs` | 30 | 训练轮数 |

训练输出示例：
```
使用设备: cuda
数据集路径: ./data/dataset
分类类别数: 102
训练集样本数: 6552
验证集样本数: 818
Epoch [1/30] Train Loss: 4.6253 Train Acc: 0.0180 Val Acc: 0.0355 LR: 0.001000
...
保存最优模型, 验证准确率: 0.8500
训练完成，最优验证准确率：0.8500
```

### 2. 测试单张图像

```bash
python test_flower_single_image.py <image_path> [model_path]
```

**示例：**
```bash
python test_flower_single_image.py ./data/dataset/test/image_06734.jpg
python test_flower_single_image.py ./my_flower.jpg ./mobilenetv3_flowers_best.pth
```

**输出示例：**
```
成功加载图像: ./data/dataset/test/image_06734.jpg

==================================================
花卉分类预测结果
==================================================
图像路径: ./data/dataset/test/image_06734.jpg

Top-5 预测结果:
  1. sunflower (类别 54): 0.9654
  2. daisy (类别 21): 0.0246
  3. buttercup (类别 5): 0.0089
  4. colt's foot (类别 33): 0.0008
  5. pink primrose (类别 1): 0.0003
==================================================
```

------

## 数据集准备

### 目录结构

```
data/
├── cat_to_name.json        # 类别ID到花卉名称的映射
├── sample_submission.csv   # 测试集提交模板
├── README.md               # 数据集说明
└── dataset/
    ├── train/              # 训练集 (6552 张)
    │   ├── 1/
    │   ├── 2/
    │   ├── ...
    │   └── 102/
    ├── valid/              # 验证集 (818 张)
    │   ├── 1/
    │   ├── 2/
    │   ├── ...
    │   └── 102/
    └── test/               # 测试集 (819 张, 无标签)
        ├── image_06734.jpg
        ├── image_06735.jpg
        └── ...
```

数据集已包含在 `data/` 目录中，无需额外下载。

------

## 模型架构

### FlowerClassifier
- 基础模型：MobileNetV3-Small (torchvision)
- 类别数：102
- 预训练：是（使用 ImageNet 预训练权重）
- 参数量：约 1.5M

------

## 图像预处理

### 训练集预处理
```python
transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
])
```

### 验证集/测试集预处理
```python
transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
])
```

------

## 关键特性

- MobileNetV3-Small 轻量级模型
- 支持 102 种花卉分类
- 完整的数据增强策略
- 余弦退火学习率调度
- 自动保存最优模型
- 支持 GPU/CPU 训练
- 单张图像快速推理
- Top-5 预测结果展示（含花卉名称）

------

## 项目结构

```
MobileNetV3-for-leaf/
├── model.py                    # 模型定义 (FlowerClassifier)
├── train_flowers.py            # 花卉分类训练脚本
├── test_flower_single_image.py # 单张图像测试脚本
├── data/                       # 数据集 (102类花卉)
│   ├── cat_to_name.json        # 类别名称映射
│   └── dataset/
│       ├── train/              # 训练集
│       ├── valid/              # 验证集
│       └── test/               # 测试集
└── README.md                   # 项目文档
```

------

## 性能提示

- 使用预训练权重可显著提升收敛速度和最终精度
- 充分的数据增强有助于提高模型泛化能力
- 建议使用 GPU 训练以加快速度
- 调整 `batch_size` 根据显存大小灵活选择

------

## 项目作者

**李中政**
GitHub: [zhongzhengli13](https://github.com/zhongzhengli13)
