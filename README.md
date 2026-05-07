
# 🌸 Flower Classification using MobileNetV3 (PyTorch) - 花卉分类

> **项目说明：** 
> 本项目基于 MobileNetV3 轻量级模型，用于 **17 种花卉分类**任务。

使用轻量级模型 **MobileNetV3-Small** 实现 **17 种花卉分类**，数据集来自 Kaggle：[17-Category Flowers Dataset](https://www.kaggle.com/datasets/saidakbarp/17-category-flowers)

支持的花卉类别：
- Tulip（郁金香）
- Sunflower（向日葵）
- Rose（玫瑰）
- Daisy（雏菊）
- Daffodil（水仙花）
- ...等共 17 类

------

## 🚀 快速开始

### 1. 训练模型

```bash
python train_flowers.py
```

**配置参数说明** （在 `train_flowers.py` 中修改）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `data_dir` | `./flowers-dataset` | 数据集根目录 |
| `batch_size` | 32 | 批大小 |
| `num_classes` | 17 | 花卉类别数 |
| `learning_rate` | 0.001 | 学习率 |
| `num_epochs` | 30 | 训练轮数 |

训练输出示例：
```
🎯 使用设备: cuda
📊 数据集路径: ./flowers-dataset
🌸 分类类别数: 17
📷 训练集样本数: 3670
📷 验证集样本数: 918
Epoch [1/30] Train Loss: 2.8342 Train Acc: 0.0891 Val Acc: 0.0850 LR: 0.001000
Epoch [2/30] Train Loss: 2.4156 Train Acc: 0.2340 Val Acc: 0.2610 LR: 0.000999
...
🎉 保存最优模型, 验证准确率: 0.9234
✅ 训练完成，最优验证准确率：0.9234
```

### 2. 测试单张图像

```bash
python test_flower_single_image.py <image_path> [model_path]
```

**示例：**
```bash
python test_flower_single_image.py ./flower.jpg ./mobilenetv3_flowers_best.pth
```

**输出示例：**
```
✅ 成功加载图像: ./flower.jpg

==================================================
🌸 花卉分类预测结果
==================================================
图像路径: ./flower.jpg

Top-5 预测结果:
  1. 类别 5: 0.9654
  2. 类别 8: 0.0246
  3. 类别 2: 0.0089
  4. 类别 12: 0.0008
  5. 类别 1: 0.0003
==================================================
```

------

## 📂 数据集准备

### 目录结构

```
flowers-dataset/
├── train/
│   ├── 0-tulip/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── 1-sunflower/
│   │   ├── image1.jpg
│   │   └── ...
│   └── ... (17个花卉类别)
└── val/
    ├── 0-tulip/
    │   ├── image1.jpg
    │   └── ...
    ├── 1-sunflower/
    │   └── ...
    └── ... (17个花卉类别)
```

### 数据集下载与处理

1. 从 [Kaggle](https://www.kaggle.com/datasets/saidakbarp/17-category-flowers) 下载数据集
2. 解压文件：`unzip 17-category-flowers.zip`
3. 按上述目录结构组织训练集和验证集

------

## 🔧 模型架构

### PlantDiseaseClassifier（原始模型 - 叶片病害识别）
- 类别数：3
- 预训练：否

### FlowerClassifier（新增模型 - 花卉分类）
- 类别数：17
- 预训练：是（使用 ImageNet 预训练权重）
- 参数量：约 1.5M

------

## 📊 图像预处理

### 训练集预处理
```python
transforms.Compose([
    transforms.Resize(256),                    # 缩放到256x256
    transforms.CenterCrop(224),                # 中心裁剪到224x224
    transforms.RandomHorizontalFlip(),         # 随机水平翻转
    transforms.RandomVerticalFlip(),           # 随机垂直翻转
    transforms.RandomRotation(20),             # 随机旋转±20°
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet 标准均值
                        [0.229, 0.224, 0.225])   # ImageNet 标准方差
])
```

### 验证集预处理
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

## ✨ 关键特性

- ✅ MobileNetV3-Small 轻量级模型
- ✅ 支持 17 种花卉分类
- ✅ 完整的数据增强策略
- ✅ 余弦退火学习率调度
- ✅ 自动保存最优模型
- ✅ 支持 GPU/CPU 训练
- ✅ 单张图像快速推理
- ✅ Top-5 预测结果展示

------

## 📌 项目结构

```
MobileNetV3-for-leaf/
├── main (分支)              # 原始叶片病害识别
│   ├── train.py            # 原始训练脚本
│   ├── model.py            # 原始模型定义
│   └── test_single_image.py # 单张图像测试
│
└── test (分支)             # 花卉分类开发分支 ⭐
    ├── train_flowers.py     # 花卉训练脚本 ✨
    ├── model.py             # 更新后的模型定义 ✨
    ├── test_flower_single_image.py  # 花卉测试脚本 ✨
    └── README_FLOWERS.md    # 本文档 ✨
```

------

## 🎯 模型对比

| 特性 | 叶片病害识别 | 花卉分类 |
|------|-----------|--------|
| 分支 | `main` | `test` |
| 类别数 | 3 | 17 |
| 预训练 | 否 | 是 ✓ |
| 数据增强 | 基础 | 增强 ✓ |
| 学习率调度 | 无 | 余弦退火 ✓ |

------

## 💡 性能提示

- 📌 使用预训练权重可显著提升收敛速度和最终精度
- 📌 充分的数据增强有助于提高模型泛化能力
- 📌 建议使用 GPU 训练以加快速度
- 📌 调整 `batch_size` 根据显存大小灵活选择

------

## 📌 项目作者

**李中政**  
GitHub: [zhongzhengli](https://github.com/zhongzhengli13)

---

**说明**：本项目在 `test` 分支中开发，不影响 `main` 分支的原始代码。
