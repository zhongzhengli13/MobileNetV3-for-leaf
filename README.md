

# 🌿 Crop Leaf Disease Classification using MobileNetV3 (PyTorch)---植物病害识别

> **项目背景：**
> 早期曾在嵌入式设备 **RV1106** 上实现过植物病害识别模型，但由于设备性能与数据集限制，识别效果一般。本项目基于更完善的数据与资源，重新使用 **MobileNetV3** 小型结构从头训练，识别三类病害，**20 个 epoch** 后模型验证准确率达 **96.67%**。实验中发现模型在 **第 10 个 epoch 左右就已表现出较好的识别能力**。

本项目使用轻量级模型 **MobileNetV3** 从头训练实现 **农作物叶子健康状况识别**，识别类别为：

- `healthy`
- `powdery`（白粉病）
- `rust`（锈病）

数据集来自 Kaggle: [Plant disease recognition dataset](https://www.kaggle.com/datasets/rashikrahmanpritom/plant-disease-recognition-dataset/data)	

------

### 训练模型

```bash
python train.py
```

支持配置：

- 所有训练超参数（学习率、批大小、epoch 数量等）均可在 `train.py` 中修改。

输出示例：

> ![image-20250731085147606](https://lzz-1340752507.cos.ap-shanghai.myqcloud.com/lzz/image-20250731085147606.png)

------

### 测试单张图像

```bash
python test_single_image.py 
```

输出示例：

> ![single测试](https://lzz-1340752507.cos.ap-shanghai.myqcloud.com/lzz/single%E6%B5%8B%E8%AF%95.png)

------

## 模型结构说明

模型采用自定义实现的 `MobileNetV3-Small`，总参数量约 **1.5M**，非常适合部署于计算资源受限的设备上：

> 模型结构：
>
> ![模型大小small版本](https://lzz-1340752507.cos.ap-shanghai.myqcloud.com/lzz/%E6%A8%A1%E5%9E%8B%E5%A4%A7%E5%B0%8Fsmall%E7%89%88%E6%9C%AC.png)

------

##  输入图像与预处理

- 原图尺寸约为 **4000x2672**
- 模型输入统一调整为 **224x224**
- 图像预处理方式：

```python
transforms.Resize(256),
transforms.CenterCrop(224),
transforms.RandomHorizontalFlip(),
transforms.ToTensor(),
transforms.Normalize([0.4717, 0.5892, 0.3972],
                     [0.1704, 0.1531, 0.1755])
```

其中 `Normalize` 的均值与标准差由 `calculator_mean_std.py` 脚本统计所得，确保模型在颜色分布上的泛化能力。

------

##  已实现功能

-  MobileNetV3 小型结构从头训练
-  自定义数据均值与标准差统计
-  支持单张图像推理测试
-  验证集准确率自动保存最优模型

------

## 📌 项目作者

**李中政**
 GitHub: [@lizhongzheng13](https://github.com/lizhongzheng13)
