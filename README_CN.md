# ResNet50 特征图可视化工具

一个全面的可视化工具，用于探索和理解 ResNet50 如何通过其内部层处理图像。

## 功能特点

### 多种可视化技术
- **特征图**：可视化不同层的激活模式
- **激活最大化**：生成最大激活特定神经元的图像
- **层级相关传播 (LRP)**：了解哪些输入像素对特定激活贡献最大
- **Grad-CAM**：可视化对分类决策重要的区域

### 高级自定义选项
- **图像预处理**：
  - 多种调整大小策略（简单缩放、保持宽高比、填充后缩放）
  - 自定义裁剪选项（中心裁剪、随机裁剪、五点裁剪、十点裁剪）
  - 标准化策略（ImageNet、自定义均值和标准差）
  - 图像增强选项

- **可视化控制**：
  - 选择特定层和通道
  - 调整网格布局
  - 色彩映射选择
  - 规范化方法控制

### 分析和导出功能
- 特征统计（均值、标准差、分布）
- 导出为 JSON/CSV 数据
- 下载所有生成的可视化图像

  ![image](https://github.com/algaradi/resvisualize/blob/main/preview/image(10).png)
![image](https://github.com/algaradi/resvisualize/blob/main/preview/image(2).png)
![image](https://github.com/algaradi/resvisualize/blob/main/preview/image(3).png)
![image](https://github.com/algaradi/resvisualize/blob/main/preview/image(1).png)
![image](https://github.com/algaradi/resvisualize/blob/main/preview/image(4).png)

![image](https://github.com/algaradi/resvisualize/blob/main/preview/image(9).png)
![image](https://github.com/algaradi/resvisualize/blob/main/preview/image(5).png) ![image](https://github.com/algaradi/resvisualize/blob/main/preview/image(6).png)

![image](https://github.com/algaradi/resvisualize/blob/main/preview/image(7).png)![image](https://github.com/algaradi/resvisualize/blob/main/preview/image(8).png)

![image](https://github.com/algaradi/resvisualize/blob/main/preview/image(11).png)

## 技术栈
- **后端**：Python、Flask、PyTorch
- **前端**：HTML、CSS (Tailwind)、JavaScript
- **可视化**：Matplotlib、Plotly
- **深度学习**：PyTorch、torchvision、Captum

## 安装指南

### 前提条件
- Python 3.8+
- pip (Python 包管理器)
- 建议：虚拟环境（如 conda 或 venv）

### 安装步骤

1. 克隆项目仓库
   ```
   git clone https://github.com/algaradi/resvisualize.git
   cd resvisualize
   ```

2. 创建并激活虚拟环境（可选但推荐）
   ```
   conda create -n resvisualize python=3.8
   conda activate resvisualize
   ```

3. 安装依赖项
   ```
   pip install -r requirements.txt
   ```

## 使用方法

1. 启动 Flask 应用
   ```
   python app.py
   ```

2. 在浏览器中访问应用
   ```
   http://localhost:5000
   ```

3. 上传图像并尝试不同的可视化技术：
   - 特征图可视化
   - 激活最大化
   - 层级相关传播 (LRP)
   - Grad-CAM

## 预处理选项详解

### 调整大小策略
- **简单缩放**：直接调整到目标尺寸，可能会扭曲宽高比
- **保持宽高比**：调整大小同时保持原始宽高比
- **填充后缩放**：添加填充以保持宽高比，然后调整大小

### 裁剪策略
- **中心裁剪**：从图像中心裁剪
- **随机裁剪**：随机位置裁剪
- **五点裁剪**：从四个角和中心进行裁剪
- **十点裁剪**：五点裁剪加水平翻转版本

### 标准化策略
- **ImageNet**：使用 ImageNet 数据集的均值和标准差
- **自定义**：指定自己的均值和标准差值

## 许可证

本项目仅供非商业用途。详情请参阅 LICENSE 文件。 
