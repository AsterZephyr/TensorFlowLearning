# TensorFlow Learning Project

这是一个用于学习机器学习（ML）、深度学习（DL）和强化学习（RL）的项目，基于TensorFlow构建。

## 🚀 快速开始

### 1. 激活虚拟环境

```bash
# 进入项目目录
cd TensorFlowLearning

# 激活虚拟环境
source tf_env/bin/activate

# 验证环境
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"
```

### 2. 启动Jupyter Lab

```bash
# 启动 Jupyter Lab（推荐）
jupyter lab

# 或者启动传统的 Jupyter Notebook
jupyter notebook
```

浏览器会自动打开，默认地址：`http://localhost:8888`

### 3. 运行示例代码

#### 方式一：在Jupyter中运行
- 打开 `notebooks/01_tensorflow_basics.ipynb` 开始基础学习
- 逐个运行代码单元格

#### 方式二：直接运行Python脚本
```bash
# 运行线性回归示例
python projects/ml_basics/linear_regression.py
```

## 📁 项目结构

```
TensorFlowLearning/
├── tf_env/                    # Python虚拟环境
├── notebooks/                 # Jupyter Notebooks
│   └── 01_tensorflow_basics.ipynb
├── datasets/                  # 数据集存放目录
├── models/                    # 训练好的模型
├── projects/                  # 项目代码
│   ├── ml_basics/            # 机器学习基础
│   │   └── linear_regression.py
│   ├── deep_learning/        # 深度学习项目
│   └── reinforcement_learning/ # 强化学习项目
├── utils/                     # 工具函数
│   └── data_utils.py
├── requirements.txt           # 依赖包列表
└── README.md                 # 本文件
```

## 🔧 环境配置

### 已安装的核心库：
- **TensorFlow 2.20+**: 深度学习框架
- **NumPy**: 数值计算
- **Pandas**: 数据处理
- **Matplotlib/Seaborn**: 数据可视化
- **Scikit-learn**: 机器学习工具
- **Jupyter Lab/Notebook**: 交互式开发环境

### 安装额外依赖：
```bash
# 激活环境后安装
pip install plotly opencv-python gym stable-baselines3 transformers

# 安装d2l (Dive into Deep Learning)
pip install --no-deps d2l
```

## 🎯 学习路径建议

### 1. 机器学习基础 (ML Basics)
- [ ] TensorFlow基础操作
- [ ] 线性回归和逻辑回归
- [ ] 决策树和随机森林
- [ ] 聚类算法
- [ ] 数据预处理和特征工程

### 2. 深度学习 (Deep Learning)
- [ ] 神经网络基础
- [ ] 卷积神经网络（CNN）
- [ ] 循环神经网络（RNN/LSTM）
- [ ] 生成对抗网络（GAN）
- [ ] 迁移学习

### 3. 强化学习 (Reinforcement Learning)
- [ ] Q-learning
- [ ] 深度Q网络（DQN）
- [ ] Policy Gradient
- [ ] Actor-Critic方法

## 💡 使用技巧

### Jupyter快捷键：
- `Shift + Enter`: 运行当前单元格并跳到下一个
- `Ctrl + Enter`: 运行当前单元格
- `A`: 在上方插入新单元格
- `B`: 在下方插入新单元格
- `DD`: 删除当前单元格
- `M`: 转换为Markdown单元格
- `Y`: 转换为代码单元格

### 项目开发流程：
1. 在 `notebooks/` 中进行实验和原型开发
2. 将成熟的代码整理到 `projects/` 对应目录
3. 将可复用的函数放入 `utils/`
4. 大型数据集放入 `datasets/`
5. 训练好的模型保存到 `models/`

## 🔍 常见问题

### Q: 如何抑制TensorFlow的protobuf警告？
在notebook开头添加以下代码：
```python
import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')
```
或者直接使用提供的 `clean_notebook_template.ipynb` 模板。

### Q: 如何检查GPU是否可用？
```python
import tensorflow as tf
print("GPU Available: ", tf.config.list_physical_devices('GPU'))
```

### Q: 如何退出虚拟环境？
```bash
deactivate
```

### Q: 如何重新安装依赖？
```bash
pip install -r requirements.txt
```

### Q: 如何解决d2l安装错误？
如果遇到d2l安装的兼容性问题，使用：
```bash
pip install --no-deps d2l
```
这会跳过依赖检查直接安装d2l。

### Q: 如何安装TensorFlow Probability？
由于版本兼容性问题，建议使用：
```bash
pip install --no-deps tensorflow-probability==0.24.0
```
注意：TFP可能有一些兼容性警告，但不影响核心功能使用。

### Q: Jupyter无法找到虚拟环境？
```bash
# 在虚拟环境中安装ipykernel
pip install ipykernel
python -m ipykernel install --user --name=tf_env --display-name="TensorFlow Env"

# 然后在Jupyter中选择对应的kernel
```

## 📚 学习资源

- [TensorFlow官方教程](https://www.tensorflow.org/tutorials?hl=zh-cn)
- [Keras官方文档](https://keras.io/)
- [Deep Learning Book](https://www.deeplearningbook.org/)
- [CS229机器学习课程](http://cs229.stanford.edu/)
- [CS231n卷积神经网络](http://cs231n.stanford.edu/)

## 🚀 下一步

1. 完成基础TensorFlow教程
2. 实践经典的ML/DL项目
3. 参加Kaggle竞赛
4. 构建自己的端到端项目

祝你学习愉快！🎉