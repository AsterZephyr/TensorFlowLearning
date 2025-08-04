"""
Jupyter启动配置
在Jupyter notebook开头运行此代码来抑制警告
"""

# 抑制TensorFlow和其他库的警告
import os
import warnings

# 设置环境变量抑制TensorFlow日志
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 抑制Python警告
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# 设置matplotlib后端
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，避免一些警告

print("✅ 警告已抑制，环境已优化")