#!/usr/bin/env python3
"""
环境测试脚本
验证TensorFlow和相关库是否正常工作
"""

# 禁用TensorFlow警告
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    print("🔍 测试环境配置...")
    
    # 测试基础库
    import numpy as np
    print("✅ NumPy:", np.__version__)
    
    import pandas as pd
    print("✅ Pandas:", pd.__version__)
    
    import matplotlib
    print("✅ Matplotlib:", matplotlib.__version__)
    
    # 测试TensorFlow
    import tensorflow as tf
    print("✅ TensorFlow:", tf.__version__)
    
    # 测试GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("✅ GPU可用:", len(gpus), "个GPU")
        for i, gpu in enumerate(gpus):
            print(f"   GPU {i}: {gpu.name}")
    else:
        print("⚠️  未检测到GPU，将使用CPU")
    
    # 简单张量操作测试
    a = tf.constant([1, 2, 3])
    b = tf.constant([4, 5, 6])
    c = tf.add(a, b)
    print("✅ 张量运算测试:", c.numpy())
    
    print("\n🎉 环境配置正常！可以开始学习了！")
    
except ImportError as e:
    print(f"❌ 导入错误: {e}")
except Exception as e:
    print(f"❌ 其他错误: {e}")