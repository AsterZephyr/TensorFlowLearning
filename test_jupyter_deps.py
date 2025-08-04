#!/usr/bin/env python3
"""
测试Jupyter中需要的依赖
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')

def test_jupyter_deps():
    """测试Jupyter中常用的依赖"""
    try:
        print("🔍 测试Jupyter常用依赖...")
        
        # 基础计算库
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
        
        import tensorflow as tf
        print(f"✅ TensorFlow: {tf.__version__}")
        
        # 尝试导入tensorflow_probability
        try:
            import tensorflow_probability as tfp
            print(f"✅ TensorFlow Probability: {tfp.__version__}")
        except ImportError as e:
            print(f"⚠️  TensorFlow Probability: {e}")
        
        # D2L相关
        import d2l
        print(f"✅ D2L: {d2l.__version__}")
        
        try:
            from d2l import tensorflow as d2l_tf
            print(f"✅ D2L TensorFlow: 导入成功")
        except ImportError as e:
            print(f"⚠️  D2L TensorFlow: {e}")
        
        # 测试基本功能
        print("\n🔍 测试基本功能...")
        
        # TensorFlow基本操作
        x = tf.constant([1.0, 2.0, 3.0])
        y = tf.square(x)
        print(f"✅ TensorFlow计算: {y.numpy()}")
        
        # 创建简单模型
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(1, input_shape=(1,))
        ])
        print("✅ Keras模型创建成功")
        
        print("\n🎉 Jupyter环境就绪！")
        return True
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False

if __name__ == "__main__":
    test_jupyter_deps()