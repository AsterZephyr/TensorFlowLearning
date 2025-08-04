#!/usr/bin/env python3
"""
测试所有依赖库是否正常工作
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def test_imports():
    """测试所有重要的库导入"""
    try:
        print("🔍 测试库导入...")
        
        # 基础库
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
        
        import pandas as pd
        print(f"✅ Pandas: {pd.__version__}")
        
        import matplotlib.pyplot as plt
        print(f"✅ Matplotlib: {plt.matplotlib.__version__}")
        
        # TensorFlow生态
        import tensorflow as tf
        print(f"✅ TensorFlow: {tf.__version__}")
        
        import tensorflow_probability as tfp
        print(f"✅ TensorFlow Probability: {tfp.__version__}")
        
        # D2L
        import d2l
        print(f"✅ D2L: {d2l.__version__}")
        
        from d2l import tensorflow as d2l_tf
        print(f"✅ D2L TensorFlow子模块: 导入成功")
        
        # 科学计算
        import scipy
        print(f"✅ SciPy: {scipy.__version__}")
        
        import sklearn
        print(f"✅ Scikit-learn: {sklearn.__version__}")
        
        print("\n🎉 所有依赖库测试通过！")
        return True
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        return False
    except Exception as e:
        print(f"❌ 其他错误: {e}")
        return False

def test_functionality():
    """测试核心功能"""
    try:
        print("\n🔍 测试核心功能...")
        
        import tensorflow as tf
        import tensorflow_probability as tfp
        import numpy as np
        
        # 测试TensorFlow
        x = tf.constant([1.0, 2.0, 3.0])
        y = tf.square(x)
        print(f"✅ TensorFlow计算: {y.numpy()}")
        
        # 测试TensorFlow Probability
        dist = tfp.distributions.Normal(0.0, 1.0)
        samples = dist.sample(3)
        print(f"✅ TFP采样: {samples.numpy()}")
        
        # 测试D2L
        from d2l import tensorflow as d2l
        # 创建一些虚拟数据测试
        X = tf.random.normal((10, 2))
        print(f"✅ D2L环境: 创建数据形状 {X.shape}")
        
        print("\n🎉 所有功能测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ 功能测试错误: {e}")
        return False

if __name__ == "__main__":
    import_ok = test_imports()
    func_ok = test_functionality()
    
    if import_ok and func_ok:
        print("\n🚀 环境完全就绪，可以开始深度学习之旅！")
    else:
        print("\n⚠️ 环境存在问题，请检查安装")