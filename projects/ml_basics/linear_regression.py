#!/usr/bin/env python3
"""
线性回归示例
使用TensorFlow实现简单的线性回归模型
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def generate_data(n_samples=100):
    """生成线性回归的样本数据"""
    np.random.seed(42)
    X = np.random.randn(n_samples, 1)
    y = 2 * X + 1 + 0.1 * np.random.randn(n_samples, 1)
    return X.astype(np.float32), y.astype(np.float32)


def linear_regression_model():
    """创建线性回归模型"""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, input_shape=(1,))
    ])
    return model


def train_model():
    """训练线性回归模型"""
    # 生成数据
    X_train, y_train = generate_data(100)
    
    # 创建模型
    model = linear_regression_model()
    
    # 编译模型
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # 训练模型
    history = model.fit(X_train, y_train, epochs=100, verbose=0)
    
    # 预测
    y_pred = model.predict(X_train)
    
    # 可视化
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train, y_train, alpha=0.6, label='真实数据')
    plt.plot(X_train, y_pred, 'r-', label='预测线')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('线性回归结果')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return model, history


if __name__ == "__main__":
    print("开始训练线性回归模型...")
    model, history = train_model()
    print("训练完成！")