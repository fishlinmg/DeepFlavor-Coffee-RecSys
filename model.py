"""
1D-CNN自编码器模型
使用1D卷积神经网络构建自编码器，将10维咖啡特征映射为64维深度表征
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
from typing import Tuple, Optional
import matplotlib.pyplot as plt


class CoffeeAutoencoder:
    """精品咖啡1D-CNN自编码器"""

    def __init__(self, input_dim: int = 10, latent_dim: int = 64):
        """
        初始化自编码器模型

        Args:
            input_dim: 输入特征维度 (默认10维感官评分)
            latent_dim: 潜在空间维度 (默认64维深度表征)
        """
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.model = None
        self.encoder = None
        self.decoder = None
        self.history = None

        print(f"初始化1D-CNN自编码器:")
        print(f"  输入维度: {input_dim}")
        print(f"  潜在维度: {latent_dim}")

    def build_model(self) -> Tuple[keras.Model, keras.Model, keras.Model]:
        """
        构建自编码器模型

        Returns:
            (完整自编码器, 编码器, 解码器)
        """
        print("\n正在构建1D-CNN自编码器模型...")

        # 输入层
        input_layer = layers.Input(shape=(self.input_dim, 1))

        # ============ 编码器 (Encoder) ============
        # 第一个1D卷积层：提取局部特征
        x = layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(input_layer)
        x = layers.MaxPooling1D(pool_size=2)(x)

        # 第二个1D卷积层：深层特征提取
        x = layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(x)
        x = layers.MaxPooling1D(pool_size=2)(x)

        # 全局平均池化：降低维度
        x = layers.GlobalAveragePooling1D()(x)

        # 潜在向量层：压缩到64维深度表征
        latent = layers.Dense(self.latent_dim, activation='relu', name='latent_vector')(x)

        # ============ 解码器 (Decoder) ============
        # 上采样到原始维度
        x = layers.Dense(128, activation='relu')(latent)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dense(self.input_dim, activation='linear')(x)

        # 添加维度以匹配输出形状
        output_layer = layers.Reshape((self.input_dim, 1))(x)

        # 构建完整模型
        autoencoder = keras.Model(input_layer, output_layer, name='coffee_autoencoder')

        # 构建编码器（仅用于推理时提取特征）
        encoder = keras.Model(input_layer, latent, name='encoder')

        # 构建解码器（仅用于测试）
        decoder_input = layers.Input(shape=(self.latent_dim,))
        decoder_output = autoencoder.layers[-3](decoder_input)  # Dense层
        decoder_output = autoencoder.layers[-2](decoder_output)  # Dense层
        decoder_output = autoencoder.layers[-1](decoder_output)  # Reshape层
        decoder = keras.Model(decoder_input, decoder_output, name='decoder')

        self.model = autoencoder
        self.encoder = encoder
        self.decoder = decoder

        # 编译模型
        autoencoder.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        print("模型构建完成！")
        print(f"\n模型结构:")
        autoencoder.summary()

        return autoencoder, encoder, decoder

    def train(self,
              X_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              epochs: int = 50,
              batch_size: int = 32,
              save_path: str = 'models/autoencoder_model.h5') -> keras.callbacks.History:
        """
        训练自编码器模型

        Args:
            X_train: 训练特征数据
            X_val: 验证特征数据 (可选)
            epochs: 训练轮数
            batch_size: 批次大小
            save_path: 模型保存路径

        Returns:
            训练历史记录
        """
        if self.model is None:
            self.build_model()

        print(f"\n开始训练模型...")
        print(f"训练数据形状: {X_train.shape}")
        if X_val is not None:
            print(f"验证数据形状: {X_val.shape}")

        # 调整输入形状为 (samples, features, 1)
        X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

        # 回调函数
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.0001
            )
        ]

        # 训练模型
        validation_data = (X_val_reshaped, X_val_reshaped) if X_val is not None else None
        if X_val is not None:
            X_val_reshaped = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)

        self.history = self.model.fit(
            X_train_reshaped, X_train_reshaped,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        # 保存模型
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.model.save(save_path)
        print(f"\n模型已保存到: {save_path}")

        return self.history

    def encode(self, X: np.ndarray) -> np.ndarray:
        """
        使用编码器将输入编码为潜在向量

        Args:
            X: 输入特征数据 (n_samples, input_dim)

        Returns:
            潜在向量 (n_samples, latent_dim)
        """
        if self.encoder is None:
            raise ValueError("模型尚未训练，请先调用 train() 方法")

        X_reshaped = X.reshape(X.shape[0], X.shape[1], 1)
        latent_vectors = self.encoder.predict(X_reshaped, verbose=0)
        return latent_vectors

    def decode(self, latent_vectors: np.ndarray) -> np.ndarray:
        """
        使用解码器将潜在向量重构为原始特征

        Args:
            latent_vectors: 潜在向量 (n_samples, latent_dim)

        Returns:
            重构特征 (n_samples, input_dim)
        """
        if self.decoder is None:
            raise ValueError("模型尚未训练，请先调用 train() 方法")

        reconstructed = self.decoder.predict(latent_vectors, verbose=0)
        return reconstructed.reshape(reconstructed.shape[0], reconstructed.shape[1])

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """
        完整重构：编码+解码

        Args:
            X: 输入特征数据

        Returns:
            重构特征
        """
        latent_vectors = self.encode(X)
        reconstructed = self.decode(latent_vectors)
        return reconstructed

    def evaluate(self, X: np.ndarray) -> dict:
        """
        评估模型性能

        Args:
            X: 测试数据

        Returns:
            评估指标字典
        """
        X_reshaped = X.reshape(X.shape[0], X.shape[1], 1)
        reconstructed = self.model.predict(X_reshaped, verbose=0)

        # 计算重构误差
        mse = np.mean((X.reshape(X.shape[0], X.shape[1]) - reconstructed.reshape(X.shape[0], X.shape[1])) ** 2)
        mae = np.mean(np.abs(X.reshape(X.shape[0], X.shape[1]) - reconstructed.reshape(X.shape[0], X.shape[1])))

        return {
            'mse': mse,
            'mae': mae,
            'rmse': np.sqrt(mse)
        }

    def load_model(self, model_path: str):
        """
        加载已训练的模型

        Args:
            model_path: 模型文件路径
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        print(f"正在加载模型: {model_path}")
        self.model = keras.models.load_model(model_path)

        # 重新构建编码器和解码器
        self.build_model()

        # 设置编码器和解码器的权重
        self.encoder.set_weights(self.model.get_weights()[:len(self.encoder.get_weights())])
        self.decoder.set_weights(self.model.get_weights()[len(self.encoder.get_weights()):])

        print("模型加载完成！")

    def plot_training_history(self, save_path: str = 'models/training_history.png'):
        """
        绘制训练历史曲线

        Args:
            save_path: 图片保存路径
        """
        if self.history is None:
            raise ValueError("模型尚未训练")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # 损失曲线
        ax1.plot(self.history.history['loss'], label='Training Loss')
        ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        # MAE曲线
        ax2.plot(self.history.history['mae'], label='Training MAE')
        ax2.plot(self.history.history['val_mae'], label='Validation MAE')
        ax2.set_title('Model MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"训练历史曲线已保存到: {save_path}")
        plt.close()

    def get_model_summary(self) -> str:
        """
        获取模型结构摘要

        Returns:
            模型结构字符串
        """
        if self.model is None:
            raise ValueError("模型尚未构建")

        summary_lines = []
        self.model.summary(print_fn=lambda x: summary_lines.append(x))
        return '\n'.join(summary_lines)


if __name__ == "__main__":
    # 示例用法
    print("正在测试1D-CNN自编码器模型...")

    # 创建示例数据
    input_dim = 10
    latent_dim = 64
    n_samples = 100

    # 生成随机数据
    X_sample = np.random.randn(n_samples, input_dim)

    # 创建模型
    autoencoder = CoffeeAutoencoder(input_dim=input_dim, latent_dim=latent_dim)

    # 训练模型
    autoencoder.build_model()
    history = autoencoder.train(X_sample, epochs=10, batch_size=16)

    # 测试编码
    latent_vectors = autoencoder.encode(X_sample[:5])
    print(f"\n编码测试:")
    print(f"输入形状: {X_sample[:5].shape}")
    print(f"潜在向量形状: {latent_vectors.shape}")

    # 测试重构
    reconstructed = autoencoder.reconstruct(X_sample[:5])
    print(f"\n重构测试:")
    print(f"重构形状: {reconstructed.shape}")

    # 评估
    metrics = autoencoder.evaluate(X_sample[:10])
    print(f"\n评估结果:")
    print(f"MSE: {metrics['mse']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
