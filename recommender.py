"""
推荐系统核心模块
实现双路径推荐机制：原始特征KNN vs 深度特征KNN
"""

import numpy as np
import pickle
import os
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Optional, Union
import joblib

from data_loader import CoffeeDataLoader
from model import CoffeeAutoencoder


class CoffeeRecommender:
    """咖啡推荐系统"""

    def __init__(self,
                 data_path: str = 'data/coffee_data.csv',
                 model_path: str = 'models/autoencoder_model.h5',
                 embedding_path: str = 'data/coffee_embeddings.npy',
                 knn_index_path: str = 'data/knn_index.pkl'):
        """
        初始化推荐系统

        Args:
            data_path: 咖啡数据文件路径
            model_path: 自编码器模型路径
            embedding_path: 深度特征保存路径
            knn_index_path: KNN索引保存路径
        """
        self.data_path = data_path
        self.model_path = model_path
        self.embedding_path = embedding_path
        self.knn_index_path = knn_index_path

        # 组件
        self.data_loader = None
        self.autoencoder = None

        # 数据
        self.X_scaled = None
        self.X_raw = None
        self.df_processed = None
        self.deep_embeddings = None

        # KNN索引
        self.knn_original = None  # 原始特征KNN
        self.knn_deep = None      # 深度特征KNN

        # 推荐参数
        self.top_k = 5
        self.similarity_threshold = 0.5

        print("咖啡推荐系统初始化...")

    def initialize(self, force_retrain: bool = False):
        """
        初始化推荐系统：加载数据、训练模型、构建索引

        Args:
            force_retrain: 是否强制重新训练
        """
        print("=" * 60)
        print("初始化咖啡推荐系统")
        print("=" * 60)

        # 1. 加载和预处理数据
        print("\n步骤 1: 加载和预处理数据")
        self.data_loader = CoffeeDataLoader(self.data_path)
        self.X_scaled, self.df_processed = self.data_loader.process_data()
        self.X_raw = self.df_processed[self.data_loader.SENSORY_FEATURES].values

        # 2. 训练或加载模型
        print("\n步骤 2: 加载/训练1D-CNN自编码器模型")
        self.autoencoder = CoffeeAutoencoder(input_dim=10, latent_dim=64)

        if os.path.exists(self.model_path) and not force_retrain:
            print(f"发现已训练模型: {self.model_path}")
            try:
                self.autoencoder.load_model(self.model_path)
            except Exception as e:
                print(f"加载模型失败，将重新训练: {e}")
                self._train_model()
        else:
            print("未找到已训练模型，开始训练...")
            self._train_model()

        # 3. 生成深度特征嵌入
        print("\n步骤 3: 生成深度特征嵌入")
        self.deep_embeddings = self.autoencoder.encode(self.X_scaled)

        # 保存深度特征
        os.makedirs(os.path.dirname(self.embedding_path), exist_ok=True)
        np.save(self.embedding_path, self.deep_embeddings)
        print(f"深度特征已保存到: {self.embedding_path}")

        # 4. 构建KNN索引
        print("\n步骤 4: 构建KNN推荐索引")
        self._build_knn_indices()

        # 保存KNN索引
        self._save_knn_indices()

        print("\n=" * 60)
        print("推荐系统初始化完成！")
        print("=" * 60)

    def _train_model(self):
        """训练自编码器模型"""
        print("正在训练1D-CNN自编码器...")

        # 分割训练集和验证集
        split_idx = int(0.8 * len(self.X_scaled))
        X_train = self.X_scaled[:split_idx]
        X_val = self.X_scaled[split_idx:]

        # 构建并训练模型
        self.autoencoder.build_model()
        history = self.autoencoder.train(
            X_train,
            X_val,
            epochs=50,
            batch_size=32,
            save_path=self.model_path
        )

        # 绘制训练历史
        try:
            self.autoencoder.plot_training_history()
        except Exception as e:
            print(f"绘制训练历史失败: {e}")

        print("模型训练完成！")

    def _build_knn_indices(self):
        """构建KNN推荐索引"""

        # 1. 原始特征KNN (使用欧氏距离)
        print("  - 构建原始特征KNN索引 (10维)")
        self.knn_original = NearestNeighbors(
            n_neighbors=self.top_k + 1,  # +1 因为会排除自身
            metric='euclidean',
            algorithm='auto'
        )
        self.knn_original.fit(self.X_scaled)

        # 2. 深度特征KNN (使用余弦相似度)
        print("  - 构建深度特征KNN索引 (64维)")
        self.knn_deep = NearestNeighbors(
            n_neighbors=self.top_k + 1,  # +1 因为会排除自身
            metric='cosine',
            algorithm='auto'
        )
        self.knn_deep.fit(self.deep_embeddings)

        print("KNN索引构建完成！")

    def _save_knn_indices(self):
        """保存KNN索引"""
        os.makedirs(os.path.dirname(self.knn_index_path), exist_ok=True)

        knn_data = {
            'knn_original': self.knn_original,
            'knn_deep': self.knn_deep,
            'data_loader': self.data_loader,
            'df_processed': self.df_processed
        }

        with open(self.knn_index_path, 'wb') as f:
            pickle.dump(knn_data, f)

        print(f"KNN索引已保存到: {self.knn_index_path}")

    def recommend_by_coffee_id(self,
                               coffee_id: int,
                               method: str = 'deep',
                               top_k: Optional[int] = None) -> List[Dict]:
        """
        根据咖啡ID推荐相似咖啡

        Args:
            coffee_id: 咖啡索引
            method: 推荐方法 ('original' | 'deep')
            top_k: 返回推荐数量

        Returns:
            推荐结果列表
        """
        if top_k is None:
            top_k = self.top_k

        if coffee_id >= len(self.df_processed):
            raise IndexError(f"咖啡ID {coffee_id} 超出范围")

        # 获取目标咖啡特征
        if method == 'original':
            target_features = self.X_scaled[coffee_id:coffee_id+1]
        else:
            target_features = self.deep_embeddings[coffee_id:coffee_id+1]

        # 推荐
        recommendations = self._get_recommendations(target_features, coffee_id, method, top_k)

        return recommendations

    def recommend_by_preferences(self,
                                 preferences: List[float],
                                 method: str = 'deep',
                                 top_k: Optional[int] = None) -> List[Dict]:
        """
        根据用户偏好推荐咖啡

        Args:
            preferences: 用户偏好评分 (10维感官评分)
            method: 推荐方法 ('original' | 'deep')
            top_k: 返回推荐数量

        Returns:
            推荐结果列表
        """
        if top_k is None:
            top_k = self.top_k

        if len(preferences) != 10:
            raise ValueError("偏好评分必须包含10个维度")

        # 标准化用户偏好
        pref_array = np.array(preferences).reshape(1, -1)
        pref_scaled = self.data_loader.scaler.transform(pref_array)

        # 编码为深度特征
        target_features = self.autoencoder.encode(pref_scaled)

        # 推荐
        recommendations = self._get_recommendations(target_features, None, method, top_k)

        return recommendations

    def _get_recommendations(self,
                            target_features: np.ndarray,
                            exclude_id: Optional[int],
                            method: str,
                            top_k: int) -> List[Dict]:
        """
        获取推荐结果

        Args:
            target_features: 目标特征向量
            exclude_id: 要排除的咖啡ID (避免推荐自己)
            method: 推荐方法
            top_k: 推荐数量

        Returns:
            推荐结果列表
        """
        # 选择KNN索引
        if method == 'original':
            knn = self.knn_original
            distances, indices = knn.kneighbors(target_features)
            # 欧氏距离转换为相似度
            similarities = 1 / (1 + distances.flatten())
        else:
            knn = self.knn_deep
            distances, indices = knn.kneighbors(target_features)
            # 余弦距离转换为相似度
            similarities = 1 - distances.flatten()

        # 构建推荐结果
        recommendations = []
        idx_counter = 0

        for idx, sim in zip(indices.flatten(), similarities):
            # 跳过自身 (如果指定了exclude_id)
            if exclude_id is not None and idx == exclude_id:
                continue

            if idx_counter >= top_k:
                break

            # 获取咖啡信息
            coffee_info = self._get_coffee_info(idx)

            # 添加推荐信息
            recommendation = {
                'id': int(idx),
                'similarity': float(sim),
                'coffee': coffee_info,
                'method': method
            }

            recommendations.append(recommendation)
            idx_counter += 1

        return recommendations

    def _get_coffee_info(self, coffee_id: int) -> Dict:
        """
        获取咖啡详细信息

        Args:
            coffee_id: 咖啡索引

        Returns:
            咖啡信息字典
        """
        row = self.df_processed.iloc[coffee_id]

        # 提取感官评分
        sensory_scores = {}
        for feature in self.data_loader.SENSORY_FEATURES:
            sensory_scores[feature] = float(row.get(feature, 0))

        coffee_info = {
            'name': row.get('Coffee Name', f'Coffee #{coffee_id}'),
            'country': row.get('Country of Origin', 'Unknown'),
            'variety': row.get('Variety', 'Unknown'),
            'processing': row.get('Processing Method', 'Unknown'),
            'owner': row.get('Owner/Farm', 'Unknown'),
            'overall_score': float(row.get('Cupper Points', 0)),
            'sensory_scores': sensory_scores
        }

        return coffee_info

    def get_radar_chart_data(self, coffee_ids: List[int]) -> Dict:
        """
        获取雷达图数据

        Args:
            coffee_ids: 咖啡ID列表

        Returns:
            雷达图数据字典
        """
        radar_data = {
            'categories': self.data_loader.SENSORY_FEATURES,
            'series': []
        }

        for coffee_id in coffee_ids:
            coffee_info = self._get_coffee_info(coffee_id)
            series = {
                'name': f"{coffee_info['name']} ({coffee_info['country']})",
                'data': list(coffee_info['sensory_scores'].values())
            }
            radar_data['series'].append(series)

        return radar_data

    def evaluate_recommendation(self, method: str = 'deep') -> Dict:
        """
        评估推荐系统性能

        Args:
            method: 评估方法 ('original' | 'deep')

        Returns:
            评估指标
        """
        print(f"\n正在评估 {method} 路径推荐性能...")

        total_mse = 0
        total_samples = min(100, len(self.df_processed))  # 评估前100个样本

        for i in range(total_samples):
            # 获取推荐
            recommendations = self.recommend_by_coffee_id(i, method=method, top_k=5)

            if not recommendations:
                continue

            # 计算平均相似度
            avg_similarity = np.mean([r['similarity'] for r in recommendations])
            total_mse += (1 - avg_similarity) ** 2

        avg_mse = total_mse / total_samples

        metrics = {
            'method': method,
            'avg_mse': avg_mse,
            'avg_similarity': 1 - np.sqrt(avg_mse),
            'evaluation_samples': total_samples
        }

        print(f"评估结果:")
        print(f"  平均相似度: {metrics['avg_similarity']:.4f}")
        print(f"  平均MSE: {metrics['avg_mse']:.4f}")

        return metrics

    def compare_methods(self) -> Dict:
        """对比两种推荐方法"""
        print("\n" + "=" * 60)
        print("推荐方法对比实验")
        print("=" * 60)

        original_metrics = self.evaluate_recommendation('original')
        deep_metrics = self.evaluate_recommendation('deep')

        comparison = {
            'original': original_metrics,
            'deep': deep_metrics,
            'improvement': {
                'similarity_gain': deep_metrics['avg_similarity'] - original_metrics['avg_similarity'],
                'mse_reduction': original_metrics['avg_mse'] - deep_metrics['avg_mse']
            }
        }

        print(f"\n对比结果:")
        print(f"  原始特征路径 - 平均相似度: {original_metrics['avg_similarity']:.4f}")
        print(f"  深度特征路径 - 平均相似度: {deep_metrics['avg_similarity']:.4f}")
        print(f"  相似度提升: {comparison['improvement']['similarity_gain']:.4f} "
              f"({(comparison['improvement']['similarity_gain']/original_metrics['avg_similarity']*100):.2f}%)")

        return comparison

    def filter_by_country(self, country: str) -> List[int]:
        """根据国家筛选咖啡"""
        country_mask = self.df_processed['Country of Origin'] == country
        return list(self.df_processed[country_mask].index)

    def filter_by_processing(self, processing: str) -> List[int]:
        """根据处理法筛选咖啡"""
        processing_mask = self.df_processed['Processing Method'] == processing
        return list(self.df_processed[processing_mask].index)


if __name__ == "__main__":
    # 示例用法
    print("正在测试咖啡推荐系统...")

    # 初始化推荐系统
    recommender = CoffeeRecommender()

    try:
        recommender.initialize()

        # 测试：根据咖啡ID推荐
        print("\n\n测试 1: 根据咖啡ID推荐")
        recommendations = recommender.recommend_by_coffee_id(0, method='deep', top_k=3)
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec['coffee']['name']} - "
                  f"{rec['coffee']['country']} - "
                  f"相似度: {rec['similarity']:.4f}")

        # 测试：根据用户偏好推荐
        print("\n\n测试 2: 根据用户偏好推荐")
        user_prefs = [8.5, 9.0, 8.0, 7.5, 8.0, 8.5, 9.0, 8.5, 7.0, 8.0]  # 10维感官评分
        recommendations = recommender.recommend_by_preferences(user_prefs, method='deep', top_k=3)
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec['coffee']['name']} - "
                  f"{rec['coffee']['country']} - "
                  f"相似度: {rec['similarity']:.4f}")

        # 对比实验
        print("\n\n测试 3: 推荐方法对比")
        comparison = recommender.compare_methods()

        print("\n推荐系统测试完成！")

    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("请确保咖啡数据集文件存在")
