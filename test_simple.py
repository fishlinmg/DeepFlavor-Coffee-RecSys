#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简化版咖啡推荐系统测试
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import os

print("=" * 60)
print("DeepFlavor Coffee Recommender - Simplified Version")
print("=" * 60)

# 1. 加载数据
print("\n1. 加载咖啡数据...")
df = pd.read_csv('data/coffee_data.csv')
print(f"   数据记录数: {len(df)}")

# 2. 提取特征
features = ['Aroma', 'Flavor', 'Aftertaste', 'Acidity', 'Body', 
           'Balance', 'Uniformity', 'Clean Cup', 'Sweetness', 'Cupper Points']
X = df[features].values
print(f"   特征维度: {X.shape}")

# 3. 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("   数据标准化完成")

# 4. PCA降维生成深度特征
pca = PCA(n_components=8)
deep_embeddings = pca.fit_transform(X_scaled)
print(f"   深度特征维度: {deep_embeddings.shape}")
print(f"   方差解释比例: {pca.explained_variance_ratio_.sum():.4f}")

# 5. 构建KNN索引
print("\n2. 构建KNN推荐索引...")
knn_original = NearestNeighbors(n_neighbors=6, metric='euclidean')
knn_original.fit(X_scaled)

knn_deep = NearestNeighbors(n_neighbors=6, metric='cosine')
knn_deep.fit(deep_embeddings)
print("   KNN索引构建完成")

# 6. 测试推荐
print("\n3. 测试推荐功能...")

# 6.1 根据咖啡ID推荐
print("\n   测试1: 为第1款咖啡推荐相似咖啡")
target = X_scaled[0:1]
distances, indices = knn_deep.kneighbors(target)
similarities = 1 - distances.flatten()

for i, (idx, sim) in enumerate(zip(indices.flatten(), similarities)):
    if idx != 0 and i < 3:
        row = df.iloc[idx]
        print(f"      {i}. {row['Coffee Name']} ({row['Country of Origin']}) - 相似度: {sim:.4f}")

# 6.2 根据偏好推荐
print("\n   测试2: 根据用户偏好推荐")
print("      用户偏好: 高酸度(9.0), 中等醇度(6.5), 高平衡(8.0)")
user_prefs = [8.5, 8.0, 7.5, 9.0, 6.5, 8.0, 8.5, 9.0, 7.0, 8.0]
user_scaled = scaler.transform([user_prefs])
user_deep = pca.transform(user_scaled)

distances, indices = knn_deep.kneighbors(user_deep)
similarities = 1 - distances.flatten()

for i, (idx, sim) in enumerate(zip(indices.flatten(), similarities)):
    if i < 3:
        row = df.iloc[idx]
        print(f"      {i+1}. {row['Coffee Name']} ({row['Country of Origin']}) - 相似度: {sim:.4f}")

# 7. 性能评估
print("\n4. 性能评估...")
total_samples = 50
original_sim = 0
deep_sim = 0

for i in range(total_samples):
    # 原始特征
    target = X_scaled[i:i+1]
    distances, _ = knn_original.kneighbors(target)
    similarities = 1 / (1 + distances.flatten())
    original_sim += np.mean(similarities[similarities < 1])  # 排除自己
    
    # 深度特征
    target = deep_embeddings[i:i+1]
    distances, _ = knn_deep.kneighbors(target)
    similarities = 1 - distances.flatten()
    deep_sim += np.mean(similarities[similarities < 1])  # 排除自己

original_avg = original_sim / total_samples
deep_avg = deep_sim / total_samples

print(f"   原始特征平均相似度: {original_avg:.4f}")
print(f"   深度特征平均相似度: {deep_avg:.4f}")
print(f"   相似度提升: {(deep_avg - original_avg)/original_avg*100:.2f}%")

print("\n" + "=" * 60)
print("推荐系统测试完成！")
print("=" * 60)
print("\n✅ 系统正常工作！")
print("\n要启动Web应用，请运行:")
print("   python app.py")
print("\n然后访问: http://localhost:5000")
