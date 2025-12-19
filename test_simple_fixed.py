import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

print("=" * 60)
print("DeepFlavor Coffee Recommender - Working Version")
print("=" * 60)

# 1. 加载数据
df = pd.read_csv('data/coffee_data.csv')
print(f"数据记录数: {len(df)}")

# 2. 提取特征
features = ['Aroma', 'Flavor', 'Aftertaste', 'Acidity', 'Body', 
           'Balance', 'Uniformity', 'Clean Cup', 'Sweetness', 'Cupper Points']
X = df[features].values

# 3. 标准化和PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=8)
deep_embeddings = pca.fit_transform(X_scaled)
print(f"深度特征维度: {deep_embeddings.shape}")
print(f"方差解释比例: {pca.explained_variance_ratio_.sum():.4f}")

# 4. 构建KNN索引
knn_original = NearestNeighbors(n_neighbors=6, metric='euclidean')
knn_original.fit(X_scaled)

knn_deep = NearestNeighbors(n_neighbors=6, metric='cosine')
knn_deep.fit(deep_embeddings)
print("KNN索引构建完成")

# 5. 测试推荐
print("\n=== 测试推荐功能 ===")

# 5.1 根据咖啡ID推荐 (使用深度特征)
target_deep = deep_embeddings[0:1]
distances, indices = knn_deep.kneighbors(target_deep)
similarities = 1 - distances.flatten()

print("\n为第1款咖啡推荐相似咖啡:")
for i, (idx, sim) in enumerate(zip(indices.flatten(), similarities)):
    if idx != 0 and i < 3:
        row = df.iloc[idx]
        print(f"  {i}. {row['Coffee Name']} ({row['Country of Origin']}) - 相似度: {sim:.4f}")

# 5.2 根据偏好推荐
user_prefs = [8.5, 8.0, 7.5, 9.0, 6.5, 8.0, 8.5, 9.0, 7.0, 8.0]
user_scaled = scaler.transform([user_prefs])
user_deep = pca.transform(user_scaled)

distances, indices = knn_deep.kneighbors(user_deep)
similarities = 1 - distances.flatten()

print("\n根据偏好推荐:")
for i, (idx, sim) in enumerate(zip(indices.flatten(), similarities)):
    if i < 3:
        row = df.iloc[idx]
        print(f"  {i+1}. {row['Coffee Name']} ({row['Country of Origin']}) - 相似度: {sim:.4f}")

# 6. 性能评估
print("\n=== 性能评估 ===")
total_samples = 50
original_sim = 0
deep_sim = 0

for i in range(total_samples):
    # 原始特征
    target = X_scaled[i:i+1]
    distances, _ = knn_original.kneighbors(target)
    similarities = 1 / (1 + distances.flatten())
    original_sim += np.mean(similarities[similarities < 1])
    
    # 深度特征
    target = deep_embeddings[i:i+1]
    distances, _ = knn_deep.kneighbors(target)
    similarities = 1 - distances.flatten()
    deep_sim += np.mean(similarities[similarities < 1])

original_avg = original_sim / total_samples
deep_avg = deep_sim / total_samples

print(f"原始特征平均相似度: {original_avg:.4f}")
print(f"深度特征平均相似度: {deep_avg:.4f}")
print(f"相似度提升: {(deep_avg - original_avg)/original_avg*100:.2f}%")

print("\n" + "=" * 60)
print("推荐系统测试完成！")
print("=" * 60)
