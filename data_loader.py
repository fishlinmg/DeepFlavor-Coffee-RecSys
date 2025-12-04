"""
数据加载与预处理模块
负责咖啡数据集的加载、清洗和预处理
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import os
from typing import Tuple, Dict, List


class CoffeeDataLoader:
    """咖啡数据预处理器"""

    # 感官评分特征列名（10维）
    SENSORY_FEATURES = [
        'Aroma',           # 干香
        'Flavor',          # 风味
        'Aftertaste',      # 余韵
        'Acidity',         # 酸度
        'Body',            # 醇度
        'Balance',         # 平衡
        'Uniformity',      # 一致性
        'Clean Cup',       # 干净度
        'Sweetness',       # 甜度
        'Cupper Points'    # 杯测师评分
    ]

    # 分类特征列名
    CATEGORICAL_FEATURES = [
        'Country of Origin',
        'Variety',
        'Processing Method',
        'Owner/Farm'
    ]

    def __init__(self, data_path: str = 'data/coffee_data.csv'):
        """
        初始化数据加载器

        Args:
            data_path: CSV数据文件路径
        """
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.original_data = None
        self.processed_data = None

    def load_data(self) -> pd.DataFrame:
        """
        加载咖啡数据集

        Returns:
            加载的数据DataFrame

        Raises:
            FileNotFoundError: 如果数据文件不存在
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"数据文件不存在: {self.data_path}\n"
                f"请确保CQI咖啡数据集在此路径。"
            )

        print(f"正在加载数据: {self.data_path}")
        df = pd.read_csv(self.data_path)
        print(f"数据加载完成: {len(df)} 条记录")

        # 保存原始数据
        self.original_data = df.copy()
        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        数据清洗：处理缺失值和异常值

        Args:
            df: 原始数据DataFrame

        Returns:
            清洗后的DataFrame
        """
        print("正在进行数据清洗...")

        # 1. 删除重复记录
        df = df.drop_duplicates()
        print(f"删除重复后剩余: {len(df)} 条记录")

        # 2. 处理缺失值
        # 数值特征用中位数填充
        for col in self.SENSORY_FEATURES:
            if col in df.columns:
                if df[col].isnull().any():
                    median_val = df[col].median()
                    df[col].fillna(median_val, inplace=True)
                    print(f"  {col}: 用中位数 {median_val:.2f} 填充缺失值")

        # 分类特征用众数填充
        for col in self.CATEGORICAL_FEATURES:
            if col in df.columns:
                if df[col].isnull().any():
                    mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                    df[col].fillna(mode_val, inplace=True)
                    print(f"  {col}: 用众数 '{mode_val}' 填充缺失值")

        # 3. 数据验证：确保感官评分在合理范围内 (0-10)
        for col in self.SENSORY_FEATURES:
            if col in df.columns:
                # 移除评分超出合理范围的记录
                before_count = len(df)
                df = df[(df[col] >= 0) & (df[col] <= 10)]
                after_count = len(df)
                if before_count != after_count:
                    print(f"  {col}: 移除了 {before_count - after_count} 条异常记录")

        print(f"数据清洗完成: {len(df)} 条有效记录")
        return df

    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        对分类特征进行标签编码

        Args:
            df: 清洗后的DataFrame

        Returns:
            编码后的DataFrame
        """
        print("正在进行分类特征编码...")

        df_encoded = df.copy()

        for col in self.CATEGORICAL_FEATURES:
            if col in df.columns:
                # 创建标签编码器
                le = LabelEncoder()
                df_encoded[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le

                print(f"  {col}: {len(le.classes_)} 个不同类别")
                print(f"    前5类: {list(le.classes_[:5])}")

        return df_encoded

    def normalize_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        标准化感官评分特征

        Args:
            df: 预处理后的DataFrame

        Returns:
            (标准化后的特征矩阵, 原始特征矩阵)
        """
        print("正在进行特征标准化...")

        # 提取数值特征
        X_raw = df[self.SENSORY_FEATURES].values
        print(f"  特征维度: {X_raw.shape}")

        # 标准化
        X_scaled = self.scaler.fit_transform(X_raw)

        # 显示统计信息
        print(f"  标准化前均值: {X_raw.mean(axis=0)[:5]}...")
        print(f"  标准化后均值: {X_scaled.mean(axis=0)[:5]}...")
        print(f"  标准化后标准差: {X_scaled.std(axis=0)[:5]}...")

        return X_scaled, X_raw

    def get_feature_info(self) -> Dict:
        """
        获取特征信息统计

        Returns:
            特征信息字典
        """
        if self.processed_data is None:
            raise ValueError("请先调用 process_data() 方法")

        info = {
            'sensory_features': self.SENSORY_FEATURES,
            'categorical_features': self.CATEGORICAL_FEATURES,
            'input_dim': len(self.SENSORY_FEATURES),
            'label_encoders': {col: list(le.classes_) for col, le in self.label_encoders.items()}
        }

        return info

    def process_data(self) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        完整的数据处理流程

        Returns:
            (标准化特征矩阵, 预处理后的完整DataFrame)
        """
        print("=" * 50)
        print("开始数据处理流程")
        print("=" * 50)

        # 1. 加载数据
        df = self.load_data()

        # 2. 数据清洗
        df_clean = self.clean_data(df)

        # 3. 分类特征编码
        df_encoded = self.encode_categorical_features(df_clean)

        # 4. 特征标准化
        X_scaled, X_raw = self.normalize_features(df_encoded)

        # 5. 保存处理结果
        self.processed_data = df_encoded

        print("=" * 50)
        print("数据处理完成！")
        print("=" * 50)

        return X_scaled, df_encoded

    def get_coffee_by_id(self, coffee_id: int) -> pd.Series:
        """
        根据索引获取咖啡信息

        Args:
            coffee_id: 咖啡索引

        Returns:
            咖啡信息Series
        """
        if self.processed_data is None:
            raise ValueError("请先调用 process_data() 方法")

        if coffee_id >= len(self.processed_data):
            raise IndexError(f"咖啡ID {coffee_id} 超出范围")

        return self.processed_data.iloc[coffee_id]

    def get_coffee_list(self, limit: int = 10) -> List[Dict]:
        """
        获取咖啡列表（用于前端展示）

        Args:
            limit: 返回数量限制

        Returns:
            咖啡信息字典列表
        """
        if self.processed_data is None:
            raise ValueError("请先调用 process_data() 方法")

        coffee_list = []
        for idx in range(min(limit, len(self.processed_data))):
            row = self.processed_data.iloc[idx]
            coffee_info = {
                'id': idx,
                'name': row.get('Coffee Name', f'Coffee #{idx}'),
                'country': row.get('Country of Origin', 'Unknown'),
                'variety': row.get('Variety', 'Unknown'),
                'processing': row.get('Processing Method', 'Unknown'),
                'overall_score': row.get('Cupper Points', 0)
            }
            coffee_list.append(coffee_info)

        return coffee_list


if __name__ == "__main__":
    # 示例用法
    loader = CoffeeDataLoader()
    try:
        X_scaled, df_processed = loader.process_data()
        print(f"\n处理完成！")
        print(f"特征矩阵形状: {X_scaled.shape}")
        print(f"咖啡记录数: {len(df_processed)}")

        # 显示前5条记录的基本信息
        print("\n前5款咖啡:")
        for i in range(5):
            coffee = loader.get_coffee_by_id(i)
            print(f"{i}: {coffee.get('Country of Origin', 'Unknown')} - "
                  f"{coffee.get('Variety', 'Unknown')} - "
                  f"评分: {coffee.get('Cupper Points', 0)}")

    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("请确保数据文件路径正确")
