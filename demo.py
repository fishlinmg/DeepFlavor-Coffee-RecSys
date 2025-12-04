#!/usr/bin/env python
"""
演示脚本
展示咖啡推荐系统的基本功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from recommender import CoffeeRecommender
import numpy as np


def demo_basic_features():
    """演示基本功能"""
    print("=" * 60)
    print("DeepFlavor Coffee 推荐系统 - 功能演示")
    print("=" * 60 + "\n")

    print("正在初始化推荐系统...")
    print("（这可能需要几分钟，特别是第一次运行时）\n")

    try:
        # 初始化推荐系统
        recommender = CoffeeRecommender()
        recommender.initialize()

        print("\n" + "=" * 60)
        print("演示 1: 根据咖啡ID推荐相似咖啡")
        print("=" * 60 + "\n")

        # 推荐相似的咖啡
        recommendations = recommender.recommend_by_coffee_id(
            coffee_id=0,
            method='deep',
            top_k=5
        )

        print("为您推荐以下相似的咖啡：\n")
        for i, rec in enumerate(recommendations, 1):
            coffee = rec['coffee']
            print(f"{i}. {coffee['name']}")
            print(f"   产地: {coffee['country']}")
            print(f"   品种: {coffee['variety']}")
            print(f"   处理法: {coffee['processing']}")
            print(f"   相似度: {rec['similarity']:.4f} ({rec['similarity']*100:.2f}%)")
            print(f"   总体评分: {coffee['overall_score']:.2f}")
            print()

        print("\n" + "=" * 60)
        print("演示 2: 根据用户偏好推荐咖啡")
        print("=" * 60 + "\n")

        # 用户偏好示例：高酸度、中等醇度、平衡
        user_preferences = [
            8.5,  # Aroma (干香)
            8.0,  # Flavor (风味)
            7.5,  # Aftertaste (余韵)
            9.0,  # Acidity (酸度) - 高酸度
            6.5,  # Body (醇度) - 中等醇度
            8.0,  # Balance (平衡) - 高平衡
            8.5,  # Uniformity (一致性)
            9.0,  # Clean Cup (干净度) - 高干净度
            7.0,  # Sweetness (甜度) - 中等甜度
            8.0   # Cupper Points (杯测师评分)
        ]

        feature_names = ['干香', '风味', '余韵', '酸度', '醇度', '平衡', '一致性', '干净度', '甜度', '杯测师评分']

        print("用户偏好设置：")
        for name, value in zip(feature_names, user_preferences):
            print(f"  {name}: {value}")
        print()

        recommendations = recommender.recommend_by_preferences(
            preferences=user_preferences,
            method='deep',
            top_k=5
        )

        print("基于您的偏好，为您推荐以下咖啡：\n")
        for i, rec in enumerate(recommendations, 1):
            coffee = rec['coffee']
            print(f"{i}. {coffee['name']}")
            print(f"   产地: {coffee['country']}")
            print(f"   品种: {coffee['variety']}")
            print(f"   相似度: {rec['similarity']:.4f} ({rec['similarity']*100:.2f}%)")
            print()

        print("\n" + "=" * 60)
        print("演示 3: 模型性能评估")
        print("=" * 60 + "\n")

        # 对比两种推荐方法
        comparison = recommender.compare_methods()

        print("\n实验结论：")
        improvement_pct = (comparison['improvement']['similarity_gain'] /
                          comparison['original']['avg_similarity'] * 100)

        print(f"✓ 深度特征推荐相比原始特征推荐：")
        print(f"  - 平均相似度提升: {comparison['improvement']['similarity_gain']:.4f}")
        print(f"  - 相对提升: {improvement_pct:.2f}%")
        print(f"  - MSE 降低: {comparison['improvement']['mse_reduction']:.6f}")

        print("\n" + "=" * 60)
        print("演示完成！")
        print("=" * 60)
        print("\n要启动Web界面，请运行: python app.py")
        print("然后访问: http://localhost:5000")

    except FileNotFoundError as e:
        print(f"\n❌ 错误: {e}")
        print("\n请确保您已:")
        print("1. 将CQI咖啡数据集放置在 'data/coffee_data.csv'")
        print("2. 或运行: python run.py --generate-sample 生成示例数据")
        return False

    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == '__main__':
    success = demo_basic_features()
    sys.exit(0 if success else 1)
