#!/usr/bin/env python
"""
快速启动脚本
用于快速初始化和启动咖啡推荐系统
"""

import os
import sys
import argparse
from datetime import datetime


def check_dependencies():
    """检查依赖是否安装"""
    print("正在检查依赖...")
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow {tf.__version__}")
    except ImportError:
        print("✗ TensorFlow 未安装，请运行: pip install tensorflow")
        return False

    try:
        import flask
        print(f"✓ Flask {flask.__version__}")
    except ImportError:
        print("✗ Flask 未安装，请运行: pip install flask")
        return False

    try:
        import sklearn
        print(f"✓ Scikit-learn {sklearn.__version__}")
    except ImportError:
        print("✗ Scikit-learn 未安装，请运行: pip install scikit-learn")
        return False

    print("✓ 所有依赖检查通过！\n")
    return True


def check_data_file():
    """检查数据文件是否存在"""
    data_path = 'data/coffee_data.csv'
    if os.path.exists(data_path):
        print(f"✓ 发现数据文件: {data_path}")
        return True
    else:
        print(f"⚠ 未找到数据文件: {data_path}")
        print("  请将CQI咖啡数据集放置在此路径")
        print("  或使用 --generate-sample 参数生成示例数据\n")
        return False


def generate_sample_data():
    """生成示例数据用于测试"""
    print("正在生成示例数据...")
    import numpy as np
    import pandas as pd

    # 创建示例咖啡数据
    np.random.seed(42)
    n_samples = 100

    data = {
        'Coffee Name': [f'Sample Coffee {i+1}' for i in range(n_samples)],
        'Country of Origin': np.random.choice(['Ethiopia', 'Brazil', 'Colombia', 'Kenya', 'Guatemala'], n_samples),
        'Variety': np.random.choice(['Arabica', 'Robusta', 'Typica', 'Bourbon'], n_samples),
        'Processing Method': np.random.choice(['Washed', 'Natural', 'Honey'], n_samples),
        'Owner/Farm': [f'Farm {i+1}' for i in range(n_samples)],

        # 10维感官评分 (0-10)
        'Aroma': np.random.uniform(6.0, 9.0, n_samples).round(2),
        'Flavor': np.random.uniform(6.0, 9.0, n_samples).round(2),
        'Aftertaste': np.random.uniform(6.0, 9.0, n_samples).round(2),
        'Acidity': np.random.uniform(5.5, 8.5, n_samples).round(2),
        'Body': np.random.uniform(6.0, 8.8, n_samples).round(2),
        'Balance': np.random.uniform(6.5, 9.0, n_samples).round(2),
        'Uniformity': np.random.uniform(6.0, 9.5, n_samples).round(2),
        'Clean Cup': np.random.uniform(6.5, 9.5, n_samples).round(2),
        'Sweetness': np.random.uniform(6.0, 9.0, n_samples).round(2),
        'Cupper Points': np.random.uniform(6.5, 9.0, n_samples).round(2),
    }

    df = pd.DataFrame(data)

    # 确保data目录存在
    os.makedirs('data', exist_ok=True)

    # 保存数据
    df.to_csv('data/coffee_data.csv', index=False)
    print(f"✓ 示例数据已生成: data/coffee_data.csv")
    print(f"  包含 {n_samples} 条咖啡记录\n")
    return True


def initialize_system():
    """初始化系统（训练模型等）"""
    print("正在初始化系统...")
    print("这可能需要几分钟时间...\n")

    try:
        from recommender import CoffeeRecommender

        recommender = CoffeeRecommender()
        recommender.initialize(force_retrain=True)

        print("\n✓ 系统初始化完成！")
        return True

    except Exception as e:
        print(f"\n✗ 系统初始化失败: {e}")
        return False


def start_server():
    """启动Flask服务器"""
    print("\n" + "="*60)
    print("启动 DeepFlavor Coffee 推荐系统")
    print("="*60)

    # 设置环境变量
    os.environ['FLASK_ENV'] = 'development'
    os.environ['PORT'] = '5000'

    # 启动应用
    from app import app
    app.run(host='0.0.0.0', port=5000, debug=True)


def main():
    parser = argparse.ArgumentParser(description='DeepFlavor Coffee 推荐系统')
    parser.add_argument('--check-deps', action='store_true', help='仅检查依赖')
    parser.add_argument('--generate-sample', action='store_true', help='生成示例数据')
    parser.add_argument('--skip-init', action='store_true', help='跳过系统初始化')

    args = parser.parse_args()

    print("\n" + "="*60)
    print("DeepFlavor Coffee Recommender")
    print("基于 1D-CNN 自编码器的精品咖啡深度推荐系统")
    print("="*60 + "\n")

    # 检查依赖
    if not check_dependencies():
        print("\n请先安装所需依赖:")
        print("pip install -r requirements.txt")
        sys.exit(1)

    # 生成示例数据（如果需要）
    if args.generate_sample or not check_data_file():
        if not args.generate_sample:
            response = input("\n是否生成示例数据进行测试？(y/n): ")
            if response.lower() == 'y':
                generate_sample_data()
        else:
            generate_sample_data()

    # 初始化系统（除非跳过）
    if not args.skip_init:
        if not initialize_system():
            sys.exit(1)
    else:
        print("⚠ 跳过系统初始化，仅启动服务器")

    # 启动服务器
    try:
        start_server()
    except KeyboardInterrupt:
        print("\n\n服务器已停止")
    except Exception as e:
        print(f"\n\n启动服务器失败: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
