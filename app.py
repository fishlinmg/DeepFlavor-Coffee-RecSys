"""
Flask Web应用
精品咖啡深度风味推荐系统 - Web界面
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for
import json
import os
import sys
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from recommender import CoffeeRecommender
from data_loader import CoffeeDataLoader

# 创建Flask应用
app = Flask(__name__)
app.config['SECRET_KEY'] = 'deepflavor-coffee-recsys-2023'

# 全局推荐系统实例
recommender = None


def get_recommender():
    """获取推荐系统实例（单例模式）"""
    global recommender
    if recommender is None:
        print("正在初始化推荐系统...")
        recommender = CoffeeRecommender()
        try:
            recommender.initialize()
            print("推荐系统初始化完成！")
        except FileNotFoundError as e:
            print(f"初始化失败: {e}")
            raise
    return recommender


@app.route('/')
def index():
    """首页：数据集概览和功能入口"""
    try:
        rec = get_recommender()
        coffee_list = rec.data_loader.get_coffee_list(limit=12)

        # 获取统计信息
        total_coffees = len(rec.df_processed)
        unique_countries = rec.df_processed['Country of Origin'].nunique()
        unique_varieties = rec.df_processed['Variety'].nunique()
        unique_processing = rec.df_processed['Processing Method'].nunique()

        # 计算平均评分
        avg_score = rec.df_processed['Cupper Points'].mean()

        stats = {
            'total_coffees': total_coffees,
            'unique_countries': unique_countries,
            'unique_varieties': unique_varieties,
            'unique_processing': unique_processing,
            'avg_score': avg_score
        }

        return render_template('index.html',
                             coffee_list=coffee_list,
                             stats=stats)
    except Exception as e:
        return f"<h1>错误</h1><p>系统初始化失败: {str(e)}</p><p>请确保数据文件存在并查看控制台输出。</p>", 500


@app.route('/search_by_coffee', methods=['POST'])
def search_by_coffee():
    """根据咖啡ID进行相似推荐"""
    try:
        data = request.get_json()
        coffee_id = int(data.get('coffee_id'))
        method = data.get('method', 'deep')
        top_k = int(data.get('top_k', 5))

        rec = get_recommender()
        recommendations = rec.recommend_by_coffee_id(coffee_id, method=method, top_k=top_k)

        # 获取雷达图数据
        coffee_ids = [rec['id'] for rec in recommendations]
        radar_data = rec.get_radar_chart_data(coffee_ids)

        return jsonify({
            'success': True,
            'recommendations': recommendations,
            'radar_data': radar_data
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/search_by_preferences', methods=['POST'])
def search_by_preferences():
    """根据用户偏好进行推荐"""
    try:
        data = request.get_json()
        preferences = data.get('preferences', [])

        if len(preferences) != 10:
            return jsonify({
                'success': False,
                'error': '偏好评分必须包含10个维度'
            }), 400

        method = data.get('method', 'deep')
        top_k = int(data.get('top_k', 5))

        rec = get_recommender()
        recommendations = rec.recommend_by_preferences(
            preferences, method=method, top_k=top_k
        )

        # 获取雷达图数据
        coffee_ids = [rec['id'] for rec in recommendations]
        radar_data = rec.get_radar_chart_data(coffee_ids)

        return jsonify({
            'success': True,
            'recommendations': recommendations,
            'radar_data': radar_data
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/coffee_list')
def coffee_list():
    """咖啡列表页面"""
    try:
        rec = get_recommender()
        limit = int(request.args.get('limit', 20))
        coffee_list = rec.data_loader.get_coffee_list(limit=limit)

        return render_template('coffee_list.html', coffee_list=coffee_list)

    except Exception as e:
        return f"<h1>错误</h1><p>{str(e)}</p>", 500


@app.route('/analysis')
def analysis():
    """数据分析页面"""
    try:
        rec = get_recommender()

        # 获取各产地平均评分排名
        country_scores = rec.df_processed.groupby('Country of Origin')['Cupper Points'].agg(['mean', 'count']).reset_index()
        country_scores = country_scores.sort_values('mean', ascending=False).head(15)
        country_data = {
            'countries': country_scores['Country of Origin'].tolist(),
            'scores': country_scores['mean'].round(2).tolist(),
            'counts': country_scores['count'].tolist()
        }

        # 获取处理法分布
        processing_dist = rec.df_processed['Processing Method'].value_counts().head(10)
        processing_data = {
            'methods': processing_dist.index.tolist(),
            'counts': processing_dist.values.tolist()
        }

        return render_template('analysis.html',
                             country_data=country_data,
                             processing_data=processing_data)

    except Exception as e:
        return f"<h1>错误</h1><p>{str(e)}</p>", 500


@app.route('/evaluation')
def evaluation():
    """模型评估页面"""
    try:
        rec = get_recommender()
        comparison = rec.compare_methods()

        return render_template('evaluation.html',
                             comparison=comparison)

    except Exception as e:
        return f"<h1>错误</h1><p>{str(e)}</p>", 500


@app.route('/api/coffee/<int:coffee_id>')
def api_coffee(coffee_id):
    """API: 获取特定咖啡信息"""
    try:
        rec = get_recommender()
        coffee_info = rec._get_coffee_info(coffee_id)
        return jsonify(coffee_info)
    except Exception as e:
        return jsonify({'error': str(e)}), 404


@app.route('/api/countries')
def api_countries():
    """API: 获取所有国家列表"""
    try:
        rec = get_recommender()
        countries = rec.df_processed['Country of Origin'].unique().tolist()
        return jsonify(countries)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/processing_methods')
def api_processing_methods():
    """API: 获取所有处理法列表"""
    try:
        rec = get_recommender()
        methods = rec.df_processed['Processing Method'].unique().tolist()
        return jsonify(methods)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    """404错误处理"""
    return render_template('error.html',
                         error_code=404,
                         error_message="页面未找到"), 404


@app.errorhandler(500)
def internal_error(error):
    """500错误处理"""
    return render_template('error.html',
                         error_code=500,
                         error_message="服务器内部错误"), 500


@app.route('/health')
def health_check():
    """健康检查端点"""
    try:
        rec = get_recommender()
        status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'total_coffees': len(rec.df_processed),
            'model_loaded': rec.autoencoder.model is not None,
            'knn_built': rec.knn_original is not None and rec.knn_deep is not None
        }
        return jsonify(status)
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'timestamp': datetime.now().isoformat(),
            'error': str(e)
        }), 500


if __name__ == '__main__':
    print("=" * 60)
    print("DeepFlavor Coffee Recommender System")
    print("基于 1D-CNN 自编码器的精品咖啡深度推荐系统")
    print("=" * 60)

    # 检查必要文件
    if not os.path.exists('data/coffee_data.csv'):
        print("警告: 未找到数据文件 'data/coffee_data.csv'")
        print("请确保CQI咖啡数据集在此路径。")
        print("系统启动后将尝试初始化...")

    # 启动Flask应用
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'

    print(f"\n正在启动服务器...")
    print(f"访问地址: http://localhost:{port}")
    print(f"调试模式: {'开启' if debug else '关闭'}")

    app.run(host='0.0.0.0', port=port, debug=debug)
