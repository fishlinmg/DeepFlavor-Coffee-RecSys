#!/bin/bash

echo "================================================"
echo "DeepFlavor Coffee Recommender System"
echo "基于 1D-CNN 自编码器的精品咖啡深度推荐系统"
echo "================================================"
echo

# 检查Python
if ! command -v python3 &> /dev/null; then
    echo "[错误] 未找到Python3，请先安装Python 3.8+"
    exit 1
fi

# 检查虚拟环境
if [ -d "venv" ]; then
    echo "[提示] 检测到虚拟环境，正在激活..."
    source venv/bin/activate
else
    echo "[提示] 未检测到虚拟环境，使用全局Python环境"
fi

# 安装依赖
echo "[步骤 1/4] 检查并安装依赖..."
pip3 install -r requirements.txt > /dev/null 2>&1

# 检查数据文件
if [ ! -f "data/coffee_data.csv" ]; then
    echo "[步骤 2/4] 未找到数据文件，正在生成示例数据..."
    python3 run.py --generate-sample --skip-init > /dev/null
else
    echo "[步骤 2/4] 发现数据文件"
fi

# 初始化系统
echo "[步骤 3/4] 正在初始化系统（可能需要几分钟）..."
python3 run.py --skip-init > /dev/null

echo
echo "[步骤 4/4] 启动服务器..."
echo
echo "================================================"
echo "服务器已启动！"
echo "访问地址: http://localhost:5000"
echo "按 Ctrl+C 停止服务器"
echo "================================================"
echo

# 启动Flask应用
python3 app.py
