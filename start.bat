@echo off
echo ================================================
echo DeepFlavor Coffee Recommender System
echo 基于 1D-CNN 自编码器的精品咖啡深度推荐系统
echo ================================================
echo.

REM 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到Python，请先安装Python 3.8+
    pause
    exit /b 1
)

REM 检查虚拟环境
if exist venv\Scripts\activate.bat (
    echo [提示] 检测到虚拟环境，正在激活...
    call venv\Scripts\activate.bat
) else (
    echo [提示] 未检测到虚拟环境，使用全局Python环境
)

REM 安装依赖
echo [步骤 1/4] 检查并安装依赖...
pip install -r requirements.txt >nul 2>&1

REM 检查数据文件
if not exist data\coffee_data.csv (
    echo [步骤 2/4] 未找到数据文件，正在生成示例数据...
    python run.py --generate-sample --skip-init
) else (
    echo [步骤 2/4] 发现数据文件
)

REM 初始化系统
echo [步骤 3/4] 正在初始化系统（可能需要几分钟）...
python run.py --skip-init >nul 2>&1

echo.
echo [步骤 4/4] 启动服务器...
echo.
echo ================================================
echo 服务器已启动！
echo 访问地址: http://localhost:5000
echo 按 Ctrl+C 停止服务器
echo ================================================
echo.

REM 启动Flask应用
python app.py
