 农业灌溉需求预测模型

这是一个基于机器学习的农业灌溉需求预测系统，通过分析土壤、气候、作物等多维度特征，智能预测农田的灌溉需求量。

 项目特点

完整的 ML Pipeline：涵盖数据清洗、特征工程、模型训练、评估与预测的全流程

智能特征工程：自动构造土壤-气候交互特征，挖掘潜在规律

类别不平衡处理：采用 SMOTE 过采样技术，解决灌溉需求分布不均问题

多模型对比：集成随机森林、梯度提升、逻辑回归三种算法，自动选择最优模型

生产级代码：包含模型持久化、预处理复用，支持直接部署

🛠 技术栈

核心语言：Python 3.8+

机器学习：Scikit-learn、Imbalanced-learn

数据处理：Pandas、NumPy

可视化：Matplotlib、Seaborn

模型持久化：Joblib

1. 克隆项目
bash
git clone https://github.com/qq94398/irrigation-prediction.git
cd irrigation-prediction
2. 安装依赖
bash
pip install -r requirements.txt
3. 准备数据

确保项目根目录包含以下文件：

纯文本
├── train.csv              # 训练数据
├── test.csv               # 测试数据
└── sample_submission.csv  # 提交样例
4. 运行训练与预测
bash
python file_util.py

运行完成后将生成：

submission.csv- 最终预测结果

best_model.pkl- 训练好的最佳模型

label_encoders.pkl- 分类变量编码器

scaler.pkl- 数值特征标准化器

  数据说明

土壤特征​:Soil_Type, Soil_pH, Soil_Moisture, Organic_Carbon
气候特征:Temperature_C, Humidity, Rainfall_mm, Sunlight_Hours
作物特征​:Crop_Type, Crop_Growth_Stage, Field_Area_hectare
灌溉特征​:Irrigation_Type, Previous_Irrigation_mm, Water_Source
目标变量​:Irrigation_Need（需预测的灌溉需求等级）

🔧 核心功能模块
1. 智能特征工程

自动生成高阶特征：
·土壤湿度与降雨量比率
·温度-湿度交互项
·日照时数与温度比率
·历史灌溉与土壤湿度关系

2. 鲁棒性预处理
·自动处理分类变量的未知类别（测试集兼容）
·数值特征标准化
·缺失值检测与报告

3. 模型训练策略

·SMOTE 过采样：解决类别不平衡
·分层抽样：保证训练/验证集分布一致
·早停机制：防止过拟合
·并行计算：充分利用 CPU 资源
 模型表现

项目会自动比较三种模型性能，选择最优模型：

模型                  特点




RandomForest    抗过拟合能力强，可解释性好




GradientBoosting    精度高，收敛快




LogisticRegression   简单高效，基线模型

注：使用平衡准确率（Balanced Accuracy）作为主要评估指标，更适合不平衡数据集。

 项目结构
纯文本
irrigation-prediction/
├── file_util.py           # 主训练与预测脚本
├── requirements.txt      # 项目依赖
├── README.md            # 项目说明
├── train.csv            # 训练数据
├── test.csv             # 测试数据
├── submission.csv       # 预测结果（生成）
├── best_model.pkl       # 最佳模型（生成）
├── label_encoders.pkl   # 编码器（生成）
└── scaler.pkl           # 标准化器（生成）
 依赖列表（requirements.txt）
txt
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
imbalanced-learn>=0.8.0
joblib>=1.1.0
matplotlib>=3.4.0
seaborn>=0.11.0
 使用预测模型

训练完成后，可使用保存的模型进行新数据预测：

python
import joblib
import pandas as pd

# 加载模型和预处理器
model = joblib.load('best_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')
scaler = joblib.load('scaler.pkl')

# 对新数据进行预测
# new_data = pd.read_csv('new_data.csv')
# predictions = model.predict(new_data)
📄 许可证

MIT License © 2026
