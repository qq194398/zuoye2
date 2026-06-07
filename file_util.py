import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import warnings
import joblib

warnings.filterwarnings('ignore')

# ===================== 1. 加载数据 =====================
print("加载数据...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
sample_sub = pd.read_csv('sample_submission.csv')

print(f"训练集形状: {train_df.shape}")
print(f"测试集形状: {test_df.shape}")


# ===================== 2. 数据探索 =====================
print("\n数据探索:")
print("训练集列名:", train_df.columns.tolist())
print("测试集列名:", test_df.columns.tolist())

# 检查缺失值
print("\n训练集缺失值:\n", train_df.isnull().sum())
print("\n测试集缺失值:\n", test_df.isnull().sum())


# ===================== 3. 数据预处理（含特征工程） =====================
def preprocess_data(df, is_train=True, label_encoders=None, scaler=None):
    df_processed = df.copy()

    # 保存ID
    ids = None
    if 'id' in df_processed.columns:
        ids = df_processed['id']
        df_processed = df_processed.drop('id', axis=1)
    else:
        ids = None

    # 处理分类变量（土壤类型、作物类型等）
    categorical_cols = ['Soil_Type', 'Crop_Type', 'Crop_Growth_Stage', 'Season',
                       'Irrigation_Type', 'Water_Source', 'Mulching_Used', 'Region']
    if label_encoders is None:
        label_encoders = {}
        for col in categorical_cols:
            if col in df_processed.columns:
                le = LabelEncoder()
                # 处理未知类别（测试集可能出现训练集没有的类别）
                df_processed[col] = df_processed[col].astype(str)
                df_processed[col] = le.fit_transform(df_processed[col])
                label_encoders[col] = le
    else:
        for col in categorical_cols:
            if col in df_processed.columns and col in label_encoders:
                le = label_encoders[col]
                # 将测试集中的未知类别标记为"unknown"
                known_categories = set(le.classes_)
                df_processed[col] = df_processed[col].apply(
                    lambda x: x if x in known_categories else 'unknown'
                )
                # 如果"unknown"不在编码器中，添加它
                if 'unknown' not in le.classes_:
                    le.classes_ = np.append(le.classes_, 'unknown')
                df_processed[col] = le.transform(df_processed[col])

    # 处理目标变量（仅训练集）
    target = None
    if is_train and 'Irrigation_Need' in df_processed.columns:
        target_encoder = LabelEncoder()
        target = target_encoder.fit_transform(df_processed['Irrigation_Need'])
        df_processed = df_processed.drop('Irrigation_Need', axis=1)
    elif is_train:
        target = None

    # 数值特征标准化（土壤pH、湿度、降雨量等）
    numeric_cols = ['Soil_pH', 'Soil_Moisture', 'Organic_Carbon', 'Electrical_Conductivity',
                   'Temperature_C', 'Humidity', 'Rainfall_mm', 'Sunlight_Hours',
                   'Wind_Speed_kmh', 'Field_Area_hectare', 'Previous_Irrigation_mm']
    if scaler is None and is_train:
        scaler = StandardScaler()
        df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])
    elif scaler is not None:
        df_processed[numeric_cols] = scaler.transform(df_processed[numeric_cols])

    # 特征工程（构造新特征）
    # 1. 土壤湿度与降雨量的比率
    df_processed['Moisture_Rain_Ratio'] = df_processed['Soil_Moisture'] / (df_processed['Rainfall_mm'] + 1)  # +1避免除零
    # 2. 温度和湿度的交互
    df_processed['Temp_Humidity_Interaction'] = df_processed['Temperature_C'] * df_processed['Humidity'] / 100
    # 3. 日照时数与温度的比率
    df_processed['Sunlight_Temp_Ratio'] = df_processed['Sunlight_Hours'] / (df_processed['Temperature_C'] + 1)  # +1避免除零
    # 4. 之前灌溉量与土壤湿度的比率
    df_processed['PrevIrrig_Moisture_Ratio'] = df_processed['Previous_Irrigation_mm'] / (df_processed['Soil_Moisture'] + 1)  # +1避免除零

    return df_processed, target, ids, label_encoders, scaler


# 预处理训练数据
print("\n预处理训练数据...")
X_train, y_train, train_ids, label_encoders, scaler = preprocess_data(train_df, is_train=True)

# 处理类别不平衡（SMOTE过采样）
print("\n处理类别不平衡...")
smote = SMOTE(random_state=42, k_neighbors=5)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
print(f"原始类别分布: {np.bincount(y_train)}")
print(f"SMOTE后类别分布: {np.bincount(y_resampled)}")


# ===================== 4. 划分训练集和验证集 =====================
print("\n划分训练集和验证集...")
X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)


# ===================== 5. 训练优化后的模型 =====================
print("\n训练优化后的模型...")
# 定义模型（参数已优化，加快速度）
models = {
    'RandomForest': RandomForestClassifier(
        n_estimators=200,    # 树的数量（增加以提升性能，减少以加快速度）
        max_depth=15,       # 树的最大深度（防止过拟合）
        min_samples_split=5,# 分裂所需最小样本数
        min_samples_leaf=2, # 叶节点最小样本数
        random_state=42,
        class_weight='balanced',  # 平衡类别权重
        n_jobs=-1           # 并行训练（使用所有CPU核心）
    ),
    'GradientBoosting': GradientBoostingClassifier(
        n_estimators=50,    # 减少树的数量以加快训练
        learning_rate=0.2,  # 提高学习率（加快收敛）
        max_depth=3,        # 减少深度（防止过拟合）
        random_state=42,
        subsample=0.8,      # 子采样（加快训练，减少过拟合）
        validation_fraction=0.1,  # 10%数据作为验证集
        n_iter_no_change=5  # 早停机制（验证集性能不变则停止）
    ),
    'LogisticRegression': LogisticRegression(
        max_iter=500,       # 增加迭代次数确保收敛
        random_state=42,
        class_weight='balanced',  # 平衡类别权重
        C=0.1,              # 正则化强度（小值=强正则化）
        solver='saga'       # 适合大规模数据的求解器
    )
}

best_model = None
best_score = 0
best_model_name = ""

for name, model in models.items():
    print(f"\n训练 {name}...")
    model.fit(X_train_final, y_train_final)

    # 在验证集上评估
    y_val_pred = model.predict(X_val)
    score = balanced_accuracy_score(y_val, y_val_pred)
    print(f"{name} 平衡准确率: {score:.4f}")

    # 更新最佳模型
    if score > best_score:
        best_score = score
        best_model = model
        best_model_name = name

print(f"\n最佳模型: {best_model_name}, 验证集平衡准确率: {best_score:.4f}")


# ===================== 6. 在整个训练集上重新训练最佳模型 =====================
print(f"\n在整个训练集上重新训练 {best_model_name}...")
best_model.fit(X_resampled, y_resampled)


# ===================== 7. 预处理测试数据 =====================
print("\n预处理测试数据...")
X_test, _, test_ids, _, _ = preprocess_data(
    test_df,
    is_train=False,
    label_encoders=label_encoders,
    scaler=scaler
)


# ===================== 8. 预测测试集 =====================
print("预测测试集...")
y_test_pred = best_model.predict(X_test)


# ===================== 9. 将数值预测转换回类别标签 =====================
target_encoder = LabelEncoder()
target_encoder.fit(train_df['Irrigation_Need'])  # 用训练集的目标变量拟合编码器
test_predictions = target_encoder.inverse_transform(y_test_pred)


# ===================== 10. 创建提交文件 =====================
print("\n创建提交文件...")
submission_df = pd.DataFrame({
    'id': test_ids,
    'Irrigation_Need': test_predictions
})

# 确保id是整数类型
submission_df['id'] = submission_df['id'].astype(int)

# 保存提交文件
submission_file = 'submission.csv'
submission_df.to_csv(submission_file, index=False)
print(f"提交文件已保存至: {submission_file}")


# ===================== 11. 生成预测结果概览 =====================
print("\n预测结果分布:")
pred_counts = pd.Series(test_predictions).value_counts()
for label, count in pred_counts.items():
    print(f"{label}: {count} ({count / len(test_predictions) * 100:.1f}%)")


# ===================== 12. 特征重要性分析（仅树模型） =====================
if hasattr(best_model, 'feature_importances_'):
    print("\n特征重要性（前20个）:")
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': best_model.feature_importances_
    })
    feature_importance = feature_importance.sort_values(by='importance', ascending=False)
    print(feature_importance.head(20).to_string())


# ===================== 13. 保存模型和预处理器 =====================
print("\n保存模型和预处理器...")
joblib.dump(best_model, 'best_model.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("模型和预处理器已保存!")


# ===================== 14. 输出前10个预测结果示例 =====================
print("\n前10个预测结果示例:")
print(submission_df.head(10).to_string(index=False))