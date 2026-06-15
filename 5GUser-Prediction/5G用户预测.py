import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 数据加载和探索
print("="*50)
print("1. 数据加载和探索")
print("="*50)

try:
    df = pd.read_csv('train.csv')
    print(f"数据集形状: {df.shape}")
    print(f"\n数据集前5行:")
    print(df.head())
    print(f"\n目标变量分布:")
    print(df['target'].value_counts())
except FileNotFoundError:
    print("错误: 找不到 train.csv 文件！")
    exit()

# 2. 数据预处理
print("\n" + "="*50)
print("2. 数据预处理")
print("="*50)

# 分离特征和目标变量
X = df.drop(['id', 'target'], axis=1) if 'id' in df.columns else df.drop('target', axis=1)
y = df['target']

# 识别类别特征和数值特征
cat_features = [col for col in X.columns if col.startswith('cat_')]
num_features = [col for col in X.columns if col.startswith('num_')]

print(f"\n类别特征数量: {len(cat_features)}")
print(f"数值特征数量: {len(num_features)}")

# 处理缺失值
print("\n处理缺失值:")
for col in X.columns:
    if X[col].isnull().sum() > 0:
        print(f"{col}: {X[col].isnull().sum()} 个缺失值")
        if col in cat_features:
            X[col] = X[col].fillna(X[col].mode()[0])
        else:
            X[col] = X[col].fillna(X[col].median())
print("缺失值处理完成！")

# 标签编码类别特征
for col in cat_features:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\n训练集大小: {X_train.shape}")
print(f"测试集大小: {X_test.shape}")

# 3. 算法1: 随机森林
print("\n" + "="*50)
print("3. 算法1: 随机森林 (Random Forest)")
print("="*50)

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_pred_proba = rf_model.predict_proba(X_test)[:, 1]

print("\n随机森林分类报告:")
print(classification_report(y_test, rf_pred))
print(f"准确率: {accuracy_score(y_test, rf_pred):.4f}")
print(f"AUC-ROC: {roc_auc_score(y_test, rf_pred_proba):.4f}")

# 随机森林特征重要性
plt.figure(figsize=(12, 8))
feature_importance = pd.Series(rf_model.feature_importances_, index=X.columns)
feature_importance.nlargest(20).plot(kind='barh')
plt.title('随机森林 - Top 20 特征重要性')
plt.tight_layout()
plt.savefig('rf_feature_importance.png')
print("特征重要性图表已保存为 rf_feature_importance.png")

# 4. 算法2: LightGBM
print("\n" + "="*50)
print("4. 算法2: LightGBM")
print("="*50)

lgb_model = lgb.LGBMClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

# 修复：简化fit调用，不使用eval_set和verbose参数
lgb_model.fit(X_train, y_train)
lgb_pred = lgb_model.predict(X_test)
lgb_pred_proba = lgb_model.predict_proba(X_test)[:, 1]

print("\nLightGBM分类报告:")
print(classification_report(y_test, lgb_pred))
print(f"准确率: {accuracy_score(y_test, lgb_pred):.4f}")
print(f"AUC-ROC: {roc_auc_score(y_test, lgb_pred_proba):.4f}")

# LightGBM特征重要性
plt.figure(figsize=(12, 8))
lgb.plot_importance(lgb_model, max_num_features=20, importance_type='split')
plt.title('LightGBM - Top 20 特征重要性')
plt.tight_layout()
plt.savefig('lgb_feature_importance.png')
print("特征重要性图表已保存为 lgb_feature_importance.png")

# 5. 模型对比
print("\n" + "="*50)
print("5. 模型性能对比")
print("="*50)

# ROC曲线对比
plt.figure(figsize=(10, 8))

# 随机森林
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_pred_proba)
plt.plot(fpr_rf, tpr_rf, label=f'随机森林 (AUC = {roc_auc_score(y_test, rf_pred_proba):.4f})')

# LightGBM
fpr_lgb, tpr_lgb, _ = roc_curve(y_test, lgb_pred_proba)
plt.plot(fpr_lgb, tpr_lgb, label=f'LightGBM (AUC = {roc_auc_score(y_test, lgb_pred_proba):.4f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('假正率 (False Positive Rate)')
plt.ylabel('真正率 (True Positive Rate)')
plt.title('ROC曲线对比')
plt.legend()
plt.tight_layout()
plt.savefig('roc_comparison.png')
print("ROC曲线对比图已保存为 roc_comparison.png")

# 混淆矩阵对比
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 随机森林混淆矩阵
cm_rf = confusion_matrix(y_test, rf_pred)
sns.heatmap(cm_rf, annot=True, fmt='d', ax=axes[0], cmap='Blues')
axes[0].set_title('随机森林 - 混淆矩阵')
axes[0].set_xlabel('预测标签')
axes[0].set_ylabel('真实标签')

# LightGBM混淆矩阵
cm_lgb = confusion_matrix(y_test, lgb_pred)
sns.heatmap(cm_lgb, annot=True, fmt='d', ax=axes[1], cmap='Greens')
axes[1].set_title('LightGBM - 混淆矩阵')
axes[1].set_xlabel('预测标签')
axes[1].set_ylabel('真实标签')

plt.tight_layout()
plt.savefig('confusion_matrix_comparison.png')
print("混淆矩阵对比图已保存为 confusion_matrix_comparison.png")

# 性能指标汇总
print("\n性能指标汇总:")
print("-"*50)
print(f"随机森林:")
print(f"  准确率: {accuracy_score(y_test, rf_pred):.4f}")
print(f"  AUC-ROC: {roc_auc_score(y_test, rf_pred_proba):.4f}")

print(f"\nLightGBM:")
print(f"  准确率: {accuracy_score(y_test, lgb_pred):.4f}")
print(f"  AUC-ROC: {roc_auc_score(y_test, lgb_pred_proba):.4f}")

# 6. 预测示例
print("\n" + "="*50)
print("6. 预测示例")
print("="*50)

# 使用前5个样本进行预测
sample_data = X_test.head()
print("\n示例数据:")
print(sample_data)

rf_sample_pred = rf_model.predict(sample_data)
rf_sample_pred_proba = rf_model.predict_proba(sample_data)

lgb_sample_pred = lgb_model.predict(sample_data)
lgb_sample_pred_proba = lgb_model.predict_proba(sample_data)

print("\n随机森林预测结果:")
for i in range(len(sample_data)):
    print(f"样本 {i+1}: 预测类别={rf_sample_pred[i]}, 概率={rf_sample_pred_proba[i][1]:.4f}")

print("\nLightGBM预测结果:")
for i in range(len(sample_data)):
    print(f"样本 {i+1}: 预测类别={lgb_sample_pred[i]}, 概率={lgb_sample_pred_proba[i][1]:.4f}")

print("\n" + "="*50)
print("分析完成！")
print("="*50)