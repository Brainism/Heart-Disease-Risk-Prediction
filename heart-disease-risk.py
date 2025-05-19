import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap

matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

sns.set(style="whitegrid")
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

# CSV 파일 불러오기
df = pd.read_csv(r'C:\Heart-Disease-Risk-Prediction\data\heart_disease_risk_dataset_earlymed.csv')

# 불필요한 공백 제거
df.columns = df.columns.str.strip() 

print("데이터 미리보기:")
print(df.info())
print(df.describe())

# 결측치 처리: 각 변수별 결측치를 평균값으로 대체
df.fillna(df.mean(), inplace=True)

# 이상치 보정: 윈저라이징 (극단치 클리핑)
def winsorize_series(s, lower_quantile=0.01, upper_quantile=0.99):
    lower_bound = s.quantile(lower_quantile)
    upper_bound = s.quantile(upper_quantile)
    return s.clip(lower_bound, upper_bound)

numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    df[col] = winsorize_series(df[col])

# 동적 Binning: 'Age' 컬럼을 5구간으로 분할하고 시각화 ; 구간 경계값 반환
if 'Age' in df.columns:
    df['Age_bin'], bin_edges = pd.qcut(df['Age'], q=5, labels=False, duplicates='drop', retbins=True)
    print("Age 구간 경계값:", bin_edges)

    # 원본 Age 분포
    counts, bin_edges = np.histogram(df['Age'], bins=20)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.figure(figsize=(8,6))
    plt.plot(bin_centers, counts, 'o-', markersize=8)
    plt.title("Original Age Distribution", fontsize=14)
    plt.xlabel("Age", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
    
    # 동적 Binning 적용 결과
    age_bin_counts = df['Age_bin'].value_counts().sort_index()
    plt.figure(figsize=(8,6))
    plt.plot(age_bin_counts.index, age_bin_counts.values, 'o-', markersize=8)
    plt.title("Dynamic Binning Applied", fontsize=14)
    plt.xlabel("Age bin", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# 파생 변수 생성: 'Weight'와 'Height'로 BMI 계산
if set(['Weight', 'Height']).issubset(df.columns):
    df['Height_m'] = df['Height'] / 100
    df['BMI'] = df['Weight'] / (df['Height_m'] ** 2)

# 타겟 변수: 'Heart_Risk'가 존재하는지 확인
if 'Heart_Risk' not in df.columns:
    raise ValueError("타겟 변수 'Heart_Risk'가 존재하지 않습니다.")

# 입력 특성: 'Heart_Risk'를 제외한 나머지 컬럼 사용
feature_cols = ['Chest_Pain', 'Shortness_of_Breath', 'Fatigue', 'Palpitations', 
                'Dizziness', 'Swelling', 'Pain_Arms_Jaw_Back', 'Cold_Sweats_Nausea', 
                'High_BP', 'High_Cholesterol', 'Diabetes', 'Smoking', 'Obesity', 
                'Sedentary_Lifestyle', 'Family_History', 'Chronic_Stress', 'Gender', 'Age']

x = df[feature_cols]
y = df['Heart_Risk']

# Train/Test 분할
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
print("학습 데이터 shape:", X_train.shape)
print("테스트 데이터 shape:", X_test.shape)

# 모델 평가 함수: 테스트 데이터 평가 및 혼동 행렬 시각화
def evaluate_model(model, model_name, X_test, y_test):
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.decision_function(X_test)
        y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # 혼동 행렬 시각화
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("Actual", fontsize=12)
    plt.title(f"{model_name} confusion matrix", fontsize=14)
    plt.show()
    
    return {"model": model_name, "accuracy": acc, "precision": prec, 
            "recall": rec, "f1_score": f1, "auc": roc_auc, "cm": cm}

# 학습 데이터 평가 함수 (과적합 점검)
def evaluate_on_train(model, model_name, X_train, y_train):
    y_pred_train = model.predict(X_train)
    acc = accuracy_score(y_train, y_pred_train)
    prec = precision_score(y_train, y_pred_train)
    rec = recall_score(y_train, y_pred_train)
    f1_val = f1_score(y_train, y_pred_train)
    print(f"{model_name} - 학습 데이터 성능능:")
    print(f"Accuracy: {acc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1_val:.3f}")
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1_val}

# 혼동 행렬 세부 정보 출력 함수
def print_confusion_details(cm, model_name):
    TN, FP, FN, TP = cm.ravel()
    print(f"{model_name} 혼동 행렬 세부 정보:")
    print(f"True Negatives: {TN}, False Positives: {FP}, False Negatives: {FN}, True Positives: {TP}")

# Hybrid Decision Tree 모델 구축 및 평가
hybrid_dt = DecisionTreeClassifier(max_depth=3, random_state=42)
hybrid_dt.fit(X_train, y_train)
metrics_dt = evaluate_model(hybrid_dt, "Hybrid Decision Tree", X_test, y_test)

# XGBoost 모델 구축 및 평가
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)
metrics_xgb = evaluate_model(xgb_model, "XGBoost", X_test, y_test)

# K-Fold Cross Validation (교차 검증 수행)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
scores = cross_val_score(cv_model, x, y, cv=skf, scoring='roc_auc')
print("XGBoost CV AUC Mean:", scores.mean())
print("XGBoost CV AUC Std Dev:", scores.std())

# 최종 학습용 XGBoost 모델 생성 (교차 검증 후 다시 fit)
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)

# 학습 데이터 평가 (과적합 점검)
train_metrics_dt = evaluate_on_train(hybrid_dt, "Hybrid Decision Tree", X_train, y_train)
train_metrics_xgb = evaluate_on_train(xgb_model, "XGBoost", X_train, y_train)

# 테스트 데이터 혼동 행렬 세부 정보 출력
print_confusion_details(metrics_dt['cm'], "Hybrid Decision Tree")
print_confusion_details(metrics_xgb['cm'], "XGBoost")

# 모델 성능 비교 결과 출력
results = pd.DataFrame([metrics_dt, metrics_xgb])
print("모델 성능 비교 결과:")
print(results)

# 슬로프 차트를 이용한 모델 성능 비교 시각화 함수
results = pd.DataFrame([metrics_dt, metrics_xgb])
results_for_plot = results.drop(columns=['cm'], errors='ignore')
results_long = pd.melt(results_for_plot, id_vars='model', var_name='metric', value_name='score')

plt.figure(figsize=(10, 6))
sns.set(style="whitegrid")

palette = {
    'Hybrid Decision Tree': '#1f77b4',
    'XGBoost': '#ff7f0e'
}

sns.lineplot(
    data=results_long,
    x='metric',
    y='score',
    hue='model',
    palette=palette,
    linewidth=2.5,
    marker='o',
    markersize=8
)

for model in results_for_plot['model']:
    subset = results_long[results_long['model'] == model]
    for x, y in zip(subset['metric'], subset['score']):
        plt.text(x, y - 0.003, f'{y:.3f}', ha='center', va='top', fontsize=8, color='gray')

plt.ylim(0.85, 1.00)
plt.title("Model Performance Comparison", fontsize=16)
plt.xlabel("Performance Metric", fontsize=12)
plt.ylabel("Score", fontsize=12)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.legend(title='Model', fontsize=11, title_fontsize=12, loc='lower right')
plt.tight_layout()
plt.show()

# XGBoost를 이용한 컬럼 특성 중요도 분석 및 시각화
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)
importances = xgb_model.feature_importances_
importance_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
print("특성 중요도 분석 결과:")
print(importance_df)

# 전체 특성 중요도 시각화
importance_df_top = importance_df.sort_values(by='Importance', ascending=True)
plt.figure(figsize=(10, 8))
palette = sns.color_palette("Blues_d", len(importance_df_top))
barplot = sns.barplot(
    x='Importance',
    y='Feature',
    data=importance_df_top,
    palette=palette
)
for container in barplot.containers:
    barplot.bar_label(container, fmt='%.3f', label_type='edge', padding=5, fontsize=10, color='black')
plt.title("Feature Importance", fontsize=12)
plt.xlabel("Importance", fontsize=12)
plt.ylabel("Feature", fontsize=12)
plt.tight_layout()
plt.show()

# 상위 특성 선택
top_features = importance_df['Feature'].head(5).tolist()
print("상위 특성:", top_features)

# 학습 데이터에서 상위 특성 선택 후 군집화
X_train_top = X_train[top_features].copy()
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_train_top)
X_train_top['Cluster'] = clusters
X_train_top['Heart_Risk'] = y_train.values

# 클러스터 라벨 매핑
cluster_risk = X_train_top.groupby('Cluster')['Heart_Risk'].mean().sort_values()
risk_order = cluster_risk.index.tolist()

cluster_names = {risk_order[0]: 'Low Risk Cluster',
                 risk_order[1]: 'Moderate Risk Cluster',
                 risk_order[2]: 'High Risk Cluster'}

X_train_top['Cluster_Label'] = X_train_top['Cluster'].map(cluster_names)

# 각 클러스터별 Heart_Risk 분포 확인
plt.figure(figsize=(8, 6))
sns.set(style="whitegrid")

sns.barplot(
    x='Cluster_Label',
    y='Heart_Risk',
    data=X_train_top,
    palette=['#d62728', '#2ca02c', '#1f77b4'],  # 빨강, 초록, 파랑
    ci=None  # 신뢰 구간 제거
)

plt.title("Distribution of Heart_Risk by Cluster", fontsize=14)
plt.xlabel("Cluster", fontsize=12)
plt.ylabel("Heart_Risk", fontsize=12)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.ylim(0, 1.0)  # Heart_Risk가 0~1 사이이므로 고정
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# 각 클러스터별 Heart_Risk 클래스 비율 확인
plt.figure(figsize=(8, 6))
sns.countplot(x='Cluster_Label', hue='Heart_Risk', data=X_train_top, palette='Set2', edgecolor='black')
plt.title("Heart_Risk Class Counts by Cluster", fontsize=14)
plt.xlabel("Cluster", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.legend(title="Heart Risk", fontsize=10, title_fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# 군집별 평균 Heart_Risk 계산 및 시각화
cluster_risk_named = X_train_top.groupby('Cluster_Label')['Heart_Risk'].mean().sort_values()

plt.figure(figsize=(10, 5))
colors = ['#e76f51', '#f4a261', '#d62828']

bars = plt.barh(
    cluster_risk_named.index, 
    cluster_risk_named.values, 
    color=colors
)

for bar in bars:
    plt.text(
        bar.get_width() + 0.015,
        bar.get_y() + bar.get_height()/2,
        f"{bar.get_width():.2f}", 
        va='center', 
        fontsize=11, 
        color='black'
    )

plt.title("Cluster-wise Heart Risk Ratio", fontsize=14)
plt.xlabel("Heart Risk Rate", fontsize=12)
plt.ylabel("Cluster", fontsize=12)
plt.xlim(0, 1.0)
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# PCA 시각화를 통한 클러스터 분포 전체 확인
pca = PCA(n_components=2)
pca_components = pca.fit_transform(X_train_top[top_features])

pca_df = pd.DataFrame(pca_components, columns=['PC1', 'PC2'])
pca_df['Cluster_Label'] = X_train_top['Cluster_Label']
pca_df['Heart_Risk'] = X_train_top['Heart_Risk']

plt.figure(figsize=(8,6))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Cluster_Label', style='Heart_Risk', palette='Set2')
plt.title("Cluster Separation via PCA", fontsize=14)
plt.xlabel("Principal Component 1", fontsize=12)
plt.ylabel("Principal Component 2", fontsize=12)
plt.legend(title="Cluster", fontsize=10)
plt.tight_layout()
plt.show()

# 상위 2개의 특성을 사용한 경계값 및 위험군 확인
feature_1, feature_2 = top_features[:2]
X_selected = X_train_top[[feature_1, feature_2]].copy()

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_selected)

x_min, x_max = X_selected[feature_1].min() - 1, X_selected[feature_1].max() + 1
y_min, y_max = X_selected[feature_2].min() - 1, X_selected[feature_2].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(x.shape)

colors = ['#f4a261', '#2a9d8f', '#e76f51']
cmap_background = ListedColormap(colors)

plt.figure(figsize=(8,6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_background)
sns.scatterplot(x=feature_1, y=feature_2, hue=X_train_top['Cluster_Label'], data=X_selected, palette=colors)
plt.title("Decision Region Plot", fontsize=14)
plt.xlabel(feature_1)
plt.ylabel(feature_2)
plt.legend(title="Cluster")
plt.tight_layout()
plt.show()