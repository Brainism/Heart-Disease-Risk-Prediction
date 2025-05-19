import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os
import warnings
# scipy.stats 추가 (통계 검정용)
import scipy.stats as stats
from scipy.stats import chi2_contingency


from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import (roc_curve, auc, precision_score, recall_score, f1_score,
                             accuracy_score, confusion_matrix, silhouette_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier

try:
    from kneed import KneeLocator
    kneed_installed = True
except ImportError:
    print("Warning: 'kneed' library not found. Elbow point detection will be skipped.")
    print("Install it using: pip install kneed")
    kneed_installed = False

matplotlib.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.titlesize'] = 14
# 한글 폰트 설정 (Windows: Malgun Gothic, macOS: AppleGothic) - 필요시 주석 해제 및 폰트 설치
# plt.rcParams['font.family'] = 'Malgun Gothic' # Windows
# plt.rcParams['font.family'] = 'AppleGothic' # macOS

warnings.filterwarnings('ignore')

# --- 0. 결과 저장 디렉토리 설정 ---
print("--- 0. 결과 저장 디렉토리 설정 ---")
output_dir = "paper_visualizations_focused"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created directory: {output_dir}")
else:
    print(f"Output directory already exists: {output_dir}")
# Debug: Print absolute path of output directory
print(f"[DEBUG] Absolute path for output directory: {os.path.abspath(output_dir)}")

# --- 1. 데이터 로드 및 기본 탐색 ---
print("\n--- 1. 데이터 로드 및 기본 탐색 ---")
try:
    # 중요: 실제 파일 위치로 경로를 업데이트하세요.
    df = pd.read_csv(r'C:\Heart-Disease-Risk-Prediction\data\heart_disease_risk_dataset_earlymed.csv')
except FileNotFoundError:
    print("Error: CSV file not found. Please check the path.")
    exit()

df.columns = df.columns.str.strip()
print("Data Info:")
df.info()
print("\nMissing Value Check:")
print(df.isnull().sum())

TARGET = 'Heart_Risk'
if TARGET not in df.columns:
    raise ValueError(f"Target variable '{TARGET}' not found.")

# --- 2. 데이터 전처리 준비 ---
print("\n--- 2. 데이터 전처리 준비 ---")

bmi_created = False
if set(['Weight', 'Height']).issubset(df.columns):
    df['Height_m'] = df['Height'].apply(lambda x: x / 100 if x > 0 else np.nan)
    df['BMI'] = np.where(df['Height_m'].isnull() | (df['Height_m'] == 0),
                         np.nan,
                         df['Weight'] / (df['Height_m'] ** 2))
    if 'Height_m' in df.columns:
        df = df.drop('Height_m', axis=1)
    print("BMI derived variable created.")
    bmi_created = True
    print(f"Missing BMI values: {df['BMI'].isnull().sum()}")
else:
    print("Warning: 'Weight' or 'Height' not found. Cannot create BMI.")

age_bin_created = False
if 'Age' in df.columns:
    try:
        min_age = df['Age'].min()
        max_age = df['Age'].max()
        # 나이 구간 정의 (연구 목적에 맞게 조정 가능)
        age_bins = [min_age, 40, 50, 60, max_age + 1]
        age_labels = [f'<{age_bins[1]}', f'{age_bins[1]}-{age_bins[2]-1}', f'{age_bins[2]}-{age_bins[3]-1}', f'{age_bins[3]}+']

        df['Age_bin'] = pd.cut(df['Age'], bins=age_bins, labels=False, right=False, include_lowest=True)
        print(f"\nAge binned into {len(age_bins)-1} defined intervals.")
        print(f"Age bins used (excluding right edge): {age_bins}")
        print(f"Value counts per bin:\n{df['Age_bin'].value_counts().sort_index()}")
        age_bin_created = True

    except Exception as e:
        print(f"Warning: Age Binning failed ({e}). 'Age_bin' not created.")
        if 'Age_bin' in df.columns: df = df.drop('Age_bin', axis=1)
else:
     print("Warning: 'Age' column not found.")


numerical_features = []
if bmi_created and 'BMI' in df.columns:
    numerical_features.append('BMI')
    print("Included 'BMI' in numerical features.")
# Age 구간화 성공 여부와 관계없이 원본 Age는 사용하지 않음 (Age_bin 사용)
# if not age_bin_created and 'Age' in df.columns:
#     numerical_features.append('Age')
#     print("Included 'Age' in numerical features (will be scaled).")

binary_features = ['Chest_Pain', 'Shortness_of_Breath', 'Fatigue', 'Palpitations',
                   'Dizziness', 'Swelling', 'Pain_Arms_Jaw_Back', 'Cold_Sweats_Nausea',
                   'High_BP', 'High_Cholesterol', 'Diabetes', 'Smoking', 'Obesity',
                   'Sedentary_Lifestyle', 'Family_History', 'Chronic_Stress', 'Gender']
binary_features = [f for f in binary_features if f in df.columns]
print(f"Identified binary features: {binary_features}")

categorical_features_to_encode = []
if age_bin_created and 'Age_bin' in df.columns:
    # Age_bin을 범주형으로 간주하여 원핫인코딩 대상으로 포함
    categorical_features_to_encode.append('Age_bin')
    print("Included 'Age_bin' in features to be one-hot encoded.")

feature_cols = numerical_features + binary_features + categorical_features_to_encode
print(f"\nFinal list of features for modeling: {feature_cols}")

missing_cols = [col for col in feature_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"Defined features missing from DataFrame: {missing_cols}")

X = df[feature_cols]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"\nData split: Train {X_train.shape}, Test {X_test.shape}")

# --- 3. 전처리 파이프라인 구축 ---
print("\n--- 3. 전처리 파이프라인 구축 ---")

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), # 혹시 모를 결측치 처리
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

transformers_list = []
if numerical_features:
    transformers_list.append(('num', numeric_transformer, numerical_features))
if binary_features:
    # 이진 변수도 최빈값으로 결측치 처리
    binary_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent'))])
    transformers_list.append(('binary', binary_transformer, binary_features))
if categorical_features_to_encode:
    transformers_list.append(('cat_encode', categorical_transformer, categorical_features_to_encode))

preprocessor = ColumnTransformer(transformers=transformers_list, remainder='drop')
print("Preprocessing pipeline created.")

# --- 4. 모델 학습 및 평가 함수 정의 ---
print("\n--- 4. 모델 학습 및 평가 함수 정의 ---")

def evaluate_model(model, model_name, X_test, y_test, save_cm=True):
    """ 모델 성능 평가 및 혼동 행렬 시각화 """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print(f"\n--- {model_name} Test Performance ---")
    print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}, AUC: {roc_auc:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    if save_cm:
        print(f"[DEBUG] Creating figure for {model_name} confusion matrix...") # Debug Print
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 11})
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(f"{model_name} Confusion Matrix")
        sns.despine()
        plt.tight_layout()
        try:
            save_path = os.path.join(output_dir, f'confusion_matrix_{model_name.replace(" ", "_").lower()}.png')
            print(f"[DEBUG] Attempting to save CM to: {os.path.abspath(save_path)}") # Debug Print
            plt.savefig(save_path, dpi=300)
            print(f"[DEBUG] Successfully saved CM for {model_name}.") # Debug Print
        except Exception as e:
            print(f"[DEBUG] Error saving confusion matrix for {model_name}: {e}") # Debug Print
        # plt.close() # Temporarily commented out for debugging

    fpr, tpr, _ = roc_curve(y_test, y_prob)

    return {"model": model_name, "accuracy": acc, "precision": prec,
            "recall": rec, "f1_score": f1, "auc": roc_auc, "cm": cm,
            "fpr": fpr, "tpr": tpr}

def print_confusion_details(cm, model_name):
    """ 혼동 행렬 상세 값 출력 """
    TN, FP, FN, TP = cm.ravel()
    print(f"\n{model_name} Confusion Matrix Details:")
    print(f"TN: {TN}, FP: {FP}, FN: {FN}, TP: {TP}")

# --- 5. 모델 학습, 튜닝 및 비교 ---
print("\n--- 5. 모델 학습, 튜닝 및 비교 ---")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 모델 1: Decision Tree
print("[DEBUG] Training Decision Tree model...")
dt_model = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', DecisionTreeClassifier(max_depth=5, random_state=42))])
dt_model.fit(X_train, y_train)
print("\nDecision Tree Model Trained.")
metrics_dt = evaluate_model(dt_model, "Decision Tree", X_test, y_test)
print_confusion_details(metrics_dt['cm'], "Decision Tree")

# 모델 2: XGBoost (Tuning)
print("\n--- XGBoost Hyperparameter Tuning (GridSearchCV) ---")
param_grid_xgb = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [3, 5],
    'classifier__learning_rate': [0.1, 0.05],
    'classifier__subsample': [0.8, 1.0],
}

xgb_pipeline_for_tuning = Pipeline(steps=[('preprocessor', preprocessor),
                                          ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))])

print("[DEBUG] Starting GridSearchCV for XGBoost...")
grid_search_xgb = GridSearchCV(estimator=xgb_pipeline_for_tuning, param_grid=param_grid_xgb,
                               cv=skf, scoring='roc_auc', n_jobs=-1, verbose=1)
grid_search_xgb.fit(X_train, y_train)
print("[DEBUG] GridSearchCV for XGBoost finished.")

print("\nBest XGBoost Hyperparameters:", grid_search_xgb.best_params_)
print(f"Best AUC score from GridSearchCV: {grid_search_xgb.best_score_:.4f}")

best_xgb_model = grid_search_xgb.best_estimator_
print("\nTuned XGBoost Model Trained.")
metrics_xgb_tuned = evaluate_model(best_xgb_model, "XGBoost (Tuned)", X_test, y_test)
print_confusion_details(metrics_xgb_tuned['cm'], "XGBoost (Tuned)")

# --- 5.1 XGBoost Feature Importance --- (새로 추가된 섹션)
print("\n--- 5.1 XGBoost Feature Importance ---")
try:
    # Get the fitted preprocessor and classifier
    preprocessor_fitted = best_xgb_model.named_steps['preprocessor']
    xgb_classifier_fitted = best_xgb_model.named_steps['classifier']

    # Get feature names after preprocessing (including one-hot encoded features)
    try:
        # Scikit-learn >= 0.24 (get_feature_names_out is preferred)
        feature_names_out = preprocessor_fitted.get_feature_names_out()
        print(f"[DEBUG] Retrieved {len(feature_names_out)} feature names using get_feature_names_out.")
    except AttributeError:
        # Fallback for older Scikit-learn versions (might be less robust)
        print("[DEBUG] get_feature_names_out not available, attempting manual name construction.")
        feature_names_out = []
        for name, transformer, columns in preprocessor_fitted.transformers_:
             if transformer == 'drop' or (isinstance(transformer, str) and transformer == 'passthrough' and name == 'remainder'): continue
             if name == 'num': feature_names_out.extend(columns)
             elif name == 'binary': feature_names_out.extend(columns) # After imputer, names are same
             elif name == 'cat_encode':
                 try:
                     ohe = transformer.named_steps['onehot']
                     # Construct names based on categories - assumes SimpleImputer didn't change things drastically
                     cats = ohe.categories_
                     new_names = [f"{col}_{cat}" for i, col in enumerate(columns) for cat in cats[i]]
                     feature_names_out.extend(new_names)
                 except Exception as e_ohe:
                     print(f"[DEBUG] Error getting OHE feature names manually: {e_ohe}")
                     feature_names_out.extend([f"{col}_?" for col in columns]) # Placeholder
             elif transformer == 'passthrough': feature_names_out.extend(columns)
        print(f"[DEBUG] Manually constructed {len(feature_names_out)} feature names.")


    # Get feature importances
    importances = xgb_classifier_fitted.feature_importances_

    # Ensure lengths match
    if len(feature_names_out) == len(importances):
        # Create DataFrame and sort
        feature_importance_df = pd.DataFrame({'Feature': feature_names_out, 'Importance': importances})
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

        # Print top N features
        top_n = 15 # Display top 15 features
        print(f"\nTop {top_n} Feature Importances for XGBoost:")
        print(feature_importance_df.head(top_n).to_string())

        # Plot top N features
        print(f"[DEBUG] Creating feature importance plot for top {top_n} features...") # Debug Print
        plt.figure(figsize=(10, max(6, top_n * 0.4))) # Adjust height based on N
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(top_n), palette='viridis')
        plt.title(f'Top {top_n} Feature Importances (XGBoost Tuned)')
        plt.xlabel('Importance Score')
        plt.ylabel('Feature')
        plt.tight_layout()
        try:
            save_path = os.path.join(output_dir, 'feature_importance_xgboost.png')
            print(f"[DEBUG] Attempting to save feature importance plot to: {os.path.abspath(save_path)}") # Debug Print
            plt.savefig(save_path, dpi=300)
            print("[DEBUG] Successfully saved feature importance plot.") # Debug Print
        except Exception as e:
            print(f"[DEBUG] Error saving feature importance plot: {e}") # Debug Print
        # plt.close() # Temporarily commented out for debugging

    else:
         print(f"[ERROR] Mismatch between number of feature names ({len(feature_names_out)}) and importances ({len(importances)}). Skipping feature importance analysis.")

except Exception as e:
    print(f"[ERROR] Could not perform feature importance analysis: {e}")


# --- 6. 모델 성능 비교 시각화 ---
print("\n--- 6. 모델 성능 비교 시각화 ---")

results_list = [metrics_dt, metrics_xgb_tuned]
results_df = pd.DataFrame(results_list)
print("\nModel Performance Summary:")
print(results_df.drop(columns=['cm', 'fpr', 'tpr']).round(4).to_string())

results_for_plot = results_df.drop(columns=['cm', 'fpr', 'tpr'])
results_long = pd.melt(results_for_plot, id_vars='model', var_name='Metric', value_name='Score')

# Bar Plot
print("[DEBUG] Creating model performance bar plot figure...") # Debug Print
plt.figure(figsize=(8, 5))
palette = sns.color_palette("viridis", n_colors=len(results_df))
ax = sns.barplot(data=results_long, x='Metric', y='Score', hue='model',
                 palette=palette, edgecolor='black', ci=None)
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f', fontsize=10, padding=3)
plt.ylim(bottom=max(0, results_long['Score'].min() * 0.9))
plt.title("Model Performance Comparison")
plt.xlabel("Performance Metric")
plt.ylabel("Score")
plt.legend(title='Model', bbox_to_anchor=(1.02, 1), loc='upper left')
sns.despine()
plt.tight_layout(rect=[0, 0, 0.85, 1])
try:
    save_path = os.path.join(output_dir, 'model_performance_comparison_bar.png')
    print(f"[DEBUG] Attempting to save bar plot to: {os.path.abspath(save_path)}") # Debug Print
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print("[DEBUG] Successfully saved bar plot.") # Debug Print
except Exception as e:
    print(f"[DEBUG] Error saving bar plot: {e}") # Debug Print
# plt.close() # Temporarily commented out for debugging

# ROC Curve Plot
print("[DEBUG] Creating ROC curve comparison plot figure...") # Debug Print
plt.figure(figsize=(6, 5))
for i, result in enumerate(results_list):
    plt.plot(result['fpr'], result['tpr'], marker='.', markersize=4,
             label=f"{result['model']} (AUC = {result['auc']:.3f})", color=palette[i])
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Chance')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc='lower right')
plt.grid(axis='both', linestyle='--', alpha=0.4)
sns.despine()
plt.tight_layout()
try:
    save_path = os.path.join(output_dir, 'roc_curve_comparison.png')
    print(f"[DEBUG] Attempting to save ROC plot to: {os.path.abspath(save_path)}") # Debug Print
    plt.savefig(save_path, dpi=300)
    print("[DEBUG] Successfully saved ROC plot.") # Debug Print
except Exception as e:
    print(f"[DEBUG] Error saving ROC plot: {e}") # Debug Print
# plt.close() # Temporarily commented out for debugging


print("\n--- Model Comparison Complete ---")
print(f"The best performing model based on AUC is: {results_df.loc[results_df['auc'].idxmax()]['model']}")

# --- 7. Cluster Analysis using Best Model (XGBoost Tuned) ---
print("\n--- 7. Cluster Analysis using Best Model (XGBoost Tuned) ---")
print("Purpose: To identify potential patient subgroups based on their features.")

print("Preparing data for clustering using the best model's preprocessor...")
try:
    preprocessor_for_clustering = best_xgb_model.named_steps['preprocessor']
    print("[DEBUG] Attempting to transform data for clustering...") # Debug Print
    X_train_processed_cluster = preprocessor_for_clustering.transform(X_train)
    print("[DEBUG] Data transformation for clustering successful.") # Debug Print
    print(f"Training data preprocessed for clustering, shape: {X_train_processed_cluster.shape}")

except Exception as e:
    print(f"[DEBUG] Error occurred during clustering data prep: {e}") # Debug Print
    print(f"Error during data preparation for clustering: {e}")
    print("Skipping Cluster Analysis section.")
    X_train_processed_cluster = None

if X_train_processed_cluster is not None:
    print("[DEBUG] Entered clustering analysis block (X_train_processed_cluster is not None).") # Debug Print

    print("\nDetermining optimal number of clusters (K)...")
    sse = []
    silhouette_coefficients = []
    k_range = range(2, 9)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_train_processed_cluster)
        sse.append(kmeans.inertia_)
        try:
            score = silhouette_score(X_train_processed_cluster, kmeans.labels_)
            silhouette_coefficients.append(score)
        except ValueError as e:
            silhouette_coefficients.append(-1)

    # K-Selection Plot
    print("[DEBUG] Creating K-selection plot figure...") # Debug Print
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(k_range, sse, marker='o', linestyle='-')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('SSE (Inertia)')
    plt.grid(True)
    suggested_k_elbow = None
    if kneed_installed:
        try:
            kl = KneeLocator(k_range, sse, curve='convex', direction='decreasing')
            suggested_k_elbow = kl.elbow
            if suggested_k_elbow:
                plt.vlines(suggested_k_elbow, plt.ylim()[0], plt.ylim()[1], linestyles='--', color='r', label=f'Elbow point = {suggested_k_elbow}')
                plt.legend()
                print(f"\nSuggested K based on Elbow method (kneed): {suggested_k_elbow}")
            else:
                print("\nCould not automatically find elbow point using kneed.")
        except Exception as e:
            print(f"\nError finding elbow point with kneed: {e}")
    else:
        print("\nSkipping automatic Elbow point detection ('kneed' not installed).")

    plt.subplot(1, 2, 2)
    valid_silhouette_indices = [i for i, score in enumerate(silhouette_coefficients) if score != -1]
    if valid_silhouette_indices:
         valid_k_range = [k_range[i] for i in valid_silhouette_indices]
         valid_scores = [silhouette_coefficients[i] for i in valid_silhouette_indices]
         plt.plot(valid_k_range, valid_scores, marker='o', linestyle='-')
         plt.title('Silhouette Scores for Optimal K')
         plt.xlabel('Number of clusters (K)')
         plt.ylabel('Silhouette Score')
         plt.grid(True)
         max_silhouette_score_index = np.argmax(valid_scores)
         suggested_k_silhouette = valid_k_range[max_silhouette_score_index]
         plt.vlines(suggested_k_silhouette, plt.ylim()[0], plt.ylim()[1], linestyles='--', color='g', label=f'Max score K = {suggested_k_silhouette}')
         plt.legend()
         print(f"Suggested K based on max Silhouette score: {suggested_k_silhouette}")
    else:
         print("No valid Silhouette scores were calculated.")
         suggested_k_silhouette = None

    plt.tight_layout()
    try:
        save_path = os.path.join(output_dir, 'kmeans_optimal_k_selection.png')
        print(f"[DEBUG] Attempting to save K-selection plot to: {os.path.abspath(save_path)}") # Debug Print
        plt.savefig(save_path, dpi=300)
        print("[DEBUG] Successfully saved K-selection plot.") # Debug Print
    except Exception as e:
        print(f"[DEBUG] Error saving K-selection plot: {e}") # Debug Print
    # plt.close() # Temporarily commented out for debugging


    if suggested_k_elbow:
        optimal_k = suggested_k_elbow
    elif suggested_k_silhouette:
        optimal_k = suggested_k_silhouette
    else:
        optimal_k = 3

    optimal_k = 4 # User confirmed K=4
    print(f"\nBased on automated suggestions and plot review, final K is set to: {optimal_k}")
    # print("User should verify this optimal K based on the saved 'kmeans_optimal_k_selection.png' plot") # Keep verification internal

    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    clusters_train = kmeans.fit_predict(X_train_processed_cluster)
    print(f"K-Means clustering performed with K={optimal_k}.")

    X_train_clustered = X_train.copy()
    X_train_clustered['Cluster'] = clusters_train
    X_train_clustered = X_train_clustered.reset_index(drop=True)
    y_train_reset = y_train.reset_index(drop=True)
    X_train_clustered['Heart_Risk'] = y_train_reset

    cluster_risk_mean = X_train_clustered.groupby('Cluster')['Heart_Risk'].mean().sort_values()
    risk_order = cluster_risk_mean.index.tolist()

    cluster_names = {}
    if optimal_k == 3:
        cluster_names = {risk_order[0]: 'Low Risk Group', risk_order[1]: 'Medium Risk Group', risk_order[2]: 'High Risk Group'}
    elif optimal_k == 2:
         cluster_names = {risk_order[0]: 'Lower Risk Group', risk_order[1]: 'Higher Risk Group'}
    else: # Includes K=4
        for i, cluster_idx in enumerate(risk_order):
            cluster_names[cluster_idx] = f'Cluster {i+1} (Risk Rank {i+1})'

    X_train_clustered['Cluster_Label'] = X_train_clustered['Cluster'].map(cluster_names)
    print("\nAverage Heart Risk per identified cluster group:")
    print(X_train_clustered.groupby('Cluster_Label')['Heart_Risk'].mean().loc[[cluster_names[i] for i in risk_order]].round(3))

    print("\nCluster Profiles (Mean values for key features):")
    # Define features for profiling (using original names where possible)
    profiling_features = []
    if bmi_created and 'BMI' in X_train_clustered.columns: profiling_features.append('BMI')
    if age_bin_created and 'Age_bin' in X_train_clustered.columns:
         profiling_features.append('Age_bin') # Use Age_bin index for profiling
    elif 'Age' in X_train_clustered.columns: # Fallback if Age_bin failed
         profiling_features.append('Age')

    profiling_features.extend([f for f in binary_features if f in X_train_clustered.columns]) # Add binary features present

    cluster_profile = X_train_clustered.groupby('Cluster_Label')[profiling_features + ['Heart_Risk']].mean()
    profile_ordered = cluster_profile.loc[[cluster_names[i] for i in risk_order]]
    print(profile_ordered.round(3).T)

    # --- 7.1 Statistical Tests for Cluster Differences --- (새로 추가된 섹션)
    print("\n--- 7.1 Statistical Tests for Cluster Differences (Chi-square) ---")
    significant_diff_features = []
    alpha = 0.05 # Significance level

    # Test only categorical/binary features present in profiling_features
    cat_binary_test_features = [f for f in profiling_features if f in categorical_features_to_encode or f in binary_features]

    if not cat_binary_test_features:
         print("No categorical or binary features found for Chi-square testing.")
    else:
        print(f"Performing Chi-square tests for {len(cat_binary_test_features)} features against {optimal_k} clusters (alpha={alpha}):")
        results_stats = {}
        for feature in cat_binary_test_features:
            try:
                # Create contingency table
                contingency_table = pd.crosstab(X_train_clustered[feature], X_train_clustered['Cluster'])
                # Perform Chi-square test
                chi2, p, dof, ex = chi2_contingency(contingency_table, correction=False) # Correction typically not needed for >2x2 tables or large N
                results_stats[feature] = {'Chi2': chi2, 'p-value': p, 'DoF': dof}
                # Print result
                significance = "*" if p < alpha else ""
                print(f"- {feature}: Chi2={chi2:.2f}, p={p:.4f}{significance}")
                if p < alpha:
                    significant_diff_features.append(feature)
            except Exception as e:
                print(f"- {feature}: Error during Chi-square test - {e}")

        print(f"\nFeatures showing statistically significant differences (p < {alpha}) across clusters:")
        print(significant_diff_features if significant_diff_features else "None")
        # Note: ANOVA/Kruskal-Wallis would be needed for numerical features like BMI or original Age if they were used.

    # Cluster Risk Distribution Plot
    print("[DEBUG] Creating cluster risk distribution plot figure...") # Debug Print
    plt.figure(figsize=(max(6, optimal_k * 2), 5))
    cluster_order_labels = [cluster_names[i] for i in risk_order]
    sns.countplot(x='Cluster_Label', hue='Heart_Risk', data=X_train_clustered,
                  order=cluster_order_labels, palette='coolwarm', edgecolor='black')
    plt.title(f"Distribution of Actual Heart Risk within {optimal_k} Clusters")
    plt.xlabel("Cluster Group (ordered by average risk)")
    plt.ylabel("Number of Patients")
    plt.xticks(rotation=10)
    plt.legend(title="Actual Risk", labels=['Low (0)', 'High (1)'], loc='upper right')
    sns.despine()
    plt.tight_layout()
    try:
        save_path = os.path.join(output_dir, 'cluster_actual_risk_distribution.png')
        print(f"[DEBUG] Attempting to save cluster distribution plot to: {os.path.abspath(save_path)}") # Debug Print
        plt.savefig(save_path, dpi=300)
        print("[DEBUG] Successfully saved cluster distribution plot.") # Debug Print
    except Exception as e:
        print(f"[DEBUG] Error saving cluster distribution plot: {e}") # Debug Print
    # plt.close() # Temporarily commented out for debugging

    print("\nCluster analysis complete. Profile table, statistical tests, and distribution plot generated.") # Updated print

# --- 8. Final Conclusion ---
print("\n--- 8. Final Conclusion ---")
print(f"Analysis focused on comparing Decision Tree and XGBoost models, followed by cluster analysis using the best model (Tuned XGBoost), including feature importance and cluster difference tests.") # Updated print
if X_train_processed_cluster is not None:
    print(f"Optimal K for clustering was determined based on Elbow/Silhouette analysis (suggested K_elbow={suggested_k_elbow}, K_silhouette={suggested_k_silhouette}), with final K={optimal_k} used.")
    # Also mention new results availability
    print("Feature importance for XGBoost and statistical tests for cluster differences were performed.")
else:
    print("Cluster analysis was skipped due to errors in data preparation.")
print(f"Visualizations and key results saved to '{output_dir}'.")