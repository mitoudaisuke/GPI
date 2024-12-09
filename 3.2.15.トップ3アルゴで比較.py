# 環境に応じたパス設定
try:
    import json
    import requests
    from pydrive2.auth import GoogleAuth
    from pydrive2.drive import GoogleDrive
    from oauth2client.service_account import ServiceAccountCredentials
    scope = ['https://www.googleapis.com/auth/drive']
    json_file = 'https://script.google.com/macros/s/AKfycbzk2Uu4poFplvsBNQMyKTJ9uhkmyZFTbnngg7SOsesWGtrC1-e3NQDsL27Giib-k9QAUQ/exec'
    key = json.loads(requests.get(json_file).text)
    credentials = ServiceAccountCredentials.from_json_keyfile_dict(key, scope)
    gauth = GoogleAuth()
    gauth.credentials = credentials
    drive = GoogleDrive(gauth)
    import sys
    sys.path.append('/content/drive/MyDrive/Colab Notebooks/modules')
    from google.colab import drive
    # drive.mount('/content/drive')
    import_path='/content/drive/MyDrive/Colab Notebooks/import_data/EEM.csv'
    export_path ='/content/drive/MyDrive/Colab Notebooks/export_data/'
    directory='/content/drive/MyDrive/Colab Notebooks/import_data/MCS/'
except:
    import_path='/Users/mito/Library/Mobile Documents/com~apple~CloudDocs/python/EEM.csv'
    export_path ='/Users/mito/Desktop/'
    directory='/Users/mito/Library/Mobile Documents/com~apple~CloudDocs/python/手動MCS/'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.utils.discovery import all_estimators
from sklearn.model_selection import RepeatedKFold

import warnings
warnings.filterwarnings('ignore')

# グローバル設定
MAX_ITER = 500
RANDOM_STATE = 42
N_OUTER_SPLITS = 3  # 外側の分割を3回
N_INNER_SPLITS = 3  # 内側の分割を10回
N_INNER_REPEATS = 3  # 内側の繰り返しを10回

# 選択する分類器の定義を修正
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegressionCV

selected_classifiers = {
    'ExtraTreesClassifier': (ExtraTreesClassifier, {}),
    'MLPClassifier': (MLPClassifier, {'max_iter': MAX_ITER}),
    'LogisticRegressionCV': (LogisticRegressionCV, {'max_iter': MAX_ITER})
}

# スケーリングクラス
class DataScaler:
    def __init__(self):
        self.std_scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()
        
    def fit_transform(self, data):
        """訓練データ用：統計量の計算と変換を行う"""
        original_shape = data.values.shape
        data_1d = data.values.ravel()
        
        # StandardScalerをfitして変換
        std_scaled_1d = self.std_scaler.fit_transform(data_1d.reshape(-1, 1))
        
        # MinMaxScalerをfitして変換
        final_scaled_1d = self.minmax_scaler.fit_transform(std_scaled_1d)
        
        # 元の形状に戻す
        final_scaled_data = final_scaled_1d.reshape(original_shape)
        return pd.DataFrame(final_scaled_data,
                          index=data.index,
                          columns=data.columns)
    
    def transform(self, data):
        """検証・テストデータ用：既存の統計量で変換のみを行う"""
        original_shape = data.values.shape
        data_1d = data.values.ravel()
        
        # StandardScalerで変換（既存の統計量を使用）
        std_scaled_1d = self.std_scaler.transform(data_1d.reshape(-1, 1))
        
        # MinMaxScalerで変換（既存の統計量を使用）
        final_scaled_1d = self.minmax_scaler.transform(std_scaled_1d)
        
        # 元の形状に戻す
        final_scaled_data = final_scaled_1d.reshape(original_shape)
        return pd.DataFrame(final_scaled_data,
                          index=data.index,
                          columns=data.columns)

# パイプライン作成関数
def create_pipeline(n_components=15):
    return Pipeline([
        ('pca', PCA(n_components=n_components)),
        ('classifier', None)
    ])

# データの読み込み
df = pd.read_csv(import_path, header=0, index_col=0, encoding="sjis")
df_280 = df[df['wavelength'] == 280].copy()
columns_to_drop = ["turbidity", "wavelength"]
X_df = df_280.drop(['species'] + columns_to_drop, axis=1, errors='ignore')
y = df_280['species']

print("Dataset shape:", X_df.shape)
print("Number of classes:", len(y.unique()))
print("Classes:", y.unique())

# メイン処理
def evaluate_classifiers(X, y, selected_classifiers):
    results = []
    
    # 外側のループ（3回の独立した評価）
    for outer_split in range(N_OUTER_SPLITS):
        print(f"\nOuter Split {outer_split + 1}/{N_OUTER_SPLITS}")
        
        # 訓練用+検証用とテスト用にデータを分割 (85:15)
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=0.15, random_state=RANDOM_STATE + outer_split, stratify=y)
        
        # 内側のクロスバリデーション設定
        inner_cv = RepeatedKFold(n_splits=N_INNER_SPLITS, 
                                n_repeats=N_INNER_REPEATS, 
                                random_state=RANDOM_STATE)
        
        # 各分類器の評価
        for name, (ClassifierClass, params) in selected_classifiers.items():
            print(f"Testing {name}...")
            
            try:
                # 分類器のインスタンス化
                if params is not None:
                    classifier = ClassifierClass(**params)
                else:
                    classifier = ClassifierClass()
                
                # クロスバリデーション
                for fold_idx, (train_idx, val_idx) in enumerate(inner_cv.split(X_train_val)):
                    # データの分割
                    X_train = X_train_val.iloc[train_idx]
                    X_val = X_train_val.iloc[val_idx]
                    y_train = y_train_val.iloc[train_idx]
                    y_val = y_train_val.iloc[val_idx]
                    
                    # スケーラーの初期化と適用
                    scaler = DataScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_val_scaled = scaler.transform(X_val)
                    
                    # モデルの学習と評価
                    pipeline = create_pipeline()
                    pipeline.steps[-1] = ('classifier', classifier)
                    pipeline.fit(X_train_scaled, y_train)
                    val_score = pipeline.score(X_val_scaled, y_val)
                    
                    # 結果の保存（バリデーション）
                    results.append({
                        'Classifier': name,
                        'Score Type': 'Validation',
                        'Accuracy': val_score,
                        'Outer Split': outer_split + 1
                    })
                
                # テストデータでの評価
                final_scaler = DataScaler()
                X_train_val_scaled = final_scaler.fit_transform(X_train_val)
                X_test_scaled = final_scaler.transform(X_test)
                
                final_pipeline = create_pipeline()
                final_pipeline.steps[-1] = ('classifier', classifier)
                final_pipeline.fit(X_train_val_scaled, y_train_val)
                test_score = final_pipeline.score(X_test_scaled, y_test)
                
                # 結果の保存（テスト）
                results.append({
                    'Classifier': name,
                    'Score Type': 'Test',
                    'Accuracy': test_score,
                    'Outer Split': outer_split + 1
                })
                
            except Exception as e:
                print(f"Error with {name}: {str(e)}")
                continue
    
    return pd.DataFrame(results)

# メインの実行部分
if __name__ == "__main__":
    # データの読み込みと前処理
    df = pd.read_csv(import_path, header=0, index_col=0, encoding="sjis")
    df_280 = df[df['wavelength'] == 280].copy()
    columns_to_drop = ["turbidity", "wavelength"]
    X_df = df_280.drop(['species'] + columns_to_drop, axis=1, errors='ignore')
    y = df_280['species']

    print("Dataset shape:", X_df.shape)
    print("Number of classes:", len(y.unique()))
    print("Classes:", y.unique())

    # 選択する分類器を定義
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.linear_model import LogisticRegressionCV

    selected_classifiers = {
        'ExtraTreesClassifier': (ExtraTreesClassifier, None),
        'MLPClassifier': (MLPClassifier, {'max_iter': MAX_ITER}),
        'LogisticRegressionCV': (LogisticRegressionCV, {'max_iter': MAX_ITER})
    }

    # 評価の実行
    results_df = evaluate_classifiers(X_df, y, selected_classifiers)

    # 結果の表示と可視化
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=results_df, x='Classifier', y='Accuracy', hue='Score Type',
                palette={'Validation': 'lightblue', 'Test': 'lightgreen'})
    
    # テストデータのポイントを重ねてプロット
    test_data = results_df[results_df['Score Type'] == 'Test']
    sns.swarmplot(data=test_data, x='Classifier', y='Accuracy', color='red', size=8)

    plt.title('Model Performance Comparison\n(300 CV splits and 3 Test sets)', 
             fontsize=14, fontweight='bold')
    plt.xlabel('Classifier', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.legend(title='Score Type')

    # 詳細な結果の表示
    print("\n=== Performance Summary ===")
    summary = results_df.groupby(['Classifier', 'Score Type'])[['Accuracy']].agg(['mean', 'std', 'min', 'max']).round(4)
    print(summary)

    # テストスコアの詳細表示
    print("\n=== Test Scores by Split ===")
    test_summary = results_df[results_df['Score Type'] == 'Test'].groupby(['Classifier', 'Outer Split'])['Accuracy'].mean().round(4)
    print(test_summary)

    # 結果の保存
    results_df.to_csv(export_path + "model_comparison_with_multiple_tests.csv")
    plt.tight_layout()
    plt.show()