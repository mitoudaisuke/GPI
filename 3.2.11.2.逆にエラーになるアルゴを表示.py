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
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.utils.discovery import all_estimators
import warnings
warnings.filterwarnings('ignore')

# グローバル設定
MAX_ITER = 500  # イテレーション回数の初期値

# データの前処理関数
def scale_data(dataframe):
    original_shape = dataframe.values.shape
    data_1d = dataframe.values.ravel()  # 1次元に変換
    
    # StandardScalerを適用
    std_scaler = StandardScaler()
    std_scaled_1d = std_scaler.fit_transform(data_1d.reshape(-1, 1))
    
    # MinMaxScalerを適用
    minmax_scaler = MinMaxScaler()
    final_scaled_1d = minmax_scaler.fit_transform(std_scaled_1d)
    
    # 元の形状に戻す
    final_scaled_data = final_scaled_1d.reshape(original_shape)
    scaled_df = pd.DataFrame(final_scaled_data,
                           index=dataframe.index,
                           columns=dataframe.columns)
    return scaled_df

# パイプライン作成関数
def create_pipeline(n_components=15):
    return Pipeline([
        ('pca', PCA(n_components=n_components)),
        ('classifier', None)
    ])

# データの読み込み
df = pd.read_csv(import_path, header=0, index_col=0, encoding="sjis")

# Excitation = 280nmのデータのみを選択
df_280 = df[df['wavelength'] == 280].copy()

# 不要なカラムを削除
columns_to_drop = ["turbidity", "wavelength"]
X_df = df_280.drop(['species'] + columns_to_drop, axis=1, errors='ignore')
y = df_280['species']

print("Dataset shape:", X_df.shape)
print("Number of classes:", len(y.unique()))
print("Classes:", y.unique())

# スケーリングを実行
X_scaled = scale_data(X_df)

# 全ての分類器を取得
classifiers = dict(all_estimators(type_filter='classifier'))

# 除外リスト
exclude_classifiers = [
    'ClassifierChain',
    'MultiOutputClassifier',
    'OneVsOneClassifier',
    'OneVsRestClassifier',
    'OutputCodeClassifier',
    'StackingClassifier',
    'VotingClassifier',
    'GradientBoostingClassifier',
    'FixedThresholdClassifier',
    'TunedThresholdClassifierCV',
    'CategoricalNB',
    'ComplementNB',
    'MultinomialNB'
]

# iterationが必要な分類器のリスト
iter_classifiers = [
    'MLPClassifier',
    'LogisticRegression',
    'LogisticRegressionCV',
    'RidgeClassifier',
    'SGDClassifier',
    'Perceptron',
    'PassiveAggressiveClassifier'
]

# 特別な初期化が必要な分類器の設定
special_init_params = {
    'RadiusNeighborsClassifier': {
        'radius': 100.0,
        'outlier_label': 'most_frequent',
        'weights': 'uniform'
    }
}

# イテレーション回数が必要な分類器には自動的にmax_iterを追加
for clf in iter_classifiers:
    special_init_params[clf] = {'max_iter': MAX_ITER}

# 除外リストの分類器を削除
for exc in exclude_classifiers:
    classifiers.pop(exc, None)

# クロスバリデーション設定（1回のみ）
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=42)

# 結果格納用
failed_classifiers = []

# 各分類器でテスト
for name, ClassifierClass in classifiers.items():
    print(f"\nTesting {name}...")
    
    try:
        # 分類器のインスタンス化
        if name in special_init_params:
            classifier = ClassifierClass(**special_init_params[name])
        else:
            classifier = ClassifierClass()
        
        # パイプラインの作成
        pipeline = create_pipeline()
        pipeline.steps[-1] = ('classifier', classifier)
        
        # クロスバリデーションスコアの計算（試行1回のみ）
        scores = cross_val_score(pipeline, X_scaled, y, scoring='accuracy', cv=cv, error_score='raise')
        
        # 結果を出力
        mean_accuracy = np.mean(scores)
        print(f"{name}: Success (Accuracy: {mean_accuracy:.4f})")
        
    except Exception as e:
        print(f"{name}: Failed (Error: {str(e)})")
        failed_classifiers.append({
            'Classifier': name,
            'Error': str(e)
        })
        continue

# エラーになった分類器の結果を出力
if len(failed_classifiers) > 0:
    print("\n=== Failed Classifiers ===")
    failed_df = pd.DataFrame(failed_classifiers)
    print(failed_df)
    failed_df.to_csv(export_path + "failed_classifiers.csv", index=False)
else:
    print("\nNo failed classifiers!")