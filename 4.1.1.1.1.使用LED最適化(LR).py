# Google Drive 認証とパス設定
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
    # drive.mount('/content/drive')  # 必要に応じてコメントアウトを外してください
    import_path = '/content/drive/MyDrive/Colab Notebooks/import_data/EEM.csv'
    export_path = '/content/drive/MyDrive/Colab Notebooks/export_data/'
except:
    import_path = '/Users/mito/Library/Mobile Documents/com~apple~CloudDocs/python/EEM.csv'
    export_path = '/Users/mito/Desktop/'

import gc
import numpy as np
import pandas as pd
import csv
import os
import optuna
from itertools import combinations
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import cross_val_score
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=UserWarning)  # UserWarningを無視

# サンプルデータの読み込み
df = pd.read_csv(import_path)

# 保存先CSVファイル名（実行日時を含める）
timestamp = datetime.now().strftime("%Y%m")
output_csv_path = export_path + f"LED_optimization_results_{timestamp}.csv"

# サンプルを識別するキー（speciesとturbidityでグループ化）
df["sample_id"] = df["species"].astype(str) + "_" + df["turbidity"].astype(str)

LED_list = [
    [214, 236], [226, 254], [244, 266], [250, 270], [252, 258], [256, 276], [256, 278],
    [264, 294], [266, 292], [264, 286], [260, 290], [268, 290], [294, 330], [298, 318],
    [310, 350], [320, 366], [336, 368], [360, 384], [360, 412], [370, 404], [370, 396],
    [374, 398], [380, 414], [382, 408], [390, 424], [390, 430], [400, 434], [404, 442],
    [410, 450], [420, 470], [440, 484], [444, 504], [490, 570], [544, 574], [556, 590], [570, 604]
]

# 初期化: ヘッダーがない場合のみ書き込み
if not os.path.exists(output_csv_path):
    with open(output_csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["n_combinations", "trial", "ranges", "score"])  # ヘッダー


def generate_combined_spectrum(df, sample_id, start_excitation, end_excitation):
    """
    積算後のスペクトルを生成し、条件に基づいて修正
    """
    sample_data = df[df["sample_id"] == sample_id]
    selected_data = sample_data[
        (sample_data["wavelength"] >= start_excitation) &
        (sample_data["wavelength"] <= end_excitation)
    ]

    fluorescence_columns = [col for col in df.columns[:-3] if col.replace('.', '', 1).isdigit()]
    wavelengths = np.array(list(map(float, fluorescence_columns)))

    fluorescence_data = selected_data.loc[:, fluorescence_columns]
    combined_spectrum = fluorescence_data.sum(axis=0)
    threshold_wavelength = end_excitation + 30
    combined_spectrum[wavelengths < threshold_wavelength] = 0

    return combined_spectrum


def prepare_dataset(df, ranges):
    """
    全サンプルに対して指定された範囲でスペクトラムを合成し、学習用データセットを構築
    """
    combined_spectra = []
    labels = []
    for sample_id in df["sample_id"].unique():
        spectra = []
        for start_excitation, end_excitation in ranges:
            combined_spectrum = generate_combined_spectrum(df, sample_id, start_excitation, end_excitation)
            spectra.extend(combined_spectrum)  # データをフラットに結合
        combined_spectra.append(spectra)
        labels.append(df[df["sample_id"] == sample_id]["species"].iloc[0])

    combined_spectra_df = pd.DataFrame(combined_spectra)
    combined_spectra_df["species"] = labels
    return combined_spectra_df


def run_optimization_with_cv(n_combinations):
    """
    10-foldクロスバリデーションを用いたOptuna最適化
    """
    def objective(trial):
        """
        Optunaで最適化する目的関数
        """
        # n_combinations個の範囲を選択
        all_combinations = list(combinations(range(len(LED_list)), n_combinations))  # 組み合わせ生成

        # 組み合わせをインデックス化
        combination_map = {i: comb for i, comb in enumerate(all_combinations)}
        selected_index = trial.suggest_categorical("indices", list(combination_map.keys()))  # インデックスを渡す

        # 選択されたインデックスから範囲を取得
        selected_indices = combination_map[selected_index]
        ranges = [LED_list[i] for i in selected_indices]

        # データセットの準備
        dataset = prepare_dataset(df, ranges)

        X = dataset.drop(columns=["species"])
        y = dataset["species"]

        # データの標準化と正規化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        normalizer = MinMaxScaler()
        X_normalized = normalizer.fit_transform(X_scaled)

        # PCAによる次元削減（次元数は20）
        pca = PCA(n_components=20, random_state=42)
        X_pca = pca.fit_transform(X_normalized)

        # LogisticRegressionCVの学習と評価（10-foldクロスバリデーション）
        model = LogisticRegressionCV(Cs=5, cv=3, solver='lbfgs', max_iter=500, random_state=42)
        scores = cross_val_score(model, X_pca, y, cv=10, scoring="accuracy")
        trial_score = np.mean(scores)

        # 結果をCSVに追記
        with open(output_csv_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([n_combinations, trial.number, ranges, trial_score])

        # キャッシュクリア
        gc.collect()
        return trial_score

    # Optunaスタディの作成
    study_name = f"optimization_{n_combinations}"
    storage_name = "sqlite:///optimization_.db"
    study = optuna.create_study(direction="maximize", study_name=study_name, storage=storage_name, load_if_exists=True)

    # Optunaによる最適化
    study.optimize(objective, n_trials=100, timeout=None)  # 各組み合わせで100回最適化


# 組み合わせ数を増やしながら最適化
for n in range(9, 11):  # 組み合わせ数を8から10まで
    print(f"\n=== Optimizing for {n} combination(s) ===")
    run_optimization_with_cv(n)

# 保存したCSVを読み込んで結果をグラフ化
results_df = pd.read_csv(output_csv_path)

# グラフ描画
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
for n_combinations, group_df in results_df.groupby("n_combinations"):
    plt.plot(group_df["trial"], group_df["score"], label=f"{n_combinations} combinations")
plt.title("Optimization Results by Combination Count")
plt.xlabel("Trial Number")
plt.ylabel("CV Accuracy")
plt.legend(title="Number of Combinations")
plt.grid()
plt.show()