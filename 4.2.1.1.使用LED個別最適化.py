# Google Drive 認証とパス設定
try:
    import json
    import requests
    from pydrive2.auth import GoogleAuth
    from pydrive2.drive import GoogleDrive
    from oauth2client.service_account import ServiceAccountCredentials
    import sys

    scope = ['https://www.googleapis.com/auth/drive']
    json_file = 'https://script.google.com/macros/s/AKfycbzk2Uu4poFplvsBNQMyKTJ9uhkmyZFTbnngg7SOsesWGtrC1-e3NQDsL27Giib-k9QAUQ/exec'
    key = json.loads(requests.get(json_file).text)
    credentials = ServiceAccountCredentials.from_json_keyfile_dict(key, scope)
    gauth = GoogleAuth()
    gauth.credentials = credentials
    drive = GoogleDrive(gauth)

    sys.path.append('/content/drive/MyDrive/Colab Notebooks/modules')
    from google.colab import drive
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
from sklearn.model_selection import train_test_split, cross_val_score
from datetime import datetime
import matplotlib.pyplot as plt  # インポートを追加
import warnings
warnings.filterwarnings("ignore", category=UserWarning)  # UserWarningを無視

# サンプルデータの読み込み
df = pd.read_csv(import_path)

# 保存先CSVファイル名（実行日時を含める）
output_csv_path = export_path + f"LED_optimization_results_LR_individual_boxplot.csv"

# CSVにヘッダーを書き込む
with open(output_csv_path, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["iteration", "num_leds", "led_indices", "test_score"])  # ヘッダー

# サンプルを識別するキー（speciesとturbidityでグループ化）
df["sample_id"] = df["species"].astype(str) + "_" + df["turbidity"].astype(str)

# LED範囲リスト
LED_list = [
    [214, 236], [226, 254], [244, 266], [250, 270], [252, 258],
    [256, 276], [256, 278], [264, 294], [266, 292], [264, 286],
    [260, 290], [268, 290], [294, 330], [298, 318], [310, 350],
    [320, 366], [336, 368], [360, 384], [360, 412], [370, 404],
    [370, 396], [374, 398], [380, 414], [382, 408], [390, 424],
    [390, 430], [400, 434], [404, 442], [410, 450], [420, 470],
    [440, 484], [444, 504], [490, 570], [544, 574], [556, 590],
    [570, 604]
]

# 関数: 特定のサンプルのスペクトルを生成
def generate_combined_spectrum(df, sample_id, start_excitation, end_excitation):
    sample_data = df[df["sample_id"] == sample_id]
    selected_data = sample_data[
        (sample_data["wavelength"] >= start_excitation) &
        (sample_data["wavelength"] <= end_excitation)
    ]

    fluorescence_columns = [col for col in df.columns if col.replace('.', '', 1).isdigit()]
    wavelengths = np.array(list(map(float, fluorescence_columns)))

    fluorescence_data = selected_data.loc[:, fluorescence_columns]
    combined_spectrum = fluorescence_data.sum(axis=0)
    threshold_wavelength = end_excitation + 30
    combined_spectrum[wavelengths < threshold_wavelength] = 0

    return combined_spectrum

# 関数: 学習データセットを準備
def prepare_dataset(df, ranges):
    combined_spectra = []
    labels = []
    for sample_id in df["sample_id"].unique():
        spectra = []
        for start_excitation, end_excitation in ranges:
            combined_spectrum = generate_combined_spectrum(df, sample_id, start_excitation, end_excitation)
            spectra.extend(combined_spectrum)  # フラットに結合
        combined_spectra.append(spectra)
        labels.append(df[df["sample_id"] == sample_id]["species"].iloc[0])

    combined_spectra_df = pd.DataFrame(combined_spectra)
    combined_spectra_df["species"] = labels
    return combined_spectra_df

# 目的関数: Optunaによる最適化
def run_optimization_with_cv(train_df, num_leds):
    """
    指定した組み合わせ数でOptunaを実行
    """
    def objective(trial):
        led_indices = []
        for i in range(num_leds):
            led_index = trial.suggest_int(f"led_{i}", 0, len(LED_list) - 1)
            led_indices.append(led_index)

        # 重複を排除
        if len(set(led_indices)) < num_leds:
            raise optuna.TrialPruned()

        # 選択されたLED範囲
        ranges = [LED_list[i] for i in led_indices]
        dataset = prepare_dataset(train_df, ranges)
        X = dataset.drop(columns=["species"])
        y = dataset["species"]

        # データの標準化と正規化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        normalizer = MinMaxScaler()
        X_normalized = normalizer.fit_transform(X_scaled)

        # PCAによる次元削減
        pca = PCA(n_components=20, random_state=42)
        X_pca = pca.fit_transform(X_normalized)

        # モデル学習
        model = LogisticRegressionCV(Cs=50, cv=10, solver='lbfgs', max_iter=500, random_state=42)
        model.fit(X_pca, y)
        scores = model.score(X_pca, y)
        return np.mean(scores)

    # Optunaによる最適化
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=3)

    return study

# 組み合わせ数を増やしながら最適化を10回繰り返す
num_iterations = 3
num_leds_range = range(1, 5)  # 組み合わせ数を1から3まで

for iteration in range(1, num_iterations + 1):
    print(f"\n=== Iteration {iteration} ===")
    for num_leds in num_leds_range:
        print(f"\n--- Optimizing for {num_leds} LED combination(s) ---")

        # データの分割
        train_df, test_df = train_test_split(df, train_size = 0.85, test_size = 0.15, random_state=iteration, stratify=df["species"])
        # Optunaによる最適化
        study = run_optimization_with_cv(train_df, num_leds)

        # ベストトライアルのLEDインデックスを取得
        best_trial = study.best_trial
        best_led_indices = [best_trial.params[f"led_{i}"] for i in range(num_leds)]
        best_ranges = [LED_list[i] for i in best_led_indices]

        # ベストLED範囲でデータセットを準備
        train_dataset = prepare_dataset(train_df, best_ranges)
        test_dataset = prepare_dataset(test_df, best_ranges)
        X_train = train_dataset.drop(columns=["species"])
        y_train = train_dataset["species"]
        X_test = test_dataset.drop(columns=["species"])
        y_test = test_dataset["species"]

        # データの標準化と正規化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)  # テストセットを変換

        normalizer = MinMaxScaler()
        X_train_normalized = normalizer.fit_transform(X_train_scaled)
        X_test_normalized = normalizer.transform(X_test_scaled)  # テストセットを変換

        pca = PCA(n_components=20, random_state=42)
        X_train_pca = pca.fit_transform(X_train_normalized)
        X_test_pca = pca.transform(X_test_normalized)

        # モデル学習と評価
        model = LogisticRegressionCV(Cs=50, cv=10, solver='lbfgs', max_iter=500, random_state=42)
        model.fit(X_train_pca, y_train)
        CV_score = model.score(X_train_pca, y_train)
        test_score = model.score(X_test_pca, y_test)

        print(f"Best LED indices: {best_led_indices}, CV Score: {CV_score}, Test Score: {test_score:.4f}")

        # 結果の保存
        with open(output_csv_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([iteration, num_leds, best_ranges, test_score])

# 結果の可視化
results_df = pd.read_csv(output_csv_path)

plt.figure(figsize=(12, 6))
for num_leds in num_leds_range:
    subset = results_df[results_df["num_leds"] == num_leds]
    plt.boxplot(subset["test_score"], positions=[num_leds], widths=0.6, labels=[f"{num_leds} LEDs"])

plt.title("Test Scores by Number of LED Combinations Across Iterations")
plt.xlabel("Number of LEDs")
plt.ylabel("Test Accuracy")
plt.xticks(ticks=range(2, 4), labels=[f"{n} LEDs" for n in num_leds_range])
plt.grid(axis='y')
plt.show()