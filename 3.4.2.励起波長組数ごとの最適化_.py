import json
import requests
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from oauth2client.service_account import ServiceAccountCredentials
import warnings
import time

# 特定の警告を無視
warnings.filterwarnings(action='ignore')

# Google Drive認証とパス設定
try:
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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage
from matplotlib.ticker import MaxNLocator, MultipleLocator, AutoMinorLocator
import matplotlib.colors
import matplotlib.cm as cm
import csv
import os
import optuna
from optuna.samplers import TPESampler
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split, cross_val_score

# サンプルデータの読み込み
df = pd.read_csv(import_path)

# サンプルを識別するキー（speciesとturbidityでグループ化）
df["sample_id"] = df["species"].astype(str) + "_" + df["turbidity"].astype(str)

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

def run_optimization_for_multiple_ranges(num_ranges, output_csv_path):
    """
    複数の開始値と終了値のペアをOptunaで最適化し、結果をCSVに保存
    """
    def objective(trial):
        """
        Optunaで最適化する目的関数
        """
        ranges = []
        for i in range(num_ranges):
            start = trial.suggest_int(f"start_{i}", 250, 598, step=2)  # 開始値（2刻み）
            end = trial.suggest_int(f"end_{i}", start + 2, 600, step=2)  # 終了値（開始値より大きくする）
            ranges.append((start, end))

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

        # データを学習用とテスト用に分割
        X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42, stratify=y)

        # LogisticRegressionCVの学習と評価
        model = LogisticRegressionCV(
            Cs=5,
            cv=10,
            solver='lbfgs',
            max_iter=500,
            random_state=42
        )
        model.fit(X_train, y_train)
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
        return np.mean(scores)

    # 最初の400回をランダムに、それ以降をTPEで最適化
    sampler = TPESampler(
        n_startup_trials=400,  # 最初の400回をランダム
        consider_prior=True, 
        prior_weight=1.0,
        consider_magic_clip=True,
        consider_endpoints=True,
        n_ei_candidates=24,
        multivariate=True,
    )

    # Optunaによる最適化
    study = optuna.create_study(direction="maximize", sampler=sampler)

    # 最適化を進めながら途中経過を出力
    def logging_callback(study, trial):
        # 試行ごとにログを出力
        print(
            f"Trial {trial.number} completed. "
            f"Params: {trial.params}. "
            f"Score: {trial.value:.4f}. "
            f"Best score so far: {study.best_value:.4f}"
        )

    study.optimize(objective, n_trials=500, callbacks=[logging_callback])  # 試行回数を500に設定

    # 結果をCSVに保存
    with open(output_csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["n_combinations", "trial", "ranges", "score"])  # ヘッダー
        for trial in study.trials:
            trial_ranges = [
                (trial.params.get(f"start_{i}"), trial.params.get(f"end_{i}")) for i in range(num_ranges)
            ]
            writer.writerow([num_ranges, trial.number, trial_ranges, trial.value])

def prepare_dataset(df, ranges):
    """
    全サンプルに対して範囲でスペクトラムを合成し、学習用データセットを構築
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

# ペア数を選択
while True:
    try:
        num_ranges = int(input("Enter the number of ranges to optimize (1-5): "))
        if num_ranges < 1 or num_ranges > 5:
            raise ValueError("Please enter a number between 1 and 5.")
        break
    except ValueError as e:
        print(e)

suffixes = ["one", "two", "three", "four", "five"]
output_csv_path = export_path + f"dynamic_{suffixes[num_ranges - 1]}_optimization_results.csv"

# 選択したペア数で最適化を実行
print(f"\n=== Optimizing for {num_ranges} range(s) ===")
run_optimization_for_multiple_ranges(num_ranges, output_csv_path)

print("Optimization completed.")