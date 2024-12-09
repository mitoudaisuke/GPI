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

# 必要なライブラリのインポート
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import PCA
import os
import warnings
import logging

# ワーニングを無視
warnings.filterwarnings('ignore')

# ...（既存のインポートと認証部分はそのまま）

# グローバル設定
RANDOM_STATE = 42
N_CV_SPLITS = 10  # クロスバリデーションの分割数
DESIRED_TOTAL_TRIALS = 100  # トータルで実行したいトライアル数
wavelength_counts = list(range(3, 11))  # 3から10までの波長数

# ...（既存のDataScalerクラスとその他の関数）

def objective(trial, df, wavelengths, classifier_hyperparams, num_wavelengths):
    # （上記の一般化した目的関数を使用）
    # ...
    pass  # 上記の目的関数をここに挿入

def save_trial_callback(study, trial):
    """各トライアルの結果をCSVに保存"""
    trial_result = {
        'trial_number': trial.number,
        'value': trial.value,
    }
    # 波長パラメータの動的取得
    for key, value in trial.params.items():
        if key.startswith('wavelength_'):
            trial_result[key] = value
        else:
            trial_result[key] = value

    num_wavelengths = sum(1 for key in trial.params if key.startswith('wavelength_'))
    trials_csv_path = os.path.join(export_path, f"optuna_trials_extratrees_{num_wavelengths}_wavelengths.csv")
    trials_df = pd.DataFrame([trial_result])

    if not os.path.exists(trials_csv_path):
        trials_df.to_csv(trials_csv_path, mode='a', index=False, header=True)
    else:
        trials_df.to_csv(trials_csv_path, mode='a', index=False, header=False)

def main():
    # データの読み込み
    try:
        df = pd.read_csv(import_path, header=0, index_col=0, encoding="sjis")
        logging.info(f"Data loaded successfully from {import_path}")
        print(f"Data loaded successfully from {import_path}")
    except Exception as e:
        logging.error(f"Error loading data from {import_path}: {e}")
        print(f"Error loading data from {import_path}: {e}")
        return

    # 波長の範囲を設定
    wavelengths = np.arange(200, 600, 2)  # 200nmから598nmまで、2nm刻み
    logging.info(f"Total wavelengths available: {len(wavelengths)}")
    print(f"Total wavelengths available: {len(wavelengths)}")

    # 各波長数ごとに最適化を実行
    for num_wavelengths in wavelength_counts:
        logging.info(f"Starting optimization for selecting {num_wavelengths} wavelengths.")
        print(f"Starting optimization for selecting {num_wavelengths} wavelengths.")

        # Optunaのスタディを作成（SQLiteを使用）
        storage_path = os.path.join(export_path, f"optuna_study_extratrees_{num_wavelengths}_wavelengths.db")
        storage_url = f"sqlite:///{storage_path}"
        sampler = optuna.samplers.TPESampler(n_startup_trials=DESIRED_TOTAL_TRIALS//10, seed=RANDOM_STATE)  # 最初の10%トライアルはランダム
        study = optuna.create_study(
            study_name=f"wavelength_{num_wavelengths}_optimization_extratrees",
            storage=storage_url,
            sampler=sampler,
            direction='maximize',
            load_if_exists=True
        )

        # トライアル数の計算
        existing_trials = len(study.trials)
        remaining_trials = classifier_hyperparams['ExtraTreesClassifier']['n_trials'] - existing_trials

        if remaining_trials <= 0:
            print(f"Study already has {existing_trials} trials, which meets or exceeds the desired {classifier_hyperparams['ExtraTreesClassifier']['n_trials']} trials.")
            logging.info(f"Study already has {existing_trials} trials, which meets or exceeds the desired {classifier_hyperparams['ExtraTreesClassifier']['n_trials']} trials.")
        else:
            print(f"Starting Optuna optimization for {remaining_trials} more trials for {num_wavelengths} wavelengths...")
            logging.info(f"Starting Optuna optimization for {remaining_trials} more trials for {num_wavelengths} wavelengths...")
            study.optimize(
                lambda trial: objective(trial, df, wavelengths, classifier_hyperparams, num_wavelengths),
                n_trials=remaining_trials,
                callbacks=[save_trial_callback],
                show_progress_bar=True
            )

            # 最適なトライアルの結果を表示
            print(f"\n=== Optimization Completed for {num_wavelengths} wavelengths ===")
            logging.info(f"=== Optimization Completed for {num_wavelengths} wavelengths ===")
            print(f"Best CV Accuracy: {study.best_value:.4f}")
            logging.info(f"Best CV Accuracy: {study.best_value:.4f}")
            print("Best Parameters:")
            logging.info("Best Parameters:")
            for key, value in study.best_trial.params.items():
                print(f"  {key}: {value}")
                logging.info(f"  {key}: {value}")

            # 最適なパラメータをCSVに保存
            best_params = study.best_trial.params.copy()
            best_score = study.best_trial.value
            # 波長のパラメータを分離
            wavelength_params = {k: v for k, v in best_params.items() if k.startswith('wavelength_')}
            other_params = {k: v for k, v in best_params.items() if not k.startswith('wavelength_')}

            best_params_df = pd.DataFrame([{
                **wavelength_params,
                'pca_n_components': best_params.get('pca_n_components'),
                **other_params,
                'cv_accuracy': best_score
            }])
            best_params_csv_path = os.path.join(export_path, f"best_params_extratrees_{num_wavelengths}_wavelengths.csv")
            best_params_df.to_csv(best_params_csv_path, index=False)
            logging.info(f"Best hyperparameters for {num_wavelengths} wavelengths have been saved to {best_params_csv_path}")
            print(f"Best hyperparameters for {num_wavelengths} wavelengths have been saved to {best_params_csv_path}")

# スクリプトの実行
if __name__ == "__main__":
    main()