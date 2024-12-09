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
except:
    import_path='/Users/mito/Library/Mobile Documents/com~apple~CloudDocs/python/EEM.csv'
    export_path ='/Users/mito/Desktop/'

# 必要なライブラリのインポート
import numpy as np
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.decomposition import PCA
import itertools
import warnings
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import joblib

warnings.filterwarnings('ignore')

# グローバル設定
RANDOM_STATE = 42
N_OUTER_SPLITS = 5  # 外側の分割数

# 各分類器ごとのハイパーパラメータ最適化設定
classifier_hyperparams = {
    'ExtraTreesClassifier': {
        'n_trials': 10,
        'n_splits': 5,
        'pca_n_components': {'type': 'int', 'low': 10, 'high': 50}
    },
    'MLPClassifier': {
        'n_trials': 3,
        'n_splits': 3,
        'pca_n_components': {'type': 'int', 'low': 10, 'high': 50}
    },
    'LogisticRegressionCV': {
        'n_trials': 8,
        'n_splits': 3,
        'pca_n_components': {'type': 'int', 'low': 10, 'high': 50}
    }
}

# 結果を保存するCSVのパス
results_csv_path = os.path.join(export_path, "wavelength_comparison_results.csv")
trials_history_csv_path = os.path.join(export_path, "wavelength_comparison_optimization_history.csv")

# データスケーリングクラス
class DataScaler:
    def __init__(self, pca_n_components=None):
        self.std_scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()
        self.pca_n_components = pca_n_components
        self.pca = None

    def fit_transform(self, data):
        """訓練データ用のスケーリングとPCA変換"""
        original_shape = data.values.shape
        data_1d = data.values.ravel()
        std_scaled_1d = self.std_scaler.fit_transform(data_1d.reshape(-1, 1))
        final_scaled_1d = self.minmax_scaler.fit_transform(std_scaled_1d)
        final_scaled_data = final_scaled_1d.reshape(original_shape)
        df_scaled = pd.DataFrame(final_scaled_data, index=data.index, columns=data.columns)
        # PCAの設定
        if self.pca_n_components:
            self.pca = PCA(n_components=self.pca_n_components, random_state=RANDOM_STATE)
            df_pca = self.pca.fit_transform(df_scaled)
            return pd.DataFrame(df_pca, index=df_scaled.index)
        return df_scaled

    def transform(self, data):
        """検証・テストデータ用のスケーリングとPCA変換"""
        original_shape = data.values.shape
        data_1d = data.values.ravel()
        std_scaled_1d = self.std_scaler.transform(data_1d.reshape(-1, 1))
        final_scaled_1d = self.minmax_scaler.transform(std_scaled_1d)
        final_scaled_data = final_scaled_1d.reshape(original_shape)
        df_scaled = pd.DataFrame(final_scaled_data, index=data.index, columns=data.columns)
        # PCAの適用
        if self.pca:
            df_pca = self.pca.transform(df_scaled)
            return pd.DataFrame(df_pca, index=df_scaled.index)
        return df_scaled

# PCAの成分数をサンプリングする関数
def sample_pca_n_components(trial, pca_config):
    if pca_config:
        if pca_config['type'] == 'int':
            return trial.suggest_int('pca_n_components', pca_config['low'], pca_config['high'])
    return None

# Optunaの目的関数
def objective_et(trial, X, y, n_splits, pca_config):
    pca_n_components = sample_pca_n_components(trial, pca_config)

    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 5, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
        'random_state': RANDOM_STATE
    }
    model = ExtraTreesClassifier(**params)
    scaler = DataScaler(pca_n_components=pca_n_components)
    X_scaled = scaler.fit_transform(X)
    scores = cross_val_score(model, X_scaled, y, cv=n_splits, scoring='accuracy', n_jobs=-1)
    return scores.mean()

def objective_mlp(trial, X, y, n_splits, pca_config):
    pca_n_components = sample_pca_n_components(trial, pca_config)

    # 隠れ層数と各層のユニット数を設定
    n_layers = trial.suggest_int('n_layers', 1, 3)
    layer_units = [trial.suggest_int(f'n_units_l{i}', 10, 100) for i in range(n_layers)]
    hidden_layer_sizes = tuple(layer_units)  # 各層のユニット数をタプルにまとめる

    params = {
        'hidden_layer_sizes': hidden_layer_sizes,  # hidden_layer_sizesにタプルを直接渡す
        'alpha': trial.suggest_loguniform('alpha', 1e-5, 1e-1),
        'learning_rate_init': trial.suggest_loguniform('learning_rate_init', 1e-4, 1e-1),
        'max_iter': 500,
        'random_state': RANDOM_STATE
    }

    model = MLPClassifier(**params)
    scaler = DataScaler(pca_n_components=pca_n_components)
    X_scaled = scaler.fit_transform(X)
    scores = cross_val_score(model, X_scaled, y, cv=n_splits, scoring='accuracy', n_jobs=-1)
    return scores.mean()

def objective_lr(trial, X, y, n_splits, pca_config):
    pca_n_components = sample_pca_n_components(trial, pca_config)

    n_cs = trial.suggest_int('Cs', 3, 10)
    params = {
        'Cs': n_cs,
        'cv': n_splits,
        'max_iter': 500,
        'random_state': RANDOM_STATE
    }

    model = LogisticRegressionCV(**params)
    scaler = DataScaler(pca_n_components=pca_n_components)
    X_scaled = scaler.fit_transform(X)
    scores = cross_val_score(model, X_scaled, y, cv=n_splits, scoring='accuracy', n_jobs=-1)
    return scores.mean()

# 試行の記録を作成する関数
def create_trial_record(trial, study, algorithm, split, start_time):
    """単一試行の記録を作成"""
    record = {
        'Algorithm': algorithm,
        'Split': split,
        'Trial': trial.number,
        'Value': trial.value,
        'Best_Value': study.best_value,
        'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'Duration': (datetime.now() - start_time).total_seconds()
    }
    for param_name, param_value in trial.params.items():
        record[f'param_{param_name}'] = param_value
    return record

# ハイパーパラメータを最適化する関数
def optimize_hyperparameters(classifier_name, X, y, split_number, hyperparams):
    """指定された分類器のハイパーパラメータ最適化を実行し、記録を保存"""
    n_trials = hyperparams.get('n_trials', 50)
    n_splits_for_optuna = hyperparams.get('n_splits', 5)
    pca_config = hyperparams.get('pca_n_components', None)

    # 目的関数を選択し、分割数とPCA構成を渡す
    if classifier_name == 'ExtraTreesClassifier':
        objective = lambda trial: objective_et(trial, X, y, n_splits_for_optuna, pca_config)
    elif classifier_name == 'MLPClassifier':
        objective = lambda trial: objective_mlp(trial, X, y, n_splits_for_optuna, pca_config)
    else:  # LogisticRegressionCV
        objective = lambda trial: objective_lr(trial, X, y, n_splits_for_optuna, pca_config)

    study = optuna.create_study(direction='maximize')
    trials_history = []
    start_time = datetime.now()

    def callback(study, trial):
        record = create_trial_record(trial, study, classifier_name, split_number, start_time)
        trials_history.append(record)
        # 即時に保存することで途中で停止してもデータが失われないようにする
        trials_df = pd.DataFrame([record])
        # ロックを使用して排他制御（簡易的に実装）
        with joblib.parallel_backend('threading'):
            if not os.path.exists(trials_history_csv_path):
                trials_df.to_csv(trials_history_csv_path, mode='a', index=False, header=True)
            else:
                trials_df.to_csv(trials_history_csv_path, mode='a', index=False, header=False)

    study.optimize(objective, n_trials=n_trials, callbacks=[callback], show_progress_bar=False)

    best_trial = study.best_trial

    # 基本的なパラメータの除外
    best_params = {k: v for k, v in best_trial.params.items() if k not in ["pca_n_components"]}

    # 分類器ごとの追加処理
    if classifier_name == 'MLPClassifier':
        n_layers = best_trial.params.get('n_layers')
        hidden_layer_sizes = tuple(best_trial.params.get(f'n_units_l{i}') for i in range(n_layers))
        best_params['hidden_layer_sizes'] = hidden_layer_sizes
        # 'n_units_l*' と 'n_layers' を除外
        best_params = {k: v for k, v in best_params.items() if not (k.startswith('n_units_l') or k == 'n_layers')}

    # 他の分類器の場合は必要に応じて追加処理

    best_pca_n_components = best_trial.params.get("pca_n_components", None)

    return best_params, study.best_value, best_pca_n_components

# 波長の処理を行う関数
def process_wavelength(wl, df, classifier_hyperparams):
    print(f"=== Evaluating wavelength: {wl} nm ===")

    try:
        # 波長 `wl` のデータのみを使用
        df_wl = df[df['wavelength'] == wl].copy()

        # 必要な列のみ抽出し、訓練データとターゲットに分割
        X_df = df_wl.drop(['species', 'turbidity', 'wavelength'], axis=1)
        y = df_wl['species']

        print("Dataset shape:", X_df.shape)
        print("Number of classes:", len(y.unique()))
        print("Classes:", y.unique())

        # モデル毎に精度を算出
        selected_classifiers = {
            'ExtraTreesClassifier': {'class': ExtraTreesClassifier, 'params': 'et'},
            'MLPClassifier': {'class': MLPClassifier, 'params': 'mlp'},
            'LogisticRegressionCV': {'class': LogisticRegressionCV, 'params': 'lr'}
        }

        results = []

        for split in range(N_OUTER_SPLITS):
            print(f"--- Split {split + 1}/{N_OUTER_SPLITS} ---")

            X_train_val, X_test, y_train_val, y_test = train_test_split(
                X_df, y, test_size=0.1, random_state=RANDOM_STATE + split, stratify=y)

            for name, classifier_info in selected_classifiers.items():
                try:
                    print(f"Optimizing {name} for Split {split + 1}...")

                    hyperparams = classifier_hyperparams.get(name, {})
                    best_params, best_cv_score, pca_n_components = optimize_hyperparameters(
                        name, X_train_val, y_train_val, split + 1, hyperparams)

                    scaler = DataScaler(pca_n_components=pca_n_components)
                    X_train_val_scaled = scaler.fit_transform(X_train_val)
                    X_test_scaled = scaler.transform(X_test)

                    # best_paramsを直接使用
                    model_params = best_params
                    model = classifier_info['class'](**model_params)

                    model.fit(X_train_val_scaled, y_train_val)
                    test_score = model.score(X_test_scaled, y_test)

                    # 評価結果を記録
                    result_dict = {
                        'Algorithm': name,
                        'CV_Score': best_cv_score,
                        'Test_Score': test_score,
                        'Split': split + 1,
                        'pca_n_components': pca_n_components,
                        'Wavelength': wl
                    }

                    for param_name, param_value in best_params.items():
                        result_dict[f'param_{param_name}'] = param_value

                    results.append(result_dict)

                    print(f"{name} - CV Score: {best_cv_score:.4f}, Test Score: {test_score:.4f}")

                except Exception as e:
                    print(f"Error optimizing {name} for Split {split + 1}: {str(e)}")
                    continue

        # 結果をDataFrameに変換
        results_df = pd.DataFrame(results)
        return results_df

    except Exception as e:
        print(f"Error processing wavelength {wl}: {str(e)}")
        return pd.DataFrame()  # 空のDataFrameを返す

def main():
    # データの読み込み
    df = pd.read_csv(import_path, header=0, index_col=0, encoding="sjis")
    wavelengths = np.arange(200, 600, 2)  # 2ずつ飛ばして波長を指定

    print(f"Total wavelengths to evaluate: {len(wavelengths)}")

    # 既存の結果を読み込み、既に処理済みの波長を特定
    if os.path.exists(results_csv_path):
        existing_results = pd.read_csv(results_csv_path)
        processed_wls = set(existing_results['Wavelength'].unique())
        print(f"Already processed wavelengths: {len(processed_wls)}")
    else:
        existing_results = pd.DataFrame()
        processed_wls = set()

    # 未処理の波長を特定
    remaining_wls = [wl for wl in wavelengths if wl not in processed_wls]
    print(f"Remaining wavelengths to process: {len(remaining_wls)}")

    if not remaining_wls:
        print("All wavelengths have been processed.")
        return

    # 並列処理の設定
    max_workers = os.cpu_count() or 4  # 利用可能なCPUコア数を取得
    print(f"Using {max_workers} workers for parallel processing.")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 各波長を並列で処理
        future_to_wl = {executor.submit(process_wavelength, wl, df, classifier_hyperparams): wl for wl in remaining_wls}

        for future in as_completed(future_to_wl):
            wl = future_to_wl[future]
            try:
                results_df = future.result()
                if not results_df.empty:
                    # 結果を即時に保存
                    if not os.path.exists(results_csv_path):
                        results_df.to_csv(results_csv_path, mode='a', index=False, header=True)
                    else:
                        results_df.to_csv(results_csv_path, mode='a', index=False, header=False)
                    print(f"Saved results for wavelength {wl} nm")
            except Exception as e:
                print(f"Error in processing wavelength {wl}: {str(e)}")
                continue

    print(f"\nAll wavelength comparison results have been saved to {results_csv_path}")

if __name__ == "__main__":
    main()