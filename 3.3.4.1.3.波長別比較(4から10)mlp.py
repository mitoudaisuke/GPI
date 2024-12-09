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
    export_path = '/Users/mito/Desktop/mlp/mlp'

# 必要なライブラリのインポート
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPClassifier  # ExtraTreesClassifier から MLPClassifier に変更
from sklearn.decomposition import PCA
import os
import warnings
import logging

# ワーニングを無視
warnings.filterwarnings('ignore')

# ログの設定
logging.basicConfig(
    filename=os.path.join(export_path, 'optimization_log_mlp.log'),  # ファイル名を変更
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# グローバル設定
RANDOM_STATE = 42
N_CV_SPLITS = 3  # クロスバリデーションの分割数
DESIRED_TOTAL_TRIALS = 10  # トータルで実行したいトライアル数
wavelength_counts = list(range(3, 20))  # 3から19までの波長数

# MLPClassifier のハイパーパラメータ最適化設定
classifier_hyperparams = {
    'MLPClassifier': {
        'n_trials': DESIRED_TOTAL_TRIALS,  # Optuna のトライアル数
        'n_splits': N_CV_SPLITS,    # クロスバリデーションの分割数
        'pca_n_components': {'type': 'int', 'low': 15, 'high': 30},
        'params': {
            'hidden_layer_sizes': {'type': 'int', 'low': 50, 'high': 200},  # 例: 隠れ層のユニット数
            'activation': {'type': 'categorical', 'choices': ['relu', 'tanh', 'logistic']},
            'solver': {'type': 'categorical', 'choices': ['adam', 'sgd', 'lbfgs']},
            'alpha': {'type': 'loguniform', 'low': 1e-5, 'high': 1e-2},  # 正則化パラメータ
            'learning_rate_init': {'type': 'loguniform', 'low': 1e-4, 'high': 1e-2},
            'max_iter': {'type': 'int', 'low': 200, 'high': 1000},
            # 他のハイパーパラメータを必要に応じて追加
        }
    }
}

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

# Optuna のコールバック関数
def save_trial_callback(study, trial):
    """各トライアルの結果をCSVに保存"""
    trial_result = {
        'trial_number': trial.number,
        'value': trial.value,
    }
    # 波長パラメータの動的取得
    for key, value in trial.params.items():
        trial_result[key] = value

    # 波長数を計算
    num_wavelengths = sum(1 for key in trial.params if key.startswith('wavelength_'))
    trials_csv_path = os.path.join(export_path, f"optuna_trials_mlp_{num_wavelengths}_wavelengths.csv")  # ファイル名を変更
    trials_df = pd.DataFrame([trial_result])

    if not os.path.exists(trials_csv_path):
        trials_df.to_csv(trials_csv_path, mode='a', index=False, header=True)
    else:
        trials_df.to_csv(trials_csv_path, mode='a', index=False, header=False)

# Optuna の目的関数
def objective(trial, df, wavelengths, classifier_hyperparams, num_wavelengths):
    selected_wavelengths = []
    step = 2  # 波長選択のステップ
    min_gap = 2  # 波長間の最小ギャップ

    # 波長を順序を保って選択
    previous_wl = min(wavelengths)
    for i in range(num_wavelengths):
        if i == num_wavelengths - 1:
            wl = trial.suggest_int(f'wavelength_{i+1}', previous_wl + min_gap, max(wavelengths), step=step)
        else:
            wl = trial.suggest_int(
                f'wavelength_{i+1}',
                previous_wl + min_gap,
                max(wavelengths) - (min_gap * (num_wavelengths - i - 1)),
                step=step
            )
        selected_wavelengths.append(wl)
        previous_wl = wl

    logging.info(f"Trial {trial.number}: Selected Wavelengths: {selected_wavelengths}")
    print(f"\nTrial {trial.number}: Selected Wavelengths: {selected_wavelengths}")

    # 選択した波長のデータを抽出してマージ
    try:
        # 最初の波長のデータをベースとする
        merged_data = df[df['wavelength'] == selected_wavelengths[0]].copy()
        merged_data = merged_data.drop(columns=['wavelength'])

        # 各追加波長のデータをリネームしてマージ
        for wl in selected_wavelengths[1:]:
            temp_df = df[df['wavelength'] == wl].copy()
            temp_df = temp_df.drop(columns=['wavelength'])

            # 'turbidity' と 'species' 以外の列にサフィックスを追加
            temp_df = temp_df.rename(columns=lambda x: f"{x}_wl{wl}" if x not in ['turbidity', 'species'] else x)

            # 'turbidity' と 'species' でマージ
            merged_data = pd.merge(merged_data, temp_df, on=['turbidity', 'species'], how='inner')

    except Exception as e:
        logging.error(f"Error merging data for wavelengths {selected_wavelengths}: {e}")
        print(f"Error merging data for wavelengths {selected_wavelengths}: {e}")
        return 0.0  # マージに失敗した場合、スコアを0に設定

    # 特徴量とターゲットに分割
    drop_columns = ['species', 'turbidity']
    X_df = merged_data.drop(columns=drop_columns, errors='ignore')
    y = merged_data['species']

    logging.info(f"Merged Dataset shape: {X_df.shape}")
    print(f"Merged Dataset shape: {X_df.shape}")
    logging.info(f"Number of classes: {len(y.unique())}, Classes: {y.unique()}")
    print(f"Number of classes: {len(y.unique())}, Classes: {y.unique()}")

    # ハイパーパラメータをサンプリング
    params = {}
    for param_name, param_info in classifier_hyperparams['MLPClassifier']['params'].items():
        if param_info['type'] == 'int':
            params[param_name] = trial.suggest_int(param_name, param_info['low'], param_info['high'])
        elif param_info['type'] == 'loguniform':
            params[param_name] = trial.suggest_loguniform(param_name, param_info['low'], param_info['high'])
        elif param_info['type'] == 'categorical':
            params[param_name] = trial.suggest_categorical(param_name, param_info['choices'])
        elif param_info['type'] == 'fixed':
            params[param_name] = param_info['value']
        else:
            pass  # 他のタイプは未対応

    # PCAの成分数をサンプリング
    pca_config = classifier_hyperparams['MLPClassifier']['pca_n_components']
    if pca_config and pca_config['type'] == 'int':
        pca_n_components = trial.suggest_int('pca_n_components', pca_config['low'], pca_config['high'])
    else:
        pca_n_components = None

    # データのスケーリングとPCA変換
    scaler = DataScaler(pca_n_components=pca_n_components)
    X_scaled = scaler.fit_transform(X_df)

    # MLPClassifierの初期化
    model = MLPClassifier(**params, random_state=RANDOM_STATE)

    # クロスバリデーションスコアの計算
    try:
        scores = cross_val_score(model, X_scaled, y, cv=N_CV_SPLITS, scoring='accuracy', n_jobs=-1)
        mean_score = scores.mean()
        logging.info(f"Trial {trial.number}: CV Accuracy = {mean_score:.4f}")
        print(f"Trial {trial.number}: CV Accuracy = {mean_score:.4f}")
    except Exception as e:
        logging.error(f"Error during cross-validation for trial {trial.number}: {e}")
        print(f"Error during cross-validation for trial {trial.number}: {e}")
        mean_score = 0.0

    return mean_score

# メイン関数
def main():
    # データの読み込みパスと保存パスを設定
    # クラウド環境かローカル環境かに応じてパスが設定されています
    # 既に try-except ブロックで設定されているため、ここでは再度設定する必要はありません
    # import_path と export_path はグローバルスコープで設定されています

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
        storage_path = os.path.join(export_path, f"optuna_study_mlp_{num_wavelengths}_wavelengths.db")  # ファイル名を変更
        storage_url = f"sqlite:///{storage_path}"
        sampler = optuna.samplers.TPESampler(n_startup_trials=int(DESIRED_TOTAL_TRIALS / 10), seed=RANDOM_STATE)  # 最初の10%トライアルはランダム
        study = optuna.create_study(
            study_name=f"wavelength_{num_wavelengths}_optimization_mlp",  # 名前を変更
            storage=storage_url,
            sampler=sampler,
            direction='maximize',
            load_if_exists=True
        )

        # トライアル数の計算
        existing_trials = len(study.trials)
        remaining_trials = classifier_hyperparams['MLPClassifier']['n_trials'] - existing_trials  # キーを変更

        if remaining_trials <= 0:
            print(f"Study already has {existing_trials} trials, which meets or exceeds the desired {classifier_hyperparams['MLPClassifier']['n_trials']} trials.")
            logging.info(f"Study already has {existing_trials} trials, which meets or exceeds the desired {classifier_hyperparams['MLPClassifier']['n_trials']} trials.")
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
            best_params_csv_path = os.path.join(export_path, f"best_params_mlp_{num_wavelengths}_wavelengths.csv")  # ファイル名を変更
            best_params_df.to_csv(best_params_csv_path, index=False)
            logging.info(f"Best hyperparameters for {num_wavelengths} wavelengths have been saved to {best_params_csv_path}")
            print(f"Best hyperparameters for {num_wavelengths} wavelengths have been saved to {best_params_csv_path}")

# スクリプトの実行
if __name__ == "__main__":
    main()