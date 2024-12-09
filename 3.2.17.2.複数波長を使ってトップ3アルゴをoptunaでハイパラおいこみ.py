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
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# グローバル設定
RANDOM_STATE = 42
N_OUTER_SPLITS = 10  # 外側の分割数
N_SPLITS = 30        # 内側の交差検証分割数
TEST_SIZE = 0.15     # 検証データの比率（15%）

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

# 各分類器ごとのハイパーパラメータ最適化設定
classifier_hyperparams = {
    'ExtraTreesClassifier': {
        'n_trials': 20,
        'pca_n_components': {'type': 'int', 'low': 10, 'high': 50}
    },
    'MLPClassifier': {
        'n_trials': 20,
        'pca_n_components': {'type': 'int', 'low': 10, 'high': 50}
    },
    'LogisticRegressionCV': {
        'n_trials': 5,
        'pca_n_components': {'type': 'int', 'low': 10, 'high': 50}
    }
}

# PCAの成分数をサンプリングする関数
def sample_pca_n_components(trial, pca_config):
    if pca_config:
        if pca_config['type'] == 'int':
            return trial.suggest_int('pca_n_components', pca_config['low'], pca_config['high'])
    return None

# Optunaの目的関数
def objective_et(trial, X, y, pca_config):
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
    scores = cross_val_score(model, X_scaled, y, cv=N_SPLITS, scoring='accuracy')
    return scores.mean()

def objective_mlp(trial, X, y, pca_config):
    pca_n_components = sample_pca_n_components(trial, pca_config)

    n_layers = trial.suggest_int('n_layers', 1, 3)
    layer_units = [trial.suggest_int(f'n_units_l{i}', 10, 100) for i in range(n_layers)]

    params = {
        'hidden_layer_sizes': tuple(layer_units),  # n_layers を削除して、tuple にまとめる
        'alpha': trial.suggest_loguniform('alpha', 1e-5, 1e-1),
        'learning_rate_init': trial.suggest_loguniform('learning_rate_init', 1e-4, 1e-1),
        'max_iter': 500,
        'random_state': RANDOM_STATE
    }

    model = MLPClassifier(**params)
    scaler = DataScaler(pca_n_components=pca_n_components)
    X_scaled = scaler.fit_transform(X)
    scores = cross_val_score(model, X_scaled, y, cv=N_SPLITS, scoring='accuracy')
    return scores.mean()

def objective_lr(trial, X, y, pca_config):
    pca_n_components = sample_pca_n_components(trial, pca_config)

    n_cs = trial.suggest_int('Cs', 3, 10)
    params = {
        'Cs': n_cs,
        'cv': N_SPLITS,
        'max_iter': 500,
        'random_state': RANDOM_STATE
    }

    model = LogisticRegressionCV(**params)
    scaler = DataScaler(pca_n_components=pca_n_components)
    X_scaled = scaler.fit_transform(X)
    scores = cross_val_score(model, X_scaled, y, cv=N_SPLITS, scoring='accuracy')
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
    pca_config = hyperparams.get('pca_n_components', None)

    # 目的関数を選択し、分割数とPCA構成を渡す
    if classifier_name == 'ExtraTreesClassifier':
        objective = lambda trial: objective_et(trial, X, y, pca_config)
    elif classifier_name == 'MLPClassifier':
        objective = lambda trial: objective_mlp(trial, X, y, pca_config)
    else:  # LogisticRegressionCV
        objective = lambda trial: objective_lr(trial, X, y, pca_config)

    study = optuna.create_study(direction='maximize')
    trials_history = []
    start_time = datetime.now()

    def callback(study, trial):
        record = create_trial_record(trial, study, classifier_name, split_number, start_time)
        trials_history.append(record)

    study.optimize(objective, n_trials=n_trials, callbacks=[callback], show_progress_bar=True)

    best_trial = study.best_trial
    # n_layersとpca_n_componentsを除外してbest_paramsを設定
    best_params = {k: v for k, v in best_trial.params.items() if k not in ["pca_n_components", "n_layers"]}
    best_pca_n_components = best_trial.params.get("pca_n_components", None)

    return best_params, study.best_value, best_pca_n_components, pd.DataFrame(trials_history)

def evaluate_classifiers(X, y, selected_classifiers, classifier_hyperparams):
    """複数の分類器を評価し、ハイパーパラメータ最適化を実行し、混同行列を1つのCSVにまとめる"""
    results = []
    all_trials = []
    all_confusion_matrices = []

    for split in range(N_OUTER_SPLITS):
        print(f"\n=== Outer Split {split + 1}/{N_OUTER_SPLITS} ===")

        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE + split, stratify=y)

        for name, classifier_info in selected_classifiers.items():
            try:
                print(f"\nOptimizing {name}...")

                hyperparams = classifier_hyperparams.get(name, {})
                best_params, best_cv_score, pca_n_components, trials_df = optimize_hyperparameters(
                    name, X_train_val, y_train_val, split + 1, hyperparams)

                all_trials.append(trials_df)

                scaler = DataScaler(pca_n_components=pca_n_components)
                X_train_val_scaled = scaler.fit_transform(X_train_val)
                X_test_scaled = scaler.transform(X_test)

                # モデルのインスタンス化に必要なパラメータを設定
                if name == 'LogisticRegressionCV':
                    model = classifier_info['class'](**best_params, cv=N_SPLITS)
                else:
                    model = classifier_info['class'](**best_params)

                model.fit(X_train_val_scaled, y_train_val)
                test_score = model.score(X_test_scaled, y_test)

                # 混同行列を計算
                y_pred = model.predict(X_test_scaled)
                cm = confusion_matrix(y_test, y_pred)

                # 混同行列を長形式に変換して保存
                for i, true_label in enumerate(model.classes_):
                    for j, pred_label in enumerate(model.classes_):
                        all_confusion_matrices.append({
                            'Classifier': name,
                            'Split': split + 1,
                            'True_Label': true_label,
                            'Predicted_Label': pred_label,
                            'Count': cm[i, j]
                        })

                # 評価結果を記録
                result_dict = {
                    'Algorithm': name,
                    'CV_Score': best_cv_score,
                    'Test_Score': test_score,
                    'Split': split + 1,
                    'pca_n_components': pca_n_components
                }

                for param_name, param_value in best_params.items():
                    result_dict[f'param_{param_name}'] = param_value

                results.append(result_dict)

                print(f"{name} - CV Score: {best_cv_score:.4f}, Test Score: {test_score:.4f}")

            except Exception as e:
                print(f"Error optimizing {name}: {str(e)}")
                continue

    if all_trials:
        all_trials_df = pd.concat(all_trials, ignore_index=True)
        all_trials_df.to_csv(f"{export_path}optimization_history.csv", index=False)

    # 混同行列を1つのCSVに保存
    confusion_matrix_df = pd.DataFrame(all_confusion_matrices)
    confusion_matrix_df.to_csv(f"{export_path}confusion_matrices.csv", index=False)

    return pd.DataFrame(results)

# メイン実行部分
if __name__ == "__main__":
    # データの読み込み
    df = pd.read_csv(import_path, header=0, index_col=0, encoding="sjis")
    wavelengths = np.arange(200,600,10)
    dfs = [df[df['wavelength'] == wl].copy() for wl in wavelengths]

    # サンプルごとに波長データを結合
    merged_data = dfs[0][['turbidity', 'species']].copy()
    for i, wl in enumerate(wavelengths):
        wl_columns = [col for col in dfs[i].columns if col not in ['turbidity', 'species', 'wavelength']]
        wl_renamed = {col: f"{col}_wl{wl}" for col in wl_columns}
        dfs[i].rename(columns=wl_renamed, inplace=True)
        merged_data = merged_data.merge(dfs[i][['turbidity', 'species'] + list(wl_renamed.values())],
                                        on=['turbidity', 'species'], how='inner')

    X_df = merged_data.drop(['species', 'turbidity'], axis=1)
    y = merged_data['species']

    print("Merged Dataset shape:", X_df.shape)
    print("Number of classes:", len(y.unique()))
    print("Classes:", y.unique())

    selected_classifiers = {
        'ExtraTreesClassifier': {'class': ExtraTreesClassifier, 'params': 'et'},
        'MLPClassifier': {'class': MLPClassifier, 'params': 'mlp'},
        'LogisticRegressionCV': {'class': LogisticRegressionCV, 'params': 'lr'}
    }

    results_df = evaluate_classifiers(X_df, y, selected_classifiers, classifier_hyperparams)
    results_df.to_csv(f"{export_path}final_results.csv", index=False)

    print(f"\nResults have been saved to {export_path}")