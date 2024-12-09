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
#    drive.mount('/content/drive')
    import_path='/content/drive/MyDrive/Colab Notebooks/import_data/EEM_.csv'
    export_path ='/content/drive/MyDrive/Colab Notebooks/export_data/'
    directory='/content/drive/MyDrive/Colab Notebooks/import_data/MCS/'
except:
    import_path='/Users/mito/Library/Mobile Documents/com~apple~CloudDocs/python/EEM_.csv'
    export_path ='/Users/mito/Desktop/'
    directory='/Users/mito/Library/Mobile Documents/com~apple~CloudDocs/python/手動MCS/'

# 必要なライブラリのインポート
import numpy as np
from datetime import datetime  # これを追加
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegressionCV
import warnings
warnings.filterwarnings('ignore')

# グローバル設定
MAX_ITER = 500
RANDOM_STATE = 42
N_OUTER_SPLITS = 2  # 外側の分割数
DEFAULT_N_TRIALS = 5
DEFAULT_N_SPLITS_FOR_OPTUNA = 3

# 各分類器ごとのハイパーパラメータ最適化設定
classifier_hyperparams = {
    'ExtraTreesClassifier': {
        'n_trials': 5,
        'n_splits': 3,
        'pca_n_components': {'type': 'int', 'low': 10, 'high': 50}
    },
    'MLPClassifier': {
        'n_trials': 10,
        'n_splits': 5,
        'pca_n_components': {'type': 'int', 'low': 10, 'high': 50}
    },
    'LogisticRegressionCV': {
        'n_trials': 2,
        'n_splits': 2,
        'pca_n_components': {'type': 'int', 'low': 10, 'high': 50}
    }
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
        std_scaled_1d = self.std_scaler.fit_transform(data_1d.reshape(-1, 1))
        final_scaled_1d = self.minmax_scaler.fit_transform(std_scaled_1d)
        final_scaled_data = final_scaled_1d.reshape(original_shape)
        return pd.DataFrame(final_scaled_data,
                            index=data.index,
                            columns=data.columns)

    def transform(self, data):
        """検証・テストデータ用：既存の統計量で変換のみを行う"""
        original_shape = data.values.shape
        data_1d = data.values.ravel()
        std_scaled_1d = self.std_scaler.transform(data_1d.reshape(-1, 1))
        final_scaled_1d = self.minmax_scaler.transform(std_scaled_1d)
        final_scaled_data = final_scaled_1d.reshape(original_shape)
        return pd.DataFrame(final_scaled_data,
                            index=data.index,
                            columns=data.columns)

# Optuna目的関数
def objective_et(trial, X, y, n_splits):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 5, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
        'random_state': RANDOM_STATE
    }
    model = ExtraTreesClassifier(**params)
    scores = cross_val_score(model, X, y, cv=n_splits, scoring='accuracy')
    return scores.mean()

def objective_mlp(trial, X, y, n_splits):
    n_layers = trial.suggest_int('n_layers', 1, 3)
    layer_units = []
    for i in range(n_layers):
        layer_units.append(trial.suggest_int(f'n_units_l{i}', 10, 100))

    params = {
        'hidden_layer_sizes': tuple(layer_units),
        'alpha': trial.suggest_loguniform('alpha', 1e-5, 1e-1),
        'learning_rate_init': trial.suggest_loguniform('learning_rate_init', 1e-4, 1e-1),
        'max_iter': MAX_ITER,
        'random_state': RANDOM_STATE
    }

    model = MLPClassifier(**params)
    scores = cross_val_score(model, X, y, cv=n_splits, scoring='accuracy')
    return scores.mean()

def objective_lr(trial, X, y, n_splits):
    Cs = trial.suggest_int('Cs', 3, 10)
    params = {
        'Cs': Cs,
        'cv': n_splits,
        'max_iter': MAX_ITER,
        'random_state': RANDOM_STATE
    }

    model = LogisticRegressionCV(**params)
    scores = cross_val_score(model, X, y, cv=n_splits, scoring='accuracy')
    return scores.mean()

def get_model_params(name, best_params):
    """最適化されたパラメータを適切な形式に変換"""
    if name == 'MLPClassifier':
        layer_units = []
        n_layers = best_params.get('n_layers', 1)
        for i in range(n_layers):
            layer_units.append(best_params.get(f'n_units_l{i}', 10))
        return {
            'hidden_layer_sizes': tuple(layer_units),
            'alpha': best_params.get('alpha', 1e-4),
            'learning_rate_init': best_params.get('learning_rate_init', 1e-3),
            'max_iter': MAX_ITER,
            'random_state': RANDOM_STATE
        }
    elif name == 'ExtraTreesClassifier':
        return {
            'n_estimators': best_params.get('n_estimators', 100),
            'max_depth': best_params.get('max_depth', None),
            'min_samples_split': best_params.get('min_samples_split', 2),
            'min_samples_leaf': best_params.get('min_samples_leaf', 1),
            'random_state': RANDOM_STATE
        }
    elif name == 'LogisticRegressionCV':
        return {
            'Cs': best_params.get('Cs', 5),
            'cv': best_params.get('cv', DEFAULT_N_SPLITS_FOR_OPTUNA),
            'max_iter': MAX_ITER,
            'random_state': RANDOM_STATE
        }
    return best_params

def create_trial_record(trial, study, algorithm, split, start_time):
    """試行の記録を作成"""
    record = {
        'Algorithm': algorithm,
        'Split': split,
        'Trial': trial.number,
        'Value': trial.value,
        'Best_Value': study.best_value,  # studyから直接best_valueを取得
        'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'Duration': (datetime.now() - start_time).total_seconds()
    }

    # アルゴリズム固有のパラメータを追加
    for param_name, param_value in trial.params.items():
        record[f'param_{param_name}'] = param_value

    return record

def optimize_hyperparameters(classifier_name, classifier_info, X, y, split_number):
    """ハイパーパラメータの最適化を実行し、履歴を記録"""
    # 各分類器ごとの設定を取得
    hyperparams = classifier_hyperparams.get(classifier_name, {
        'n_trials': DEFAULT_N_TRIALS,
        'n_splits': DEFAULT_N_SPLITS_FOR_OPTUNA,
        'pca_n_components': {'type': 'int', 'low': 10, 'high': 50}
    })
    n_trials = hyperparams['n_trials']
    n_splits_for_optuna = hyperparams['n_splits']

    # 分類器に応じたOptunaの設定を動的に調整
    if classifier_name == 'ExtraTreesClassifier':
        objective = objective_et
    elif classifier_name == 'MLPClassifier':
        objective = objective_mlp
    else:  # LogisticRegressionCV
        objective = objective_lr

    study = optuna.create_study(direction='maximize')
    trials_history = []
    start_time = datetime.now()

    def callback(study, trial):
        record = create_trial_record(trial, study, classifier_name, split_number, start_time)
        trials_history.append(record)

    # n_trialsとn_splits_for_optunaを動的に設定してoptuna.optimizeを実行
    study.optimize(lambda trial: objective(trial, X, y, n_splits_for_optuna),
                  n_trials=n_trials,
                  callbacks=[callback],
                  show_progress_bar=True)

    return study.best_params, study.best_value, pd.DataFrame(trials_history)

def plot_optimization_progress(all_trials_df, export_path):
    """最適化の進行を可視化"""
    if all_trials_df.empty:
        print("No trials to plot.")
        return

    # 1. アルゴリズムごとの進行
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    for algorithm in all_trials_df['Algorithm'].unique():
        data = all_trials_df[all_trials_df['Algorithm'] == algorithm]
        plt.plot(data['Trial'], data['Best_Value'], label=algorithm)

    plt.title('Optimization Progress by Algorithm', fontsize=12, fontweight='bold')
    plt.xlabel('Trial', fontsize=10)
    plt.ylabel('Best Value', fontsize=10)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. 分割ごとの進行
    plt.subplot(1, 2, 2)
    for split in all_trials_df['Split'].unique():
        data = all_trials_df[all_trials_df['Split'] == split]
        plt.plot(data['Trial'], data['Value'], label=f'Split {split}', alpha=0.5)

    plt.title('Optimization Progress by Split', fontsize=12, fontweight='bold')
    plt.xlabel('Trial', fontsize=10)
    plt.ylabel('Value', fontsize=10)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{export_path}optimization_progress.png", dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_classifiers(X, y, selected_classifiers):
    """分類器の評価を実行"""
    results = []
    all_trials = []

    for split in range(N_OUTER_SPLITS):
        print(f"\nOuter Split {split + 1}/{N_OUTER_SPLITS}")

        # データの分割
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=0.15, random_state=RANDOM_STATE + split, stratify=y)

        for name, classifier_info in selected_classifiers.items():
            try:
                print(f"\nOptimizing {name}...")

                # ハイパーパラメータの最適化
                best_params, best_cv_score, trials_df = optimize_hyperparameters(
                    name, classifier_info, X_train_val, y_train_val, split + 1)

                all_trials.append(trials_df)

                # パイプラインの再構築
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('minmax', MinMaxScaler()),
                    ('pca', PCA(n_components=best_params.get('pca_n_components', 10))),
                    ('classifier', classifier_info['class'](**get_model_params(name, best_params)))
                ])

                # モデルの訓練
                pipeline.fit(X_train_val, y_train_val)
                test_score = pipeline.score(X_test, y_test)

                # 結果を保存
                result_dict = {
                    'Algorithm': name,
                    'CV_Score': best_cv_score,
                    'Test_Score': test_score,
                    'Split': split + 1,
                    'pca_n_components': best_params.get('pca_n_components', 10)
                }

                # パラメータも保存
                for param_name, param_value in best_params.items():
                    result_dict[f'param_{param_name}'] = param_value

                results.append(result_dict)

                print(f"{name} - CV Score: {best_cv_score:.4f}, Test Score: {test_score:.4f}")

            except Exception as e:
                print(f"Error with {name}: {str(e)}")
                continue

    # 最適化履歴をまとめて保存
    if all_trials:
        all_trials_df = pd.concat(all_trials, ignore_index=True)
        all_trials_df.to_csv(f"{export_path}optimization_history.csv", index=False)

        # 最適化の進行を可視化
        plot_optimization_progress(all_trials_df, export_path)

    return pd.DataFrame(results)

def plot_final_results(results_df, export_path):
    """最終結果の可視化"""
    if results_df.empty:
        print("No results to plot.")
        return

    plt.figure(figsize=(15, 6))

    # 左：クロスバリデーションスコア
    plt.subplot(1, 2, 1)
    sns.boxplot(data=results_df, x='Algorithm', y='CV_Score')
    plt.title('Cross-Validation Scores\n(Optuna Best Parameters)',
             fontsize=12, fontweight='bold')
    plt.xlabel('Algorithm', fontsize=10)
    plt.ylabel('CV Score', fontsize=10)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    # 右：テストスコア
    plt.subplot(1, 2, 2)
    for i, algorithm in enumerate(results_df['Algorithm'].unique()):
        data = results_df[results_df['Algorithm'] == algorithm]
        plt.scatter([i] * len(data), data['Test_Score'],
                   label=f'Split Test Scores',
                   color='red', alpha=0.6, s=100)

        # 平均値と標準偏差を表示
        mean = data['Test_Score'].mean()
        std = data['Test_Score'].std()
        plt.errorbar(i, mean, yerr=std, color='black', capsize=5,
                    label='Mean ± Std' if i == 0 else '')

    plt.title('Test Scores by Split\nwith Mean ± Std',
             fontsize=12, fontweight='bold')
    plt.xlabel('Algorithm', fontsize=10)
    plt.ylabel('Test Score', fontsize=10)
    plt.xticks(range(len(results_df['Algorithm'].unique())),
               results_df['Algorithm'].unique(), rotation=45)
    plt.grid(True, alpha=0.3)
    if len(results_df['Algorithm'].unique()) > 0:  # 凡例は1回だけ表示
        plt.legend()

    plt.tight_layout()
    plt.savefig(f"{export_path}final_results.png", dpi=300, bbox_inches='tight')
    plt.close()

def print_summary(results_df):
    """結果のサマリーを表示"""
    if results_df.empty:
        print("\nNo results to summarize.")
        return

    print("\n=== Performance Summary ===")

    # スコアの要約
    summary = results_df.groupby('Algorithm').agg({
        'CV_Score': ['mean', 'std', 'min', 'max'],
        'Test_Score': ['mean', 'std', 'min', 'max']
    }).round(4)

    print("\nScores Summary:")
    print(summary)

    # パラメータの要約
    print("\nBest Parameters Summary:")
    for algorithm in results_df['Algorithm'].unique():
        print(f"\n{algorithm}:")
        algo_data = results_df[results_df['Algorithm'] == algorithm]
        param_cols = [col for col in algo_data.columns if col.startswith('param_')]
        for param in param_cols:
            param_name = param.replace('param_', '')
            param_values = algo_data[param].values
            if len(set(param_values)) == 1:
                print(f"  {param_name}: {param_values[0]} (constant)")
            else:
                print(f"  {param_name}: {param_values.mean():.4f} ± {param_values.std():.4f} "
                      f"(range: {min(param_values):.4f} - {max(param_values):.4f})")

if __name__ == "__main__":
    # データの読み込み
    df = pd.read_csv(import_path, header=0, index_col=0, encoding="sjis")

    # 説明変数から 'turbidity' と 'wavelength' を除外
    # 'species' と 'SampleID' は保持
    feature_columns = [col for col in df.columns if col not in ['species', 'turbidity', 'wavelength', 'SampleID']]
    df_features = df[['SampleID', 'wavelength'] + feature_columns + ['species']].copy()

    # 蛍光波長の列名を抽出（数値列のみ）
    emission_columns = [col for col in feature_columns if col.isdigit()]

    # ピボット操作
    # 各 SampleID が複数の 'wavelength'（励起波長）に対応し、それぞれの emission_columns（蛍光波長）に対して強度があると仮定
    # pivot_table を使用して SampleID をインデックス、wavelength を列、emission_columns の値を取得
    df_pivot = df_features.pivot_table(index='SampleID', columns='wavelength', values=emission_columns, aggfunc='mean')

    # カラム数と新しいカラム名の数を確認
    unique_excitation = df_pivot.columns.levels[0].nunique()
    num_emission = len(emission_columns)
    expected_columns = unique_excitation * num_emission
    actual_columns = len(df_pivot.columns)
    print(f"Unique excitation wavelengths: {unique_excitation}")
    print(f"Emission columns: {num_emission}")
    print(f"Expected number of columns after pivot: {expected_columns}")
    print(f"Actual number of columns after pivot: {actual_columns}")

    # MultiIndex のカラムをフラットにする
    # カラム名を 'exc_<励起波長>_em_<蛍光波長>' の形式に変換
    try:
        df_pivot.columns = [f"exc_{exc}_em_{em}" for exc, em in df_pivot.columns]
    except Exception as e:
        print(f"Error in flattening columns: {e}")
        print("df_pivot.columns:", df_pivot.columns)
        raise

    # 特徴量データフレームとターゲット変数を作成
    X_df = df_pivot.reset_index()
    X_df = X_df.drop('SampleID', axis=1)  # 'SampleID' を説明変数から除外
    X_df.fillna(X_df.mean(), inplace=True)  # 欠損値を平均値で補完
    y = df.groupby('SampleID')['species'].first().values  # ターゲット変数

    print("Dataset shape:", X_df.shape)
    print("Number of classes:", len(np.unique(y)))
    print("Classes:", np.unique(y))

    # 分類器の定義
    selected_classifiers = {
        'ExtraTreesClassifier': {'class': ExtraTreesClassifier, 'params': 'et'},
        'MLPClassifier': {'class': MLPClassifier, 'params': 'mlp'},
        'LogisticRegressionCV': {'class': LogisticRegressionCV, 'params': 'lr'}
    }

    # 評価の実行
    results_df = evaluate_classifiers(X_df, y, selected_classifiers)

    # 結果の可視化
    plot_final_results(results_df, export_path)

    # サマリーの表示と保存
    print_summary(results_df)
    results_df.to_csv(f"{export_path}final_results.csv", index=False)

    print(f"\nResults have been saved to {export_path}")