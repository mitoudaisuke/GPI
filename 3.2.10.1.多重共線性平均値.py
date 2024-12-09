# 環境に応じたパス設定
import_path='/Users/mito/Library/Mobile Documents/com~apple~CloudDocs/python/EEM.csv'
export_path ='/Users/mito/Desktop/'
directory='/Users/mito/Library/Mobile Documents/com~apple~CloudDocs/python/手動MCS/'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy import stats

# データの読み込み
df = pd.read_csv(import_path, header=0, index_col=0, encoding="sjis")
# 特定の菌種を選択
species = 'STAU'
species_df = df[df["species"] == species]
def graph(process):
    title = process

def prepare_eem_data(df, species, target_turbidity=None):
    # 特定の菌種のデータを抽出
    if target_turbidity is None:
        filtered_df = df[df["species"] == species]
    else:
        filtered_df = df[(df["species"] == species) & (df["turbidity"] == target_turbidity)]

    # データが空でないか確認
    if filtered_df.empty:
        raise ValueError(f"No data found for species {species}")

    # ユニークな濁度値を取得
    turbidities = filtered_df["turbidity"].unique()

    # 各濁度でのデータを格納するリスト
    all_data = []

    # 各濁度のデータを処理
    for turb in turbidities:
        turb_df = filtered_df[filtered_df["turbidity"] == turb]

        # データを分解
        excitation = turb_df["wavelength"].values
        turb_df = turb_df.drop(columns=["species", "turbidity", "wavelength"])
        emission = turb_df.columns.values.astype(int)
        turb_df.columns = emission
        turb_df.index = excitation

        all_data.append(turb_df)

    return all_data, emission, excitation, turbidities

def analyze_collinearity(data_matrices):
    # 各濁度での相関行列を計算
    correlation_matrices = []

    for data_matrix in data_matrices:
        corr_matrix = np.corrcoef(data_matrix)
        correlation_matrices.append(corr_matrix)

    # 平均相関行列を計算
    avg_correlation = np.mean(correlation_matrices, axis=0)
    std_correlation = np.std(correlation_matrices, axis=0)

    return avg_correlation, std_correlation
# ... [前のインポートと設定部分は同じ]

def plot_collinearity(correlation_matrix, title, wavelengths):
    fig, ax = plt.subplots(figsize=(5, 5))
    
    # ヒートマップの作成
    im = ax.imshow(correlation_matrix, 
                   cmap='PuOr',
                   aspect='equal',
                   vmin=-1,
                   vmax=1,
                   origin='lower')
    
    # 軸ラベルの設定（50刻み）
    step = 50
    tick_indices = np.where(wavelengths % step == 0)[0]
    tick_values = wavelengths[tick_indices]
    
    # 軸の設定とフォントサイズ調整
    ax.set_xticks(tick_indices)
    ax.set_yticks(tick_indices)
    ax.set_xticklabels(tick_values, rotation=45, fontsize=10)
    ax.set_yticklabels(tick_values, fontsize=10)
    
    # タイトルと軸ラベルを条件分岐で設定
    if 'Excitation' in title:
        plt.title(r'$\mathbf{\lambda_{ex}}$ Collinearity', 
                 fontweight='bold', fontsize=14)
        ax.set_xlabel(r'$\mathbf{\lambda_{ex}}$ (nm)', fontweight='bold', fontsize=12)
        ax.set_ylabel(r'$\mathbf{\lambda_{ex}}$ (nm)', fontweight='bold', fontsize=12)
    else:
        plt.title(r'$\mathbf{\lambda_{em}}$ Collinearity', 
                 fontweight='bold', fontsize=12)
        ax.set_xlabel(r'$\mathbf{\lambda_{em}}$ (nm)', fontweight='bold', fontsize=12)
        ax.set_ylabel(r'$\mathbf{\lambda_{em}}$ (nm)', fontweight='bold', fontsize=12)
    
    # カラーバー
    cbar = plt.colorbar(im, 
                       label='Average Correlation Coefficient',
                       shrink=0.7)
    
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label('Average Correlation Coefficient', fontweight='bold', fontsize=12)
    
    # レイアウトの調整
    plt.tight_layout()
    
    return fig

# メイン処理部分は同じ
def main():
    try:
        # データの準備
        all_data, emission, excitation, turbidities = prepare_eem_data(df, species)
        
        print(f"分析対象の濁度: {turbidities}")
        
        # 励起波長の相関分析
        excitation_data = [df.values for df in all_data]
        avg_excitation_correlation, std_excitation_correlation = analyze_collinearity(excitation_data)
        
        # 蛍光波長の相関分析
        emission_data = [df.values.T for df in all_data]
        avg_emission_correlation, std_emission_correlation = analyze_collinearity(emission_data)
        
        # プロット作成と保存
        fig1 = plot_collinearity(avg_excitation_correlation, 'Excitation', excitation)
        plt.savefig("/Users/mito/Desktop/excitation_collinearity.tiff", 
                    dpi=400, bbox_inches='tight', format='tiff')
        plt.close(fig1)
        
        fig2 = plot_collinearity(avg_emission_correlation, 'Emission', emission)
        plt.savefig("/Users/mito/Desktop/emission_collinearity.tiff", 
                    dpi=400, bbox_inches='tight', format='tiff')
        plt.close(fig2)
        
        print("画像の保存が完了しました。")
        
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")
        
if __name__ == "__main__":
    main()