# 必要なライブラリのインポート
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import sys
import_path='/Users/mito/Library/Mobile Documents/com~apple~CloudDocs/python/EEM.csv'
export_path ='/Users/mito/Desktop/'
directory='/Users/mito/Library/Mobile Documents/com~apple~CloudDocs/python/手動MCS/'

# データの読み込みとフィルタリングは既存のコードを使用
df = pd.read_csv(import_path, header=0, index_col=0, encoding="sjis")
species = 'STAU'

def prepare_eem_data_for_pca(df, species):
    # 特定の菌種のデータを抽出
    species_df = df[df['species'] == species]
    
    # ユニークな濁度値を取得
    turbidities = species_df['turbidity'].unique()
    
    # データを格納するリスト
    data_list = []
    sample_ids = []
    
    # 各濁度のデータを処理
    for turbidity in turbidities:
        turb_df = species_df[species_df['turbidity'] == turbidity]
        excitation = turb_df['wavelength'].values  # 励起波長
        turb_df = turb_df.drop(columns=['species', 'turbidity', 'wavelength'])
        emission = turb_df.columns.values.astype(int)
        turb_df.columns = emission
        turb_df.index = excitation
        
        # EEMマトリックスを取得
        eem_matrix = turb_df.values
        
        # マトリックスを1次元にフラット化
        eem_vector = eem_matrix.flatten()
        
        data_list.append(eem_vector)
        sample_ids.append(turbidity)
    
    # データリストを2次元配列に変換
    data_matrix = np.array(data_list)
    
    return data_matrix, sample_ids

def main():
    try:
        # データの準備
        data_matrix, sample_ids = prepare_eem_data_for_pca(df, species)
        
        # 標準化（オプション）
        scaler = StandardScaler()
        data_matrix_scaled = scaler.fit_transform(data_matrix)
        
        # PCAの適用
        n_components = 20
        pca = PCA(n_components=n_components)
        pca_data = pca.fit_transform(data_matrix_scaled)
        
        # PCA成分間の相関行列を計算
        correlation_matrix = np.corrcoef(pca_data, rowvar=False)
        
        # ヒートマップの作成
        fig, ax = plt.subplots(figsize=(5, 5))
        cax = ax.matshow(correlation_matrix, cmap='PuOr', vmin=-1, vmax=1)
        
        # カラーバーの追加（短く設定）
        cbar = fig.colorbar(cax, shrink=0.7)  # カラーバーの長さを65%に調整
        cbar.set_label('Average Correlation Coefficient', fontsize=12, fontweight='bold')
        
        # 軸の設定（2から開始し、間隔を2ずつに設定）
        tick_positions = np.arange(1, n_components, 2)  # 2から開始
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
        ax.set_xticklabels(tick_positions + 1, fontsize=10)  # ラベルは+1して2から表示
        ax.set_yticklabels(tick_positions + 1, fontsize=10)  # ラベルは+1して2から表示
        plt.xticks(rotation=45)
        
        # xtickラベルを下に配置
        ax.xaxis.tick_bottom()
        ax.tick_params(bottom=True, top=False)
        
        # y軸を上下反転
        ax.invert_yaxis()
        
        # タイトルとラベルの設定
        plt.title('PCA components Collinearity', fontweight='bold', fontsize=14)
        ax.set_xlabel('PC', fontweight='bold', fontsize=12)
        ax.set_ylabel('PC', fontweight='bold', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(export_path + "pca_collinearity.tiff", dpi=400, bbox_inches='tight', format='tiff')
        plt.show()
        
        print("PCA後の多重共線性のヒートマップを保存しました。")
        
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")

if __name__ == "__main__":
    main()