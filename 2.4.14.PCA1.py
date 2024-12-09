import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# CSVファイルの読み込み
file_path = 'EEM.csv'
df = pd.read_csv(file_path, header=0, index_col=0, encoding="sjis")

# 菌株名とラベル名の辞書定義
species_list_ = {
    'ACBA':'A. baumanii(19606)',
    'ESCO#8':'E. coli(8739)',
    'ESCO#2':'E. coli(25922)',
    'KLPN':'K. pneumoniae(13883)',
    'PRMI':'P. mirabilis(29245)',
    'PSAE#9':'P. aeruginosa(9027)',
    'PSAE#1':'P. aeruginosa(10145)',
    'PSAE#2':'P. aeruginosa(27853)',
    'PSAE#3':'P. aeruginosa(35422)',
    'SATY':'S. Typhimurium(14028)',
    'STAU':'S. aureus(29213)',
    'STHA':'S. haemolyticus(29970)',
    'STPN':'S. pneumoniae(46919)',
    'STPY':'S. pyogenes(19615)'
}

# 各菌種に異なる色を割り当てるためのカラーマップを取得
cmap = plt.colormaps['tab20']  # 'tab20'カラーマップを直接取得
species_keys = list(species_list_.keys())
num_species = len(species_keys)

# 'tab20'には20色あるため、インデックスを繰り返して色を割り当てる
# ここで species_colors をラベル名に基づいてマッピング
species_colors = {
    species_list_[species]: cmap(i % 20) for i, species in enumerate(species_keys)
}

# 各サンプルごとに、すべての波長でのスペクトルデータを結合
data_list = []
species_labels = []
turbidity_values = []

# 各サンプルのデータを収集し、波長ごとに横に結合
for (species, turbidity), group in df.groupby(['species', 'turbidity']):
    concatenated_data = group.drop(['species', 'turbidity', 'wavelength'], axis=1).values.flatten()
    data_list.append(concatenated_data)
    species_labels.append(species_list_.get(species, species))  # ラベル名を追加
    turbidity_values.append(turbidity)

# データの整形とPCA実行
data_matrix = np.array(data_list)
pca = PCA(n_components=1)  # コンポーネント数を1に設定
pca_result = pca.fit_transform(data_matrix)

# PC1の範囲の95パーセンタイルで表示範囲を設定
pc1_min, pc1_max = np.percentile(pca_result[:, 0], [10, 90])

# 濁度に基づいてプロットサイズをスケーリング（1から200の範囲）
size_min, size_max = 1, 200
turbidity_min, turbidity_max = min(turbidity_values), max(turbidity_values)
sizes = [
    size_min + (t - turbidity_min) / (turbidity_max - turbidity_min) * (size_max - size_min)
    for t in turbidity_values
]

# 表示範囲内に収まるデータのみをフィルタリング
filtered_pca_result = []
filtered_species_labels = []
filtered_sizes = []

for i in range(len(pca_result)):
    pc1 = pca_result[i, 0]
    if pc1_min <= pc1 <= pc1_max:
        filtered_pca_result.append(pc1)
        filtered_species_labels.append(species_labels[i])
        filtered_sizes.append(sizes[i])

filtered_pca_result = np.array(filtered_pca_result)

# 可視化
plt.figure(figsize=(8, 2.5))

# フィルタリングしたデータのみを1Dでプロット
for i, species in enumerate(filtered_species_labels):
    color = species_colors.get(species, "gray")  # ラベル名に基づいて色を取得
    plt.scatter(filtered_pca_result[i], 0, label=None, s=filtered_sizes[i], alpha=0.5, color=color)

# 軸ラベルとタイトル
plt.xlabel("PC1", fontsize=12, fontweight='bold')
plt.title("1D PCA plot", fontsize=12, fontweight='bold')

# 主要部分のみ表示
plt.xlim(pc1_min, pc1_max)
plt.gca().axes.get_yaxis().set_visible(False)  # y軸を非表示にする

# グラフのレイアウトを自動調整して保存
plt.tight_layout()
plt.savefig("/Users/mito/Desktop/1D_PCA_plot_tab20_color_scheme.tiff", dpi=400)
plt.show()