import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3Dプロット用
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

# 各サンプルごとに、すべての波長でのスペクトルデータを結合
data_list = []
species_labels = []
turbidity_values = []

# 各サンプルのデータを収集し、波長ごとに横に結合
for (species, turbidity), group in df.groupby(['species', 'turbidity']):
    concatenated_data = group.drop(['species', 'turbidity', 'wavelength'], axis=1).values.flatten()
    data_list.append(concatenated_data)
    species_labels.append(species_list_.get(species, species))
    turbidity_values.append(turbidity)

# データの整形とPCA実行
data_matrix = np.array(data_list)
pca = PCA(n_components=3)  # コンポーネント数を3に設定
pca_result = pca.fit_transform(data_matrix)

# PC1、PC2、PC3の範囲の95パーセンタイルで表示範囲を設定
pc1_min, pc1_max = np.percentile(pca_result[:, 0], [10, 90])
pc2_min, pc2_max = np.percentile(pca_result[:, 1], [10, 90])
pc3_min, pc3_max = np.percentile(pca_result[:, 2], [10, 90])

# 濁度に基づいてプロットサイズをスケーリング（1から200の範囲）
size_min, size_max = 1, 200
turbidity_min, turbidity_max = min(turbidity_values), max(turbidity_values)
sizes = [size_min + (t - turbidity_min) / (turbidity_max - turbidity_min) * (size_max - size_min) for t in turbidity_values]

# 表示範囲内に収まるデータのみをフィルタリング
filtered_pca_result = []
filtered_species_labels = []
filtered_sizes = []

for i in range(len(pca_result)):
    pc1, pc2, pc3 = pca_result[i, 0], pca_result[i, 1], pca_result[i, 2]
    if pc1_min <= pc1 <= pc1_max and pc2_min <= pc2 <= pc2_max and pc3_min <= pc3 <= pc3_max:
        filtered_pca_result.append([pc1, pc2, pc3])
        filtered_species_labels.append(species_labels[i])
        filtered_sizes.append(sizes[i])

filtered_pca_result = np.array(filtered_pca_result)

# 可視化
fig = plt.figure(figsize=(8, 6))
fig.patch.set_facecolor("white")  # 背景を白に設定
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor("white")  # 3Dプロット領域の背景も白に設定

# 各座標平面の背景色を白に設定
ax.xaxis.pane.fill = True
ax.yaxis.pane.fill = True
ax.zaxis.pane.fill = True
ax.xaxis.pane.set_facecolor("white")
ax.yaxis.pane.set_facecolor("white")
ax.zaxis.pane.set_facecolor("white")

# フィルタリングしたデータのみを3Dでプロット
for i, species in enumerate(filtered_species_labels):
    ax.scatter(filtered_pca_result[i, 0], filtered_pca_result[i, 1], filtered_pca_result[i, 2],
               label=None, s=filtered_sizes[i], alpha=0.5)

# 軸ラベルとタイトル
ax.set_xlabel("PC1", fontsize=12, fontweight='bold', labelpad=15)
ax.set_ylabel("PC2", fontsize=12, fontweight='bold', labelpad=15)
ax.set_zlabel("PC3", fontsize=12, fontweight='bold', labelpad=15)
plt.title("3D PCA plot", fontsize=12, fontweight='bold', pad=-20)  # タイトル位置を調整

# 主要部分のみ表示
ax.set_xlim(pc1_min, pc1_max)
ax.set_ylim(pc2_min, pc2_max)
ax.set_zlim(pc3_min, pc3_max)

# グラフのレイアウトを自動調整して保存
plt.tight_layout()
plt.savefig("/Users/mito/Desktop/3D_PCA_plot_concatenated_limited_filtered.tiff", dpi=400)
plt.show()