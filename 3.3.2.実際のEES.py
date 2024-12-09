import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage
from matplotlib.ticker import MaxNLocator, MultipleLocator, AutoMinorLocator
import matplotlib.colors
import matplotlib.cm as cm

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
###########################[[[関数の定義]]]####################################
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def FileName(filename):
    filename = filename.split('/')  # /で名前を区切り
    filename = filename[-1].split('.')  # ファイル名と拡張子に分割
    return filename  

def CSVtoDataFrame(filename):
    dataframe = pd.read_csv(filename, header=17, index_col=0, encoding="sjis")
    dataframe.index = dataframe.index.astype(int)
    dataframe.columns = dataframe.columns.astype(int)
    dataframe = dataframe.T
    dataframe = dataframe.fillna(0)
    return dataframe

def Graph_minmax(dataframe, filter):
    if filter in ['SobelX', 'SobelXX', 'Laplacian', 'LaplacianGaussian']:
        min_val, max_val, n_round = 0.05, 0.095, 0
    else:
        min_val, max_val, n_round = 0.2, 0.9, 2
    valuemin = round(dataframe.quantile(min_val).quantile(min_val), n_round)
    valuemax = round(dataframe.quantile(max_val).quantile(max_val), n_round)
    return valuemin, valuemax

def Heatmap2D(dataframe, csv, filter):
    fig = plt.figure('ウィンドウタイトル', figsize=(6, 5))
    ax = fig.add_subplot(1, 1, 1)
    vmin, vmax = Graph_minmax(dataframe, filter)
    mappable = ax.pcolor(dataframe.columns, dataframe.index, dataframe + 1, cmap='nipy_spectral', 
                        norm=matplotlib.colors.LogNorm(vmin=3, vmax=15000))  # 対数表示
    # mappable = ax.pcolor(dataframe.columns, dataframe.index, dataframe, cmap='nipy_spectral')  # 通常表示
    cbar = fig.colorbar(mappable, ax=ax, aspect=50)
    cbar.set_label("Intensity", fontsize=12, fontweight='bold')
    # ax.yaxis.set_major_locator(MaxNLocator(nbins=10, min_n_ticks=5))
    # ax.xaxis.set_major_locator(MaxNLocator(nbins=10, min_n_ticks=5))
    plt.xlabel(r"$\mathbf{λ_{em}}$ $(nm)$", fontsize=12)
    plt.ylabel(r"$\mathbf{λ_{ex}}$ $(nm)$", fontsize=12)
    plt.title("S. aureus", fontsize=14, fontstyle='italic', fontfamily='Times New Roman', fontweight='bold')
    plt.savefig("/Users/mito/Desktop/untitled.tiff", dpi=400)
    plt.show()

def Mean(dataframe):
    kernel = np.full((5, 5), 1/25)  # 平均フィルタ
    array = ndimage.convolve(dataframe, kernel)
    dataframe = pd.DataFrame(array, index=dataframe.index, columns=dataframe.columns)
    return dataframe

def PlotRow(dataframe, row_name):
    if row_name in dataframe.index:
        plt.figure(figsize=(8, 5))
        plt.plot(dataframe.columns, dataframe.loc[row_name], lw=1)
        plt.xlabel('Emission Wavelength (nm)', fontsize=12)
        plt.ylabel('Intensity', fontsize=12)
        plt.title(f'EEM at Excitation {row_name} nm', fontsize=14)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(f"/Users/mito/Desktop/EEM_{row_name}nm.png", dpi=300)
        plt.show()
    else:
        print(f"Row {row_name} not found in the dataframe.")

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
###########################[[[実処理]]]####################################
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# CSVファイルのパスを指定
csv='/Users/mito/Library/Mobile Documents/com~apple~CloudDocs/python/EEMdata/機械の基礎データ/条件変更用/バンド幅を変えてみる/STAU0.07.V400.X2.M2.BX5.BM10.csv'
csv = '/Users/mito/Library/Mobile Documents/com~apple~CloudDocs/python/EEMdata/機械の基礎データ/条件変更用/バンド幅を変えてみる/STAU0.07.V300.X2.M2.BX20.BM10.csv'  # フィルター全部なし

filter = Mean

# データフレームの作成
EEM = CSVtoDataFrame(csv)  # 1csvをdataframe化
EEM = filter(EEM)
print(EEM)

# 行名が280のデータをプロット
PlotRow(EEM, 320)