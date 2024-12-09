import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage
from matplotlib.ticker import MaxNLocator, MultipleLocator, AutoMinorLocator
import matplotlib.colors
import matplotlib.cm as cm

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

def PlotRowSumWithDoubleThreshold(dataframe, row_start, row_end, threshold_low, threshold_high):
    """
    指定された範囲の行（row_startからrow_end）の合計をプロットし、
    threshold_low未満およびthreshold_highより大きい列を0にした合計も同時にプロットする関数。
    
    Parameters:
    - dataframe: プロット対象のPandas DataFrame
    - row_start: 開始行名（整数）
    - row_end: 終了行名（整数）
    - threshold_low: 低い閾値の列名（整数）。この値未満の列のデータを0にする。
    - threshold_high: 高い閾値の列名（整数）。この値より大きい列のデータを0にする。
    """
    # 指定範囲の行が存在するか確認
    available_rows = dataframe.loc[dataframe.index.isin(range(row_start, row_end + 1))]
    if available_rows.empty:
        print(f"No rows found in the range {row_start} to {row_end}.")
        return
    
    # 行の合計を計算（元のデータ）
    summed_original = available_rows.sum()
    
    # 行の合計を計算（列名 < threshold_low および > threshold_high を0に設定）
    summed_modified = available_rows.copy()
    summed_modified.loc[:, summed_modified.columns < threshold_low] = 0
    summed_modified.loc[:, summed_modified.columns > threshold_high] = 0
    summed_modified = summed_modified.sum()
    
    # プロットの設定
    plt.figure(figsize=(12, 7))
    
    # 元の合計データを点線でプロット
    plt.plot(summed_original.index, summed_original.values, lw=2, color='blue', linestyle='--', label=f'Summed {row_start}-{row_end}nm (Original)')
    
    # 修正後の合計データを実線でプロット
    plt.plot(summed_modified.index, summed_modified.values, lw=2, color='red', linestyle='-', label=f'Summed {row_start}-{row_end}nm (columns < {threshold_low} or > {threshold_high} set to 0)')
    
    plt.xlabel('Emission Wavelength (nm)', fontsize=14)
    plt.ylabel('Summed Intensity', fontsize=14)
    plt.title(f'Summed EEM from Excitation {row_start} nm to {row_end} nm', fontsize=16)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(f"/Users/mito/Desktop/EEM_Summed_{row_start}_to_{row_end}nm.png", dpi=300)
    plt.show()

# CSVファイルのパスを指定
csv = '/Users/mito/Library/Mobile Documents/com~apple~CloudDocs/python/EEMdata/機械の基礎データ/条件変更用/バンド幅を変えてみる/STAU0.07.V400.X2.M2.BX5.BM10.csv'

filter = Mean

# データフレームの作成
EEM = CSVtoDataFrame(csv)  # 1csvをdataframe化
EEM = filter(EEM)
print(EEM)

# 行名260から290までのデータの合計をプロットし、
# 列名320未満および480より大きいデータを0にしたデータもプロット
PlotRowSumWithDoubleThreshold(EEM, row_start=254, row_end=268, threshold_low=290, threshold_high=480)