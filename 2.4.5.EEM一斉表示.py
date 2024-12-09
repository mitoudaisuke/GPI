import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
from scipy import ndimage
from matplotlib.ticker import MaxNLocator,MultipleLocator,AutoMinorLocator

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
###########################[[[関数の定義]]]####################################
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def Graph_minmax(dataframe):
  min,max,n_round = 0.2,0.9,2
  valuemin = round(dataframe.quantile(min).quantile(min),n_round) #このn_roundがroundでround()とかぶってたら value_errorとなった
  valuemax = round(dataframe.quantile(max).quantile(max),n_round)  
  return valuemin, valuemax

def selection(dataframe,columns,word):
  dataframe = dataframe[dataframe[columns]==word]
  return dataframe

def select(n_species,n_turbidity):
  _df = selection(df,"species",species_list[n_species])
  turbidity_list = np.unique(_df["turbidity"])
#  print(turbidity_list)
  _df = selection(_df,"turbidity",turbidity_list[n_turbidity])
  excitation = _df["wavelength"].values
  _df = _df.drop(["species","turbidity","wavelength"],axis=1)
  emission = _df.columns.values.astype(int)
#  vmin,vmax = Graph_minmax(df)
#  print(vmin,vmax)
  return _df,emission,excitation

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
###########################[[[実処理]]]####################################
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
file_path = 'EEM.csv'#通常ver
df = pd.read_csv(file_path, header=0, index_col=0, encoding = "sjis",)
species_list = np.unique(df["species"])
species_list_ = {'ACBA':'A. baumanii(19606)','ESCO#8':'E. coli(8739)','ESCO#2':'E. coli(25922)',
                 'KLPN':'K. pneumoniae(13883)','PRMI':'P. mirabilis(29245)','PSAE#9':'P. aeruginosa(9027)',
                 'PSAE#1':'P. aeruginosa(10145)','PSAE#2':'P. aeruginosa(27853)','PSAE#3':'P. aeruginosa(35422)',
                 'SATY':'S. Typhimurium(14028)','STAU':'S. aureus(29213)','STHA':'S. haemolyticus(29970)',
                 'STPN':'S. pneumoniae(46919)','STPY':'S. pyogenes(19615)'}

fig = plt.figure('ウィンドウタイトル',figsize=(10, 7))#

def plot(type):
  for i in range(len(species_list)):
      dataframe, emission, excitation = select(i, 30)
      ax = fig.add_subplot(3, 5, i + 1)
      if type=="normal": mappable = ax.pcolor(dataframe.columns, dataframe.index, dataframe + 1, cmap='nipy_spectral')  # 対数表示
      if type=="log": mappable = ax.pcolor(dataframe.columns, dataframe.index, dataframe + 1, cmap='nipy_spectral', norm=matplotlib.colors.LogNorm(vmin=3, vmax=15000))  # 対数表示

      # タイトル設定：菌株名をitalic、番号を通常フォントで表示
      species_name = species_list_[species_list[i]].split('(')
      strain_name = species_name[0]  # 菌株名部分
      strain_number = f"({species_name[1]}"  # 番号部分

      ax.text(0.5, 1.075, strain_name, transform=ax.transAxes, ha='center', va='bottom',
              fontsize=13, fontstyle='italic', fontfamily='Times New Roman', weight='heavy')
      ax.text(0.5, 1.075, strain_number, transform=ax.transAxes, ha='center', va='top',
              fontsize=11, fontstyle='normal')  # 通常フォントで表示

      # 軸のフォーマットを非表示
      ax.xaxis.set_major_formatter(plt.NullFormatter())
      ax.yaxis.set_major_formatter(plt.NullFormatter())

  # カラーマップを右端に追加
  cbar_ax = fig.add_axes([0.875, 0.025, 0.01, 0.265])  # 位置を調整して右端に配置
  if type=="normal":
    cb = fig.colorbar(mappable, cax=cbar_ax, aspect=20)
    cb.set_label("Intensity", weight='bold')  # カラーバーのラベル
    cb.update_ticks()  # カラーバーの目盛りを更新
  if type=="log":
    cb = fig.colorbar(mappable, cax=cbar_ax, aspect=20)
    cb.set_label("Intensity (Log Scale)", weight='bold')  # カラーバーのラベル
    from matplotlib.ticker import MaxNLocator, LogFormatter
    cb.formatter = LogFormatter(10, labelOnlyBase=False)  # 対数フォーマット
    cb.update_ticks()  # カラーバーの目盛りを更新
  plt.tight_layout()
  plt.savefig("/Users/mito/Desktop/untitled.tiff", dpi=400)
  plt.show()

plot("normal")

