import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage
from matplotlib.ticker import MaxNLocator,MultipleLocator,AutoMinorLocator
import matplotlib.colors
import matplotlib.cm as cm
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
print(species_list)

df,emission,excitation = select(1,20)

fig = plt.figure('ウィンドウタイトル',figsize=(8, 6))#subplot内に引数として「projection='3d'」を渡すと3Dグラフになる
ax = fig.add_subplot(1,1,1)#(作成する行数、列数、何番目のグラフか)
mappable = ax.pcolor(emission, excitation,df, cmap='nipy_spectral',)
fig.colorbar(mappable, ax=ax,aspect=50)
ax.yaxis.set_major_locator(MaxNLocator(nbins=10,min_n_ticks=5))
ax.xaxis.set_major_locator(MaxNLocator(nbins=10,min_n_ticks=5))
plt.xlabel(r"$\mathbf{λ_{em}}$ $\bf[nm]$",fontsize=14)
plt.ylabel(r"$\mathbf{λ_{ex}}$ $\bf[nm]$",fontsize=14)
#plt.title(samplename, weight='bold')
#  ax.set_xlim(250,300)
#  ax.set_ylim(200,250)
plt.show()
