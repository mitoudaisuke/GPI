import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
###########################[[[初期設定部分]]]####################################
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
file_path = 'EEM.csv'#通常ver
df = pd.read_csv(file_path, header=0, index_col=0, encoding = "sjis",)
species_list = np.unique(df["species"])
species_list_ = {'ACBA':'ACBA(19606)','ESCO#8':'ESCO(8739)','ESCO#2':'ESCO(25922)',
                 'KLPN':'KLPN(13883)','PRMI':'PRMI(29245)','PSAE#9':'PSAE(9027)',
                 'PSAE#1':'PSAE(10145)','PSAE#2':'PSAE(27853)','PSAE#3':'PSAE(35422)',
                 'SATY':'SATY(14028)','STAU':'STAU(29213)','STHA':'STHA(29970)',
                 'STPN':'STPN(46919)','STPY':'STPY(19615)'}

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
###########################[[[関数の定義]]]####################################
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def Selection(dataframe,columns,word):
  dataframe = dataframe[dataframe[columns]==word]
  return dataframe

#speciesで抽出してさらに濁度で抽出
def SelectSpecies(dataframe,species):
  _df = Selection(dataframe,"species",species)
  return _df

#speciesで抽出してさらに波長で抽出
def SelectSpeciesWL(dataframe,species,wl):
  _df = Selection(dataframe,"species",species)
  _df=Selection(_df,"wavelength",wl)
  return _df

#speciesで抽出してさらに濁度で抽出
def SelectSpeciesTurbidity(dataframe,species,n_turbidity):
  _df = Selection(dataframe,"species",species)
  turbidity_list = np.unique(_df["turbidity"])
  _df = Selection(_df,"turbidity",turbidity_list[n_turbidity])
  return _df

#dfを分離
def Dismantle(dataframe):
  excitation = dataframe["wavelength"].values
  dataframe = dataframe.drop(["species","turbidity","wavelength"],axis=1)
  emission = dataframe.columns.values.astype(int)
  dataframe.columns=emission
  dataframe.index=excitation
  return dataframe,emission,excitation

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
###########################[[[初期パラメータ]]]####################################
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
wl=250
species = "STAU"

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
###########################[[[実処理部]]]####################################
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
data=SelectSpeciesWL(df,species,wl)
data,emission,excitation=Dismantle(data)

#data=data/data.max().max()

fig = plt.figure('',figsize=(5, 5))#subplot内に引数として「projection='3d'」を渡すと3Dグラフになる
ax = fig.add_subplot(1,1,1)#(作成する行数、列数、何番目のグラフか)

color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
color = next(color_cycle)
ax.plot(data.mean(),lw=1, c='black', alpha=1, linestyle = "-", label='mean')
ax.fill_between(emission, data.max(), data.min(), color='black', alpha=0.2, label='dispersion')

plt.xlabel(r"$\mathbf{λ_{em}}$ $(nm)$",fontsize=12)
plt.ylabel(r"$\bf{normalized \;\; signal \;\;intensity}$",fontsize=12)
plt.title("S. aureus", fontsize=14, fontstyle='italic', fontfamily='Times New Roman', fontweight='bold')
ax.text(0.99, 0.99, r"$\mathbf{λ_{ex}}$ = 250nm", va='top', ha='right', transform=ax.transAxes)
ax.text(0.99, 0.95, r"$\bf{turbidity}$ : 0.05~0.90", va='top', ha='right', transform=ax.transAxes)
plt.legend(bbox_to_anchor=(1, 0.9), loc='upper right',)
plt.tight_layout()
plt.savefig("/Users/mito/Desktop/untitled.tiff",dpi=400)
plt.show()
