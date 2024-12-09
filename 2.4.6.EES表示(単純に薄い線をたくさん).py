import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
def SelectSpecies(species):
  _df = Selection(df,"species",species)
  return _df

#speciesで抽出してさらに濁度で抽出
def SelectSpeciesTurbidity(species,n_turbidity):
  _df = Selection(df,"species",species)
  turbidity_list = np.unique(_df["turbidity"])
  _df = Selection(_df,"turbidity",turbidity_list[n_turbidity])
  return _df

#dfを分離
def Dismantle(_df):
  excitation = _df["wavelength"].values
  _df = _df.drop(["species","turbidity","wavelength"],axis=1)
  emission = _df.columns.values.astype(int)
  _df.columns=emission
  _df.index=excitation
  return _df,emission,excitation

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
###########################[[[初期パラメータ]]]####################################
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
wl=250
species = "STAU"

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
###########################[[[実処理部]]]####################################
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def single_smaple():
  fig = plt.figure('',figsize=(5, 5))#subplot内に引数として「projection='3d'」を渡すと3Dグラフになる
  ax = fig.add_subplot(1,1,1)#(作成する行数、列数、何番目のグラフか)
  for i in range(10,11):
    try:
      data=SelectSpeciesTurbidity(species,i)
      data,emission,excitation=Dismantle(data)
      mappable=ax.plot(data.loc[wl],lw=1, c='k', alpha=1, linestyle = "-", label='label')
          #linestyleは'-(実線)'、'--(破線)'、':(細破線)'から選択
    except:pass
  plt.xlabel(r"$\mathbf{λ_{em}}$ $(nm)$",fontsize=12)
  plt.ylabel(r"$\bf{signal \;\;intensity}$",fontsize=12)
  plt.title("S. aureus", fontsize=14, fontstyle='italic', fontfamily='Times New Roman', fontweight='bold')
  ax.text(0.99, 0.99, r"$\mathbf{λ_{ex}}$ = 250nm", va='top', ha='right', transform=ax.transAxes)
  ax.text(0.99, 0.95, r"$\bf{turbidity}$ = 0.25", va='top', ha='right', transform=ax.transAxes)

  plt.tight_layout()
  plt.savefig("/Users/mito/Desktop/untitled.tiff",dpi=400)
  plt.show()

def multi_smaple():
  fig = plt.figure('',figsize=(5, 5))#subplot内に引数として「projection='3d'」を渡すと3Dグラフになる
  ax = fig.add_subplot(1,1,1)#(作成する行数、列数、何番目のグラフか)
  for i in range(45):
    try:
      data=SelectSpeciesTurbidity(species,i)
      data,emission,excitation=Dismantle(data)
      mappable=ax.plot(data.loc[wl],lw=1, c='k', alpha=(i*i)/(45*45), linestyle = "-")
          #linestyleは'-(実線)'、'--(破線)'、':(細破線)'から選択
    except:pass

# ダミーのプロットを作成して、透明度に応じた凡例を表示
  for alpha_value in [0.1 * i for i in range(1, 10)]:
      ax.plot([], [], lw=1.5, c='k', alpha=alpha_value, linestyle="-", label=f"turbidity = {alpha_value:.1f}")
  ax.legend(loc='upper right', bbox_to_anchor=(0.98,0.85))

  plt.xlabel(r"$\mathbf{λ_{em}}$ $(nm)$",fontsize=12)
  plt.ylabel(r"$\bf{signal \;\;intensity}$",fontsize=12)
  plt.title("S. aureus", fontsize=14, fontstyle='italic', fontfamily='Times New Roman', fontweight='bold')
  ax.text(0.99, 0.99, r"$\mathbf{λ_{ex}}$ = 250nm", va='top', ha='right', transform=ax.transAxes)
  ax.text(0.99, 0.95, r"$\bf{turbidity}$ : 0.05~0.90", va='top', ha='right', transform=ax.transAxes)

  plt.tight_layout()
  plt.savefig("/Users/mito/Desktop/untitled.tiff",dpi=400)
  plt.show()

#single_smaple()
multi_smaple()