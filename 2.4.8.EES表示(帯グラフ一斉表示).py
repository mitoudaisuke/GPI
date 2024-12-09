import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle
import matplotlib.ticker as ticker
color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
###########################[[[初期設定部分]]]####################################
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
file_path = 'EEM.csv'#通常ver
df = pd.read_csv(file_path, header=0, index_col=0, encoding = "sjis",)
species_list = np.unique(df["species"])
species_list_ = {'ACBA':'A. baumanii(19606)','ESCO#8':'E. coli(8739)','ESCO#2':'E. coli(25922)',
                 'KLPN':'K. pneumoniae(13883)','PRMI':'P. mirabilis(29245)','PSAE#9':'P. aeruginosa(9027)',
                 'PSAE#1':'P. aeruginosa(10145)','PSAE#2':'P. aeruginosa(27853)','PSAE#3':'P. aeruginosa(35422)',
                 'SATY':'S. Typhimurium(14028)','STAU':'S. aureus(29213)','STHA':'S. haemolyticus(29970)',
                 'STPN':'S. pneumoniae(46919)','STPY':'S. pyogenes(19615)'}

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

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
###########################[[[実処理部]]]####################################
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


def plot_with_ylabel(wl):
  fig = plt.figure('',figsize=(5.1, 8))#subplot内に引数として「projection='3d'」を渡すと3Dグラフになる
  ax = fig.add_subplot(1,1,1)#(作成する行数、列数、何番目のグラフか)
  for i in range(14):
      data=SelectSpeciesWL(df,species_list[i],wl)
      data,emission,excitation=Dismantle(data)
      data=data/data.max().max()
      data = data*0.9
      data=data-i
      color = next(color_cycle)
      ax.plot(data.mean(),lw=1, c=color, alpha=1, linestyle = "-", label=species_list_[species_list[i]])
      ax.fill_between(emission, data.max(), data.min(), color=color, alpha=0.2)
  ax.set_xlim(wl,wl*2)
  ax.set_yticks(range(0,-14,-1))  # y軸の目盛り位置を0から10まで設定
  yticks_labels = []
  for species in species_list[:14]:
      label = species_list_[species]
      if "(" in label:
          italic_part, normal_part = label.split("(")
  #        yticks_labels.append(f"${{\\mathit{{{italic_part}}}}}$\n({normal_part}")
          yticks_labels.append(f"\n({normal_part}")
      else:
          yticks_labels.append(f"${{\\mathit{{{label}}}}}$")

  ax.set_yticklabels(yticks_labels)
  plt.xlabel(r"$\mathbf{λ_{em}}$ $(nm)$",fontsize=12)
  #plt.title("Mean fluorescence spectrum of each strains", fontsize=12, fontweight='bold', loc='left', x=-0.25)
  ax.text(0.99, 0.99, rf"$\mathbf{{\lambda_{{ex}}}}$ = {wl}nm", va='top', ha='right', transform=ax.transAxes)

  ax.set_xlim(wl,min(wl*2,700))
  #plt.ylabel(r"$\bf{normalized \;\; signal \;\;intensity}$",fontsize=12)
  # y軸の菌種名ラベルを ax.text で配置
  for idx, species in enumerate(species_list[:14]):
      label = species_list_[species]
      italic_part, normal_part = label.split("(")
      # italic部分をボールドイタリックで直接指定
      ax.text(wl * 0.975, -idx + 0.1, f"{italic_part}", 
              ha='right', va='center', fontsize=12, fontfamily='Times New Roman', fontstyle='italic', fontweight='bold')


def plot_without_ylabel(wl):
  fig = plt.figure('',figsize=(4, 8))#subplot内に引数として「projection='3d'」を渡すと3Dグラフになる
  ax = fig.add_subplot(1,1,1)#(作成する行数、列数、何番目のグラフか)
  for i in range(14):
      data=SelectSpeciesWL(df,species_list[i],wl)
      data,emission,excitation=Dismantle(data)
      data=data/data.max().max()
      data=data-i
      color = next(color_cycle)
      ax.plot(data.mean(),lw=1, c=color, alpha=1, linestyle = "-", label=species_list_[species_list[i]])
      ax.fill_between(emission, data.max(), data.min(), color=color, alpha=0.2)
  ax.set_xlim(wl,wl*2)
  ax.set_yticks(range(0,-14,-1))  # y軸の目盛り位置を0から10まで設定
  ax.set_yticklabels([])
  plt.xlabel(r"$\mathbf{λ_{em}}$ $(nm)$",fontsize=12)
  #plt.title("Mean fluorescence spectrum of each strains", fontsize=12, fontweight='bold', loc='left', x=-0.25)
  ax.text(0.99, 0.99, rf"$\mathbf{{\lambda_{{ex}}}}$ = {wl}nm", va='top', ha='right', transform=ax.transAxes)
  ax.set_xlim(wl,min(wl*2,700))

wl=350
#plot_with_ylabel(wl)
plot_without_ylabel(wl)

plt.tight_layout()
plt.savefig("/Users/mito/Desktop/untitled.tiff",dpi=400)
plt.show()
