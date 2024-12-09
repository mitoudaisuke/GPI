import serial
import struct
import matplotlib.pyplot as plt
import numpy as np
import time


ser = serial.Serial('/dev/cu.usbmodem3495325C35381')
ser.reset_input_buffer()
CDC_number = 5000
n_zeros = 50 #とりあえず判定用のデータブロック。100個の0をデータ末尾につけているので、50個ずつ読み取っていれば必ずどこかのグループは全部0になるはず
window = 100 #移動平均のウィンドウ幅
w = np.ones(window)/window

def plot():
    binarydata = ser.read(2 * CDC_number)#2バイトごとにCDCを読み込み、STM32側のbuffer数ずつにグループ
    rawdata = struct.unpack('>' + 'H' * CDC_number, binarydata)#読み取ったやつを16進数に変換
    rawdata = np.array(rawdata)#なんかtupleなので使いやすいndarrayに変換
    rawdata = rawdata/max(rawdata)#ノーマライズ
    rawdata_line.set_ydata(rawdata)#生データでリプロット
    smootheddata = np.convolve(rawdata, w, mode='same')#生データを移動平均でtracing
    smootheddata_line.set_ydata(smootheddata)#移動平均データでリプロット
    plt.legend()
    plt.draw()#グラフを再描画
    plt.pause(0.001)#グラフの更新頻度

plt.ion()
fig, ax = plt.subplots()
rawdata_line, = ax.plot(np.zeros(CDC_number),color='k', alpha=0.05,label='raw')#生データプロット用書式
smootheddata_line, = ax.plot(np.zeros(CDC_number),color='orange',alpha=0.75,label='SMA')#移動平均データプロット用書式
ax.set_ylim(0, 1)#y軸を固定する場合

while True:
    bytedata = ser.read(2 * n_zeros)#2バイトごとにCDCを読み込み、n_zeros個(つまり50個)ずつにグループ
    values = struct.unpack('>' + 'H' * n_zeros, bytedata)#読み取ったやつを16進数に変換
    if sum(values)==0:#グループ50個が全部0のとき
        plot()

