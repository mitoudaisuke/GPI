import signal
import time
import serial
import struct
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button,Slider
import sys
import japanize_matplotlib

plt.ion()
ser = serial.Serial('/dev/cu.usbmodem3495325C35381')
CDC_number = 5000
n_zeros = 50
window = 100
w = np.ones(window) / window

def CDC_receive_start_signal_check():
    signal = False #CDC読み込みスタートポジションフラッグを作成
    while signal==False:
        bytedata = ser.read(2 * n_zeros)#2バイトごとにCDCを読み込み、n_zeros個(つまり50個)ずつにグループ
        values = struct.unpack('>' + 'H' * n_zeros, bytedata)#読み取ったやつを16進数に変換
        if sum(values)==0:#グループ50個が全部0のとき
            signal=True#CDC読み込みスタートポジションフラッグをオンにして
            print('signal on')
    return signal#フラグの状態を返す

def graphplot():
    binarydata = ser.read(2 * CDC_number)#2バイトごとにCDCを読み込み
    rawdata = struct.unpack('>' + 'H' * CDC_number, binarydata)
#    rawdata = np.array(rawdata)#扱いづらいのでndarrayにする
    rawdata = list(rawdata)#扱いづらいのでndarrayにする
    rawdata = [i/max(rawdata) for i in rawdata]
    rawdata_offsets = np.stack([range(5000),rawdata],1)#scatterはグラフ更新に座標ペアを与える必要があるのでペアを作成し
    rawdata_line.set_offsets(rawdata_offsets)#データを更新する
    smootheddata = np.convolve(rawdata, w, mode='same')#移動平均
    smootheddata_line.set_ydata(smootheddata)#plotはそのままyデータを渡してあげるだけでよい
    ser.close()
    ser.open()

fig, ax = plt.subplots()
rawdata_line = ax.scatter(np.zeros(CDC_number),np.zeros(CDC_number), color='k', alpha=0.05, s=2, label='raw')
smootheddata_line, = ax.plot(np.zeros(CDC_number), color='orange', alpha=0.75, label='SMA')
ax.set_ylim(0, 1)

ax_button1 = plt.axes([0.7, 0.01, 0.1, 0.05])
button1 = Button(ax_button1, 'PAUSE')
start_button_flag =True
def on_button1_click(event):# ボタン1がクリックされたときの処理
    global start_button_flag
    print(start_button_flag)
    if start_button_flag==True:start_button_flag=False
    else: start_button_flag=True
button1.on_clicked(on_button1_click)

ax_button2 = plt.axes([0.81, 0.01, 0.1, 0.05])
button2 = Button(ax_button2, 'END')
def on_button2_click(event):# ボタン2がクリックされたときの処理
    sys.exit() 
button2.on_clicked(on_button2_click)

ax_slider = plt.axes([0.25, 0.01, 0.35, 0.03], facecolor='lightgoldenrodyellow')
slider = Slider(ax_slider, '積算時間(0.01-1sec)', 0.001, 1.0, valinit=0.1)
frame_rate =0.1
def set_frame_rate(value):
    global frame_rate
    frame_rate =value
slider.on_changed(set_frame_rate)  # スライダーの値が変更されたときのコールバック関数を設定

while CDC_receive_start_signal_check()==True:#CDC読み込みスタートポジションフラッグがTrueのとき
    fig.canvas.draw()
    fig.canvas.flush_events()
    if start_button_flag==True:    
        graphplot()
    else:
        continue
    time.sleep(frame_rate)

        