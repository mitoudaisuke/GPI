from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from scipy.stats import linregress

# 画像を読み込む
image_path = '/Users/mito/Library/Mobile Documents/com~apple~CloudDocs/python/D論用ファイル/4章/neon2.png'  # 自分の画像パスを指定
image = Image.open(image_path)

# グレースケール画像に変換
gray_image = image.convert("L")

# 画像をnumpy配列に変換
gray_array = np.array(gray_image)

# オフセット値（中央ラインからのずらし幅）
offset = 50  # 正の値で下、負の値で上

# 画像の中央ラインをオフセット付きで取得
central_index = gray_array.shape[0] // 2 + offset
if central_index < 0 or central_index >= gray_array.shape[0]:
    raise ValueError("オフセット値が画像範囲を超えています")

central_line = gray_array[central_index, :]

# ピークを検出
peaks, _ = find_peaks(central_line, height=22)  # height=22は閾値（調整可能）
peak_intensities = central_line[peaks]  # 各ピークの強度

# 波長データ（事前に与えられた値）
wavelength = [
    585.24878, 588.1895, 594.4834, 597.55343,
    602.99968, 607.43376, 609.6163, 614.30627, 616.35937,
    621.72812, 626.64952, 630.47893, 633.44276, 638.29914,
    640.2248, 650.65277, 653.28824, 659.89528, 667.82766,
    671.7043, 692.94672, 703.24128
]

# ピークと波長の対応確認
if len(peaks) != len(wavelength):
    raise ValueError("ピークの数と波長データの数が一致していません")

# 校正曲線を作成
pixel_positions = peaks
slope, intercept, r_value, p_value, std_err = linregress(pixel_positions, wavelength)

# ピクセル位置から波長を計算する関数
pixel_to_wavelength = lambda x: slope * x + intercept


def plot(peak_index,xlim_min, xlim_max):
    # 1つ目のピーク（585.24878に対応する）
    # 1つ目のピークのインデックス
    peak_position = peaks[peak_index]  # ピクセル位置
    peak_height = peak_intensities[peak_index]  # ピークの高さ

    # 半最大値の計算
    half_max = peak_height / 2

    # 左側の半最大値の位置を線形補間で計算
    left_indices = np.where(central_line[:peak_position] <= half_max)[0]
    left_fwhm_position = (
        interp1d(
            [central_line[left_indices[-1]], central_line[left_indices[-1] + 1]],
            [left_indices[-1], left_indices[-1] + 1]
        )(half_max)
        if len(left_indices) > 0 else peak_position
    )

    # 右側の半最大値の位置を線形補間で計算
    right_indices = np.where(central_line[peak_position:] <= half_max)[0]
    right_fwhm_position = (
        interp1d(
            [central_line[right_indices[0] + peak_position - 1], central_line[right_indices[0] + peak_position]],
            [right_indices[0] + peak_position - 1, right_indices[0] + peak_position]
        )(half_max)
        if len(right_indices) > 0 else peak_position
    )

    # FWHMの計算
    fwhm_pixels = right_fwhm_position - left_fwhm_position
    fwhm_wavelength = pixel_to_wavelength(right_fwhm_position) - pixel_to_wavelength(left_fwhm_position)

    # プロットの作成
    xlim_range = (xlim_min, xlim_max)
    fig, ax1 = plt.subplots(figsize=(8, 3.5))

    # メインプロット（ピクセル位置 vs 強度）
    ax1.plot(central_line, color='black', lw=1, label='Intensity')
    ax1.axhline(half_max, color='steelblue', lw=1, alpha=1, label='Half Maximum')
    ax1.axvline(left_fwhm_position, color='orange', lw=1, alpha=1, label='FWHM')
    ax1.axvline(right_fwhm_position, color='orange', lw=1, alpha=1)

    # x軸の範囲をクローズアップ
    ax1.set_xlim(xlim_range)
    ax1.set_xlabel("Pixel", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Intensity", fontsize=12, fontweight='bold')
    ax1.grid(axis='both', linestyle='--', alpha=0.7)
    ax1.legend()  # legendの枠を黒で統一

    # 二重x軸（波長を上側に表示）
    x_ticks_pixel = np.arange(xlim_range[0], xlim_range[1] + 1, 10)
    x_ticks_wavelength = pixel_to_wavelength(x_ticks_pixel)
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(x_ticks_pixel)
    ax2.set_xticklabels([f"{tick:.1f}" for tick in x_ticks_wavelength])
    ax2.set_xlabel(r"$\mathbf{λ}$ $\bf(nm)$", fontsize=12, fontweight='bold')

    # FWHM情報をプロットに追加
    fwhm_text = (
        f"FWHM ("+r"$\bf{Pixels}$"+ f"): {fwhm_pixels:.2f}\n"
        f"FWHM ("+r"$\mathbf{λ}$" +f"): {fwhm_wavelength:.2f} nm"
    )
    props = dict(boxstyle='round', facecolor='white', edgecolor='lightgrey', alpha=0.8, pad=0.5)
    ax1.text(
        0.05, 0.95, fwhm_text, transform=ax1.transAxes, fontsize=10,
        verticalalignment='top', bbox=props
    )

    # 保存と表示
    plt.tight_layout()
    plt.savefig("/Users/mito/Desktop/fwhm_analysis_with_wavelength.tiff", dpi=400)
    plt.show()

plot(0,50,150)
#plot(-1,2400,2500)
