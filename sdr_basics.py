"""第1章 無線通信とSDR：IQデータの読み込みとスペクトルの表示

  python sdr_basics.py data/adsb.iq 1090000000 2000000
"""
import sys
import numpy as np
import matplotlib.pyplot as plt


def load_iq(path, count=-1):
    """IQファイルのパス → 複素数の配列（-1〜+1、平均0）。count は読むバイト数（-1で全部）"""
    # HackRFのファイルは符号付き8bitのI, Q, I, Q, ...の並び
    raw = np.fromfile(path, dtype=np.int8, count=count).astype(np.float32)
    x = (raw[0::2] + 1j * raw[1::2]) / 128.0  # I+jQにし、-1〜+1にそろえる
    return x - np.mean(x)                     # 平均（DC成分）を引いて中心を0にする


def show_spectrum(x, fs, fc, nfft=1024):
    """IQ、サンプルレート [Hz]、中心周波数 [Hz] → スペクトルとスペクトログラム"""
    n = len(x) // nfft  # nfftサンプルずつに区切ったときのブロック数
    # 1行が1ブロックの表にし、
    # 端をなめらかに0へ近づける窓（ハニング窓）を掛ける
    blocks = x[:n * nfft].reshape(n, nfft) * np.hanning(nfft)
    # ブロックごとにFFTし、低い周波数から順に並べ替えて（fftshift）電力にする
    power = np.abs(np.fft.fftshift(np.fft.fft(blocks, axis=1), axes=1)) ** 2
    # 各点の周波数：中心からのずれ + 中心周波数 [MHz]
    freq = (np.fft.fftshift(np.fft.fftfreq(nfft, 1 / fs)) + fc) / 1e6

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7))
    # 上：全ブロックの平均（スペクトル）。
    # dBにする（log(0)を避けるため小さな値を足す）
    ax1.plot(freq, 10 * np.log10(power.mean(axis=0) + 1e-12))
    ax1.set_xlabel("Frequency [MHz]")
    ax1.set_ylabel("Power [dB]")
    # 下：ブロックごとの電力を色で並べたもの（スペクトログラム）。縦が時間
    ax2.imshow(10 * np.log10(power + 1e-12), aspect="auto",
               extent=[freq[0], freq[-1], len(x) / fs, 0])
    ax2.set_xlabel("Frequency [MHz]")
    ax2.set_ylabel("Time [s]")
    plt.show()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/adsb.iq"  # IQファイル
    fc = float(sys.argv[2]) if len(sys.argv) > 2 else 1090e6     # 中心周波数 [Hz]
    fs = float(sys.argv[3]) if len(sys.argv) > 3 else 2e6        # サンプルレート [Hz]
    x = load_iq(path, count=int(fs) * 2)  # 先頭 1 秒分
    show_spectrum(x, fs, fc)
