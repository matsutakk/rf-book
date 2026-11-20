"""hackrf_sweep の CSV から、周波数ごとの受信電力の最大値を描く（第7章 Wi-Fi のスイープの図）

記録の例:
    hackrf_sweep -f 5100:5900 -w 1000000 -l 32 -g 20 -N 50 > data/sweep_5g.csv
実行:
    python sweep_plot.py data/sweep_5g.csv
"""
import sys
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Hiragino Sans", "Noto Sans CJK JP", "IPAexGothic", "sans-serif"]


def load_sweep(path):
    """CSV を読み、各周波数ビンの最大値（dB）を返す

    1行の形式: 日付, 時刻, 開始Hz, 終了Hz, ビン幅Hz, サンプル数, dB, dB, ...
    """
    peak = defaultdict(lambda: -np.inf)  # 周波数 [Hz] → これまでの最大値 [dB]
    with open(path) as f:
        for line in f:
            cols = line.strip().split(",")
            if len(cols) < 7:  # 途中で切れた行などは飛ばす
                continue
            hz_low, width = float(cols[2]), float(cols[4])  # その行の開始周波数とビン幅
            for i, db in enumerate(cols[6:]):
                freq = hz_low + (i + 0.5) * width  # i番目のビンの中心周波数
                peak[freq] = max(peak[freq], float(db))  # 何回もスイープした中の最大値
    freqs = np.array(sorted(peak))              # 周波数の低い順
    power = np.array([peak[f] for f in freqs])  # それぞれの最大値
    return freqs / 1e6, power - np.median(power)  # MHz、雑音レベル（中央値）を 0dB にする


def channel_freq(ch):
    """5GHz 帯の Wi-Fi チャネルの中心周波数 [MHz]"""
    return 5000 + 5 * ch


def plot_sweep(freqs, power, path="wifi_sweep.png"):
    """周波数 [MHz] と電力 [dB] → チャネル番号や注記を付けた図を保存・表示"""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(freqs, power, lw=0.8)
    ax.set_xlim(5100, 5900)
    ax.set_xlabel("周波数 [MHz]")
    ax.set_ylabel("受信電力の最大値 [dB]（雑音レベル = 0dB）")

    # 記録した範囲（ch112〜124）を強調する
    ax.axvspan(channel_freq(112) - 10, channel_freq(124) + 10, color="tab:orange", alpha=0.15)

    # 上側にチャネル番号の目盛り
    chs = list(range(36, 65, 4)) + list(range(100, 145, 4)) + list(range(149, 166, 4))
    top = ax.secondary_xaxis("top")
    top.set_xticks([channel_freq(c) for c in chs])
    top.set_xticklabels([str(c) for c in chs], fontsize=7, rotation=90)
    top.set_xlabel("Wi-Fi チャネル")

    # 80MHz 幅（ch116〜128、中心 ch122）を括弧で示す
    ymax = power.max()
    y = ymax + 6  # 括弧を描く高さ（いちばん高い山の少し上）
    lo, hi = channel_freq(116) - 10, channel_freq(128) + 10
    ax.plot([lo, lo, hi, hi], [y - 2, y, y, y - 2], color="black", lw=1)
    ax.text((lo + hi) / 2, y + 1, "80MHz 幅（ch116〜128、中心 ch122）", ha="center", va="bottom", fontsize=9)

    # ch120 にプライマリチャネルの矢印
    f120 = channel_freq(120)
    i120 = np.argmin(np.abs(freqs - f120))  # ch120にいちばん近いビン
    ax.annotate("プライマリチャネル（ビーコン）ch120", xy=(f120, power[i120]), xytext=(f120 + 80, ymax - 5),
                arrowprops=dict(arrowstyle="->"), fontsize=9)

    ax.set_ylim(min(-5, power.min()), y + 10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.show()


def main(path):
    freqs, power = load_sweep(path)  # 周波数 [MHz] と、雑音レベルを0dBにした電力
    # 範囲、ビンの数、いちばん強い周波数を表示する
    print(f"{freqs[0]:.1f}〜{freqs[-1]:.1f} MHz, {len(freqs)} bins, max {power.max():.1f} dB "
          f"at {freqs[np.argmax(power)]:.1f} MHz")
    plot_sweep(freqs, power)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/sweep_5g.csv")
