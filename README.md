# サンプルコード

本書『（書名）』のサンプルコードです。各章の処理を1つのPythonファイルにまとめ、章ごとに手順を追えるJupyterノートブックを用意しています。

## ファイル一覧

| 章 | Pythonファイル | ノートブック | 使うIQデータ（例） |
|---|---|---|---|
| 1 無線通信とSDR | `sdr_basics.py` | `notebooks/01_sdr_basics.ipynb` | `adsb.iq` |
| 2 ADS-B | `adsb.py` | `notebooks/02_adsb.ipynb` | `adsb.iq` |
| 3 BLE | `ble.py` | `notebooks/03_ble.ipynb` | `ble_ch37.iq`（なければテスト用IQを生成） |
| 4 Zigbee（コラム Thread） | `zigbee.py`、`thread.py` | `notebooks/04_zigbee.ipynb` | `zigbee_ch20.iq`、`thread_ch11.iq` |
| 5 GPS | `gps.py` | `notebooks/05_gps.ipynb` | `gps.iq` |
| 6 LRPT | `lrpt.py` | `notebooks/06_lrpt.ipynb` | `meteor5.iq`（Meteor-M N2-4）、または sigidwiki で公開されている WAV |
| 7 Wi-Fi | `wifi.py` | `notebooks/07_wifi.ipynb` | `wifi_5g_ch120.iq` |

`thread.py` は `zigbee.py` の関数を読み込んで使うので、同じフォルダに置いたまま使ってください。

## 準備

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## IQデータ

IQデータは容量が大きく、周囲の通信が含まれることもあるため、このリポジトリには含めていません。各章のコマンドで記録し、`data/` フォルダに置いてください。

| ファイル | 記録のコマンド（HackRF） |
|---|---|
| `adsb.iq` | `hackrf_transfer -r adsb.iq -f 1090000000 -s 2000000 -n 60000000` |
| `ble_ch37.iq` | `hackrf_transfer -r ble_ch37.iq -f 2402000000 -s 4000000 -n 40000000` |
| `zigbee_ch20.iq` | `hackrf_transfer -r zigbee_ch20.iq -f 2450000000 -s 4000000 -n 120000000` |
| `thread_ch11.iq` | `hackrf_transfer -r thread_ch11.iq -f 2405000000 -s 4000000 -n 240000000` |
| `gps.iq` | `hackrf_transfer -r gps.iq -f 1575420000 -s 4000000 -p 1 -a 0 -l 40 -g 40 -n 720000000` |
| `meteor5.iq` | `hackrf_transfer -r meteor5.iq -f 137650000 -s 2000000 -a 0 -l 24 -g 40 -n 1200000000`（外付けLNA。衛星が見える時間に合わせて開始） |
| `wifi_5g_ch120.iq` | `hackrf_transfer -r wifi_5g_ch120.iq -f 5600000000 -s 20000000 -n 60000000` |

ゲイン（`-l`、`-g`、`-a`）は環境に合わせて調整してください。チャネルは、お使いの機器に合わせて変えてください（Zigbee/Threadのチャネル `k` の中心周波数は `2405 + 5(k - 11)` MHz）。

LRPTの章では、HackRF で記録した Meteor-M N2-4 の信号（`meteor5.iq`）を使っています。手元に記録がなければ、[sigidwiki の LRPT のページ](https://www.sigidwiki.com/wiki/Low_Rate_Picture_Transmission_(LRPT)) で公開されている、古い Meteor-M N2 の録音でも試せます。

`-p 1`（アンテナ端子への給電）は、給電が必要なアクティブアンテナをつないでいるときだけ指定してください。

## 実行

コマンドラインから（`code/` フォルダで実行）:

```sh
python adsb.py data/adsb.iq
python ble.py data/ble_ch37.iq
python ble.py --make-test test.iq && python ble.py test.iq   # テスト用IQで試す
python zigbee.py data/zigbee_ch20.iq
python thread.py data/thread_ch11.iq
python thread.py --test                                            # テスト用IQで試す
python gps.py data/gps.iq --skip 25 --seconds 60                 # 信号が強い区間を選ぶ
python lrpt.py data/meteor5.iq -f cs8 -r 2000000 --offset 250000 --decim 8 -m oqpsk -o out/   # HackRF の記録（N2-4）
python lrpt.py data/LRPT_16-29-02_137900kHz.wav -o out/                                         # sigidwiki の録音（N2）
python wifi.py data/wifi_5g_ch120.iq
```

ノートブックは `notebooks/` フォルダで開いてください。IQデータを別の場所に置いている場合は、環境変数 `SDR_BOOK_DATA` にそのフォルダを指定します。

```sh
cd notebooks
SDR_BOOK_DATA=/path/to/data jupyter lab
```

## 注意

受信した通信の内容を他人に漏らしたり、自分のために利用したりすることは、電波法で禁止されています。自分の管理する機器や、誰でも受信することを前提に送信されている信号を対象にしてください。電波の送信には、原則として無線局の免許が必要です。
