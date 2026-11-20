"""Wi-Fi (IEEE 802.11a/g OFDM, 20MHz) を IQ データから復調するサンプル

hackrf_transfer -r wifi_5g_ch120.iq -f 5600000000 -s 20000000 -n 60000000 -a 1 -l 32 -g 24
で記録したファイルを想定
"""
import sys
import zlib
import numpy as np

SAMPLE_RATE = 20_000_000  # サンプルレート [Hz]（20MHz幅をそのまま受ける）
N_FFT = 64                # FFTの点数 = サブキャリアの本数（間隔312.5kHz）
# ガードインターバル（CP）のサンプル数（0.8マイクロ秒）
N_CP = 16
# FFTの窓をガードインターバル側に少し早めて、前のシンボルの影響を避ける
BACKOFF = 4

# 802.11a/g のロングトレーニング系列（LTF）。各サブキャリアに載せる既知の値
# サブキャリア -26〜+26 の値
LTF_SEQ = np.array([1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1,
                    -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 0, 1, -1, -1, 1, 1,
                    -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, 1,
                    -1, 1, -1, 1, 1, 1, 1])
LTF = np.zeros(N_FFT)
LTF[np.arange(-26, 27) % N_FFT] = LTF_SEQ  # FFTのビン番号（k mod 64）に並べる
# 時間波形にしたLTF（相関で探すときに使う）
LTF_TIME = np.fft.ifft(LTF)

# パイロット（位相補正用の既知の値）のサブキャリア
PILOT_K = np.array([-21, -7, 7, 21])
PILOT_VAL = np.array([1, 1, 1, -1])   # パイロットの値
# データを載せるサブキャリア（48本）
DATA_K = np.array([k for k in range(-26, 27) if k != 0 and k not in PILOT_K])

# RATE の4ビット → (Mbps, 1サブキャリアあたりのビット数, 符号化率)
RATES = {
    0b1101: (6, 1, (1, 2)), 0b1111: (9, 1, (3, 4)),
    0b0101: (12, 2, (1, 2)), 0b0111: (18, 2, (3, 4)),
    0b1001: (24, 4, (1, 2)), 0b1011: (36, 4, (3, 4)),
    0b0001: (48, 6, (2, 3)), 0b0011: (54, 6, (3, 4)),
}


# ---------------------------------------------------------------- IQ とバースト

def load_iq(path):
    """IQファイルのパス → 複素数の配列（-1〜+1、平均0）"""
    # HackRFのファイルは符号付き8bitのI, Q, I, Q, ...の並び
    raw = np.fromfile(path, dtype=np.int8).astype(np.float32)
    x = (raw[0::2] + 1j * raw[1::2]) / 128.0  # I+jQにし、-1〜+1にそろえる
    return x - np.mean(x)                     # 平均（DC成分）を引いて中心を0にする


def detect_bursts(x, win=80, factor=10.0, min_len=400):
    """IQ（複素数の配列） → 電波が出ている区間 (開始, 終了) のリスト"""
    # 電力を4マイクロ秒（80サンプル）で平均し、雑音の10倍を超える区間を探す
    power = np.convolve(np.abs(x) ** 2, np.ones(win) / win, mode="same")
    # 下から20%の電力を雑音レベルとみなす
    # （中央値だと混んだチャネルで信号に引っぱられる）
    active = power > np.percentile(power, 20) * factor
    edges = np.diff(active.astype(np.int8))  # 立ち上がり+1、立ち下がり-1
    starts = np.where(edges == 1)[0]         # 立ち上がりのサンプル番号
    ends = np.where(edges == -1)[0]          # 立ち下がりのサンプル番号
    if len(ends) and len(starts) and ends[0] < starts[0]:
        ends = ends[1:]  # 記録の開始時に途中だったバーストは除く
    # 20マイクロ秒（400サンプル）より短い区間は捨てる
    return [(s, e) for s, e in zip(starts, ends) if e - s > min_len]


# ---------------------------------------------------------------- 同期

def coarse_cfo(burst):
    """バーストのIQ → 周波数のずれ [rad/サンプル]"""
    # ショートトレーニング（0.8マイクロ秒 = 16サンプル周期）の位相の回転から
    # 周波数のずれを求める
    s = burst[:160]  # STFの8マイクロ秒分（16サンプルの繰り返し10回）
    # 16サンプル離れた点どうしを掛けると、その間に回った位相が残る。
    # 足して平均する
    return np.angle(np.sum(s[:144] * np.conj(s[16:160]))) / 16  # [rad/sample]


def find_ltf(y, threshold=0.7):
    """IQ → 1つ目のLTFの先頭位置。見つからなければ None"""
    # 既知の LTF との正規化相関で、64サンプル離れた2つのピークを探す
    seg = y[:900]  # バーストの先頭付近だけを調べる
    # 各位置でのLTFとの相関
    corr = np.abs(np.correlate(seg, LTF_TIME, mode="valid"))
    # 各位置から64サンプル分の信号の大きさ
    energy = np.sqrt(np.convolve(np.abs(seg) ** 2, np.ones(N_FFT), mode="valid"))
    # 信号の強さで割って、0〜1の相関にそろえる
    c = corr / (energy * np.linalg.norm(LTF_TIME) + 1e-12)
    # LTFは2回繰り返すので、64サンプル後にもピークがある位置を探す
    cand = [k for k in range(len(c) - N_FFT) if c[k] > threshold and c[k + N_FFT] > threshold]
    if not cand:
        return None
    # 最初のほうの候補のうち、2つのピークの合計がいちばん大きい位置
    return max(cand[:8], key=lambda k: c[k] + c[k + N_FFT])


def synchronize(burst):
    """バーストのIQ → (周波数を補正したIQ, LTFの先頭位置)。無理なら None"""
    w = coarse_cfo(burst)  # STFから求めた周波数のずれ [rad/サンプル]
    y = burst * np.exp(1j * w * np.arange(len(burst)))  # 粗い補正
    p = find_ltf(y)        # LTFの先頭位置
    if p is None:
        return None
    # 2つの LTF の位相差から、残った周波数のずれを細かく補正する
    w2 = np.angle(np.sum(y[p:p + 64] * np.conj(y[p + 64:p + 128]))) / 64
    y = y * np.exp(1j * w2 * np.arange(len(y)))
    return y, p


# ---------------------------------------------------------------- OFDM シンボル

def estimate_channel(y, p):
    """IQとLTFの位置 → 64本のサブキャリアの伝わり方（複素数の配列）"""
    a = np.fft.fft(y[p - BACKOFF:p - BACKOFF + 64])        # 1回目のLTFのFFT
    b = np.fft.fft(y[p + 64 - BACKOFF:p + 128 - BACKOFF])  # 2回目のLTFのFFT
    h = (a + b) / 2  # 2回分を平均して雑音を減らす
    # 受信値を既知の値で割ると、各サブキャリアの伝わり方（振幅と位相）になる
    return np.where(LTF != 0, h / np.where(LTF == 0, 1, LTF), 1)


PILOT_POLARITY = None  # 後で定義（scrambler を使うため）


def ofdm_symbol(y, p, h, i):
    """IQ、LTFの位置、チャネル、番号 i → データの48本のサブキャリアの値"""
    # i番目の OFDM シンボル（0 = SIGNAL）を FFT し、
    # 等化とパイロットによる位相補正をする
    # LTF 2回分（128）の後、1シンボル = CP 16 + 本体 64 = 80サンプルずつ並ぶ
    start = p + 128 + N_CP + 80 * i - BACKOFF
    if start + N_FFT > len(y):  # バーストの終わりを越えた
        return None
    # FFTし、チャネルで割る（等化）
    s = np.fft.fft(y[start:start + N_FFT]) / h
    # パイロットに既知の値と極性を掛け、残った位相の回転を求める
    pilots = s[PILOT_K % N_FFT] * PILOT_VAL * PILOT_POLARITY[i % 127]
    s = s * np.exp(-1j * np.angle(np.sum(pilots)))  # 回転を戻す
    return s[DATA_K % N_FFT]  # データの48本だけを取り出す


# ---------------------------------------------------------------- 復号

def demap(d, nbpsc):
    """48本の値と1本あたりのビット数 → 軟判定ビット（48行 nbpsc 列の配列）"""
    # 軟判定ビット（+ が 1）。グレイ符号の QAM
    if nbpsc == 1:  # BPSK：実部の符号が1ビット
        return d.real[:, None]
    if nbpsc == 2:  # QPSK：実部と虚部の符号が1ビットずつ
        return np.stack([d.real, d.imag], axis=1)
    if nbpsc == 4:
        # 16QAM：点は -3, -1, 1, 3。符号で1ビット目、|i| < 2 かで2ビット目
        # 平均電力1の点を -3〜3 の目盛りに戻す
        i, q = d.real * np.sqrt(10), d.imag * np.sqrt(10)
        return np.stack([i, 2 - np.abs(i), q, 2 - np.abs(q)], axis=1)
    # 64QAM：点は -7〜7 の奇数。同じ考え方を3段にする
    i, q = d.real * np.sqrt(42), d.imag * np.sqrt(42)  # -7〜7 の目盛りに戻す
    return np.stack([i, 4 - np.abs(i), 2 - np.abs(4 - np.abs(i)),
                     q, 4 - np.abs(q), 2 - np.abs(4 - np.abs(q))], axis=1)


def interleave_index(ncbps, nbpsc):
    """1シンボルのビット数など → 並べ替えの順番（整数の配列）"""
    # 符号化後 k 番目のビットが、送信時に何番目に置かれるか
    s = max(nbpsc // 2, 1)  # 2段目で入れ替える単位
    k = np.arange(ncbps)    # 符号化後のビットの番号 0, 1, 2, ...
    i = (ncbps // 16) * (k % 16) + k // 16  # 1段目：16列の表に書いて縦に読む
    # 2段目：QAMの強いビットと弱いビットが交互になるよう入れ替える
    return s * (i // s) + (i + ncbps - (16 * i) // ncbps) % s


def depuncture(soft, rate):
    """間引かれた軟判定ビットと符号化率 → 符号化率1/2の並びに戻した配列"""
    if rate == (1, 2):  # 間引きなし
        return soft
    if rate == (3, 4):  # A0 B0 A1 B2 → A0 B0 A1 _ _ B2
        n = len(soft) // 4  # 4ビットで1組
        out = np.zeros((n, 6))  # 1組を6ビットに広げる（間引いた位置は0のまま）
        out[:, [0, 1, 2, 5]] = soft[:n * 4].reshape(n, 4)
    else:  # (2, 3): A0 B0 A1 → A0 B0 A1 _
        n = len(soft) // 3  # 3ビットで1組
        out = np.zeros((n, 4))  # 1組を4ビットに広げる
        out[:, [0, 1, 2]] = soft[:n * 3].reshape(n, 3)
    return out.reshape(-1)  # 1列に並べ直す


def viterbi(soft):
    """軟判定ビットの配列（+が1） → 復号したビット（0/1の配列）"""
    # 畳み込み符号（K=7, 133/171）。状態 = 直近6ビット（新しいビットが上位）
    n = len(soft) // 2   # 入力ビット数（1ビットにつき符号2ビット）
    # 7ビットのレジスタ（入力 + 状態6ビット）のすべての値
    reg = np.arange(128)
    # レジスタのうち、指定した段のビットをXORする（符号器の出力の計算）
    taps = lambda r, ds: np.bitwise_xor.reduce([(r >> (6 - d)) & 1 for d in ds])
    e0 = 2.0 * taps(reg, [0, 2, 3, 5, 6]) - 1  # 133（8進）
    e1 = 2.0 * taps(reg, [0, 1, 2, 3, 6]) - 1  # 171（8進）
    ns = np.arange(64)    # 遷移先の状態 0〜63
    pa, pb = (ns << 1) & 63, ((ns << 1) & 63) | 1  # 2つの遷移元
    # それぞれの遷移のときのレジスタの値（入力ビット + 遷移元の状態）
    ra, rb = ((ns >> 5) << 6) | pa, ((ns >> 5) << 6) | pb
    metric = np.full(64, -1e9)  # 各状態にたどり着く道の「もっともらしさ」
    metric[0] = 0  # 状態0から始まる
    # 時刻・状態ごとに、どちらの遷移元を選んだか
    dec = np.zeros((n, 64), np.uint8)
    for t in range(n):
        s0, s1 = soft[2 * t], soft[2 * t + 1]  # この時刻の受信値2つ
        # 遷移元a, bそれぞれから来た場合のメトリック
        # （受信値と期待する符号の一致度を足す）
        ma = metric[pa] + s0 * e0[ra] + s1 * e1[ra]
        mb = metric[pb] + s0 * e0[rb] + s1 * e1[rb]
        dec[t] = mb > ma  # どちらの遷移元を選んだか
        metric = np.maximum(ma, mb)  # 良いほうだけを残す
        # 値が大きくなりすぎないよう、最大を0にそろえる
        metric -= metric.max()
    # 最後の状態から、選んだ遷移元を逆にたどってビットを決める
    out = np.zeros(n, np.uint8)
    st = 0  # 末尾は 0 を6個入れて状態 0 に戻している
    for t in range(n - 1, -1, -1):
        out[t] = st >> 5  # 状態の最上位ビットが、その時刻の入力
        st = ((st << 1) & 63) | dec[t, st]  # 1つ前の時刻の状態へ戻る
    return out


def scrambler(n, state):
    """長さ n と初期状態 → スクランブラの出力（0/1の配列）"""
    # x^7 + x^4 + 1 の LFSR（BLE のホワイトニングと同じ多項式）
    out = np.empty(n, np.uint8)
    for i in range(n):
        b = ((state >> 6) ^ (state >> 3)) & 1  # 7段目と4段目のXOR
        # 左へずらし、新しいビットを入れる
        state = ((state << 1) | b) & 0x7F
        out[i] = b
    return out


# パイロットの極性：同じLFSRの出力を +1/-1 にした127周期の列
PILOT_POLARITY = 1 - 2 * scrambler(127, 0x7F).astype(int)


def decode_signal(sym):
    """SIGNALの48本の値 → (RATEのコード, LENGTH [バイト])。無理なら None"""
    soft = sym.real[interleave_index(48, 1)]  # BPSKなので実部だけ
    # 24ビット：RATE 4, 予約 1, LENGTH 12, パリティ 1, テイル 6
    bits = viterbi(soft)
    if bits[18:24].any() or bits[:17].sum() % 2 != bits[17]:
        return None  # テイルビットかパリティが合わない
    code = int("".join(map(str, bits[0:4])), 2)  # RATEの4ビット
    if code not in RATES or bits[4]:  # 未定義のRATEか、予約ビットが1
        return None
    # LENGTH は12ビットで、下位ビットが先
    length = int(np.dot(bits[5:17], 1 << np.arange(12)))
    return code, length


def decode_ppdu(burst):
    """バーストのIQ → 速度、PSDU（バイト列）、QAMの点の辞書。無理なら None"""
    sync = synchronize(burst)
    if sync is None:
        return None
    y, p = sync                 # 周波数を補正したIQと、LTFの先頭位置
    h = estimate_channel(y, p)  # 各サブキャリアの伝わり方
    sig = decode_signal(ofdm_symbol(y, p, h, 0))  # 0番目のシンボルがSIGNAL
    if sig is None:
        return None
    code, length = sig
    mbps, nbpsc, rate = RATES[code]
    ncbps = 48 * nbpsc                  # 1シンボルあたりの符号化後のビット数
    ndbps = ncbps * rate[0] // rate[1]  # 1シンボルあたりのデータのビット数
    # SERVICE 16ビット + データ + テイル6ビットが入るシンボル数
    n_sym = int(np.ceil((16 + 8 * length + 6) / ndbps))
    idx = interleave_index(ncbps, nbpsc)  # インターリーブを戻す並べ方
    soft, points = [], []  # soft：軟判定ビット、points：QAMの点（図を描く用）
    for i in range(1, n_sym + 1):  # SIGNALの次から、DATAのシンボルを順に
        d = ofdm_symbol(y, p, h, i)
        if d is None:
            return None
        points.append(d)
        # QAMの点 → 軟判定ビット → インターリーブの並べ替えを戻す
        soft.append(demap(d, nbpsc).reshape(-1)[idx])
    # 間引かれたビットを0（どちらとも言えない）で埋めてから復号する
    bits = viterbi(depuncture(np.concatenate(soft), rate))
    # SERVICE の先頭7ビットは 0 なので、
    # 受信値がそのままスクランブラの出力になる
    state = 0
    for b in bits[:7]:
        # 先頭7ビットがそのままスクランブラの状態になる
        state = (state << 1) | int(b)
    # その状態から続きを作り、全体のスクランブラの出力にする
    seq = np.concatenate([bits[:7], scrambler(len(bits) - 7, state)])
    data = bits ^ seq  # もう一度XORすると、元のデータに戻る
    # 各バイトは下位ビットから送られるので、並びを反転してバイトにする
    psdu = np.packbits(data[16:16 + 8 * length].reshape(-1, 8)[:, ::-1]).tobytes()
    return {"mbps": mbps, "psdu": psdu, "points": np.concatenate(points)}


def check_fcs(psdu):
    """PSDUのバイト列 → 末尾4バイトのFCSが合っていれば True"""
    # FCSはイーサネットと同じCRC-32（末尾4バイト、リトルエンディアン）
    return len(psdu) >= 4 and zlib.crc32(psdu[:-4]) == int.from_bytes(psdu[-4:], "little")


# ---------------------------------------------------------------- フレームの解析

# (タイプ, サブタイプ) → フレームの種類の名前。
# タイプ 0：管理、1：制御、2：データ
FRAME_TYPES = {
    (0, 8): "Beacon", (0, 4): "Probe Request", (0, 5): "Probe Response", (0, 13): "Action",
    (0, 14): "Action No Ack", (1, 4): "Beamforming Poll", (1, 5): "VHT NDP Announce",
    (1, 8): "Block Ack Request", (1, 9): "Block Ack", (1, 11): "RTS", (1, 12): "CTS", (1, 13): "ACK",
    (2, 0): "Data", (2, 4): "Null", (2, 8): "QoS Data", (2, 12): "QoS Null",
}


def mac(b):
    """6バイト → 「aa:bb:cc:dd:ee:ff」形式の文字列"""
    return ":".join(f"{v:02x}" for v in b)  # 送られてくる順のまま表示する


def parse_frame(psdu):
    """PSDUのバイト列 → フレームの種類、アドレス、情報要素などの辞書"""
    fc = psdu[0]  # フレーム制御の1バイト目
    # bit2〜3がタイプ、bit4〜7がサブタイプ
    ftype, subtype = (fc >> 2) & 3, (fc >> 4) & 15
    frame = {"type": FRAME_TYPES.get((ftype, subtype), f"type{ftype}/{subtype}"),
             "addr1": mac(psdu[4:10])}  # アドレス1：宛先
    if frame["type"] == "Beacon":
        # アドレス3：アクセスポイントのBSSID
        frame["bssid"] = mac(psdu[16:22])
        # MACヘッダ24バイトの後、
        # タイムスタンプ8バイトに続く2バイトがビーコンの間隔 [TU]
        frame["interval"] = int.from_bytes(psdu[32:34], "little")
        # 間隔と能力情報（2バイト）の後ろから、末尾のFCSの手前までが情報要素
        frame["elements"] = parse_elements(psdu[36:-4])
    return frame


def parse_elements(body):
    """ビーコンの本体 → 要素ID → データ（バイト列）の辞書"""
    elems, i = {}, 0  # elems：結果、i：読んでいる位置
    while i + 2 <= len(body):
        # 要素ID（0：SSID、3：チャネル など）と長さ
        eid, n = body[i], body[i + 1]
        # 同じIDが複数あれば最初のもの
        elems.setdefault(eid, body[i + 2:i + 2 + n])
        i += 2 + n  # 次の要素へ（ID 1バイト + 長さ 1バイト + データ）
    return elems


# ---------------------------------------------------------------- 11ac のフレーム

def read_vht_sig_a(burst):
    """バーストのIQ → (帯域幅 [MHz], MCS, ストリーム数)。無理なら None"""
    # L-SIG の後ろの2シンボル（BPSK と 90度回転した BPSK）を読み、
    # 帯域幅と MCS を取り出す
    sync = synchronize(burst)
    if sync is None:
        return None
    y, p = sync
    h = estimate_channel(y, p)
    # 0番目はL-SIG。その後ろの2シンボルがVHT-SIG-A
    s1, s2 = ofdm_symbol(y, p, h, 1), ofdm_symbol(y, p, h, 2)
    # 2シンボル目が虚軸に乗っていれば（QBPSK）VHT形式
    if s2 is None or np.mean(np.abs(s2.imag)) < 2 * np.mean(np.abs(s2.real)):
        return None
    idx = interleave_index(48, 1)  # BPSKと同じ並べ替え
    # 1シンボル目は実部、2シンボル目は虚部に情報がある
    bits = viterbi(np.concatenate([s1.real[idx], s2.imag[idx]]))
    if bits[42:48].any():  # 末尾のテイルビット6個は0のはず
        return None
    # 値はどれも下位ビットが先
    # 帯域幅 [MHz]
    bw = {0: 20, 1: 40, 2: 80, 3: 160}[int(bits[0] | (bits[1] << 1))]
    # 変調方式と符号化率の番号
    mcs = int(np.dot(bits[28:32], 1 << np.arange(4)))
    # ストリーム数（値 + 1）
    nsts = int(np.dot(bits[10:13], 1 << np.arange(3))) + 1
    return bw, mcs, nsts


# ---------------------------------------------------------------- まとめ

def main(path):
    x = load_iq(path)                       # IQ（複素数の配列）
    bursts = detect_bursts(x)               # Step 1：(開始, 終了) のリスト
    print(f"bursts: {len(bursts)}")
    # stats：(速度, 種類) → 数、aps：BSSID → (SSID, チャネル, 間隔)
    # qam：24MbpsのQAMの点、vht：(帯域幅, MCS, ストリーム数) → 数
    stats, aps, qam, vht = {}, {}, [], {}
    for s, e in bursts:
        burst = x[max(0, s - 40):e + 200]   # 前後に少し余裕を持たせる
        # Step 2〜5：同期 → OFDMの復調 → SIGNAL → DATAの復号。PSDUなどの辞書
        r = decode_ppdu(burst)
        if r is None or not check_fcs(r["psdu"]):  # Step 6：FCSの検査
            # 読めなければ 11ac のフレームかを調べる
            v = read_vht_sig_a(burst)
            if v:
                vht[v] = vht.get(v, 0) + 1
            continue
        f = parse_frame(r["psdu"])          # Step 7：MACフレームの辞書
        key = (r["mbps"], f["type"])        # 速度とフレームの種類ごとに数える
        stats[key] = stats.get(key, 0) + 1
        if f["type"] == "Beacon":
            # 要素0：SSID
            ssid = f["elements"].get(0, b"").decode("utf-8", "replace")
            ch = f["elements"].get(3, b"\x00")[0]  # 要素3：チャネル
            aps[f["bssid"]] = (ssid, ch, f["interval"])
        if r["mbps"] == 24:                 # 16QAMの点を図に使うため保存する
            qam.append(r["points"])
    # 集計結果の表示
    print("decoded frames (FCS OK):")
    for (mbps, t), n in sorted(stats.items()):
        print(f"  {mbps:2d} Mbps  {t:18s} {n:5d}")
    print("access points:")
    for bssid, (ssid, ch, iv) in aps.items():
        print(f"  {bssid}  ch={ch:3d}  interval={iv} TU  SSID={ssid!r}")
    print("VHT (11ac) frames: (bandwidth MHz, MCS, streams)")
    for k, n in sorted(vht.items(), key=lambda kv: -kv[1])[:6]:
        print(f"  {k}  {n}")
    if qam:
        np.save("wifi_16qam_points.npy", np.concatenate(qam))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/wifi_5g_ch120.iq")
