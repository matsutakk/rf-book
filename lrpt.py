"""
Meteor-M LRPT デコーダ(I/Q録音 または 軟判定シンボル .s → 画像)

処理の流れ:
  0. I/Q録音(.wav / 生I/Q)なら復調して軟判定シンボルを作る
     (粗い周波数補正 → RRC整合フィルタ → Gardnerタイミング同期 + コスタスループ)
  1. 軟判定シンボル読み込み(.s の場合。int8 の I,Q が交互に並んだファイル)
  2. 位相不確定性と符号の流儀を総当たりで判定(同期語が見つかる組み合わせを採用)
  3. ビタビ復号(畳み込み符号 r=1/2, K=7)。OQPSK(M2-3・M2-4)では NRZ-M も外す
  4. 同期語 0x1ACFFC1D を探して CADU(1024バイト)を切り出す
  5. PN系列で デランダマイズ
  6. Reed-Solomon (255,223) ×4インターリーブで誤り訂正
  7. VCDU → M_PDU → CCSDSパケットに分解
  8. APID 64〜69 の画像パケットを JPEG風にデコードして画像化

使い方:
  python lrpt.py 録音.wav -o out/                      # SDR# などの I/Q WAV
  python lrpt.py 録音.wav -m oqpsk -o out/             # Meteor-M2-3 / M2-4
  python lrpt.py meteor.iq -f cs8 -r 2000000 --offset 250000 --decim 8 -m oqpsk -o out/
                                                       # HackRF(中心137.65MHz)で記録した M2-4
  python lrpt.py 録音.raw -f cf32 -r 250000 -o out/    # GQRX などの生I/Q
  python lrpt.py 録音.s -o out/                        # 復調済みシンボル
"""
import argparse
import os
import time
from collections import defaultdict

import numpy as np
from numba import njit
from PIL import Image
from reedsolo import RSCodec, ReedSolomonError

# ============================================================
# 定数
# ============================================================
ASM = 0x1ACFFC1D            # 同期語
ASM_BITS = np.array([(ASM >> (31 - i)) & 1 for i in range(32)], np.uint8)
FRAME_BITS = 8192           # CADU 1024バイト = 8192ビット
VCDU_LEN = 892              # RS訂正後のデータ部
MCU_PER_LINE = 196          # 1ライン = 196 MCU = 1568画素
MCU_PER_PACKET = 14

# ============================================================
# 1. PN系列(CCSDS: x^8 + x^7 + x^5 + x^3 + 1, 初期値オール1)
# ============================================================
def pn_sequence(n=255):
    """（入力なし） → デランダマイズ用の疑似乱数列（255バイト）"""
    state = 0xFF                    # 8ビットのレジスタ（初期値はすべて1）
    out = bytearray()
    for _ in range(n):              # 1バイトずつ作る
        byte = 0
        for _ in range(8):          # 1ビットずつ作る
            byte = (byte << 1) | ((state >> 7) & 1)  # 最上位ビットが出力
            # CCSDSの疑似乱数列（x^8 + x^7 + x^5 + x^3 + 1）の帰還
            fb = ((state >> 7) ^ (state >> 4) ^ (state >> 2) ^ state) & 1
            state = ((state << 1) | fb) & 0xFF  # 左へずらし、帰還を入れる
        out.append(byte)
    return np.frombuffer(bytes(out), np.uint8)  # 整数の配列にする

# 255バイト周期の列を、フレームの長さ（1020バイト）まで繰り返す
PN = np.tile(pn_sequence(), 4)[:1020]

# ============================================================
# 2. Reed-Solomon (CCSDS: GF多項式 0x187, fcr=112, 根の間隔 11)
#    ※ Meteor LRPT の実データは双対基底変換なしで符号語として正しい
# ============================================================
def _gf_mul(a, b, prim=0x187):
    r = 0
    while b:
        if b & 1:
            r ^= a
        a <<= 1
        if a & 0x100:
            a ^= prim
        b >>= 1
    return r

def _alpha_pow(n):
    x = 1
    for _ in range(n):
        x = _gf_mul(x, 2)
    return x

RS = RSCodec(nsym=32, nsize=255, fcr=112, prim=0x187,
             generator=_alpha_pow(11), c_exp=8)

# 双対基底(Berlekamp表現)⇔ 通常表現 の変換表
_TAL = [0x8d, 0xef, 0xec, 0x86, 0xfa, 0x99, 0xaf, 0x7b]
TO_DUAL = np.zeros(256, np.uint8)     # 通常 → 双対
FROM_DUAL = np.zeros(256, np.uint8)   # 双対 → 通常
for _i in range(256):
    v = 0
    for _j in range(8):
        for _k in range(8):
            if _i & (1 << _k):
                v ^= _TAL[7 - _k] & (1 << _j)
    TO_DUAL[_i] = v
    FROM_DUAL[v] = _i

# シンドローム計算(誤りが無ければ全部0)。きれいな符号語の重いデコードを省くため numba で事前チェック
_EXP = np.zeros(512, np.int32)
_LOG = np.zeros(256, np.int32)
_x = 1
for _i in range(255):
    _EXP[_i] = _x
    _LOG[_x] = _i
    _x = _gf_mul(_x, 2)
_EXP[255:510] = _EXP[:255]

@njit(cache=True)
def _syndromes_zero(cw, exp, log):
    for i in range(32):
        root_log = (11 * (112 + i)) % 255
        s = 0
        for c in cw:                      # ホーナー法
            if s != 0:
                s = exp[(log[s] + root_log) % 255]
            s ^= c
        if s != 0:
            return False
    return True

def rs_correct(frame):
    """1020バイト → (訂正後の892バイト, 訂正数)。無理なら (None, -1)"""
    out = np.empty(VCDU_LEN, np.uint8)   # 訂正後のデータ（892バイト）
    total = 0                            # 訂正したバイトの数
    for i in range(4):                   # 4つの符号語が1バイトずつ交互に並ぶ
        cw = frame[i::4]  # 4バイトおきに取り出して、インターリーブを解く
        # シンドロームがすべて0なら誤りなし
        if _syndromes_zero(cw, _EXP, _LOG):
            data = cw[:223]  # 誤りがなければそのまま
        else:
            try:
                # 訂正する：(データ, 符号語全体, 誤りの位置) が返る
                msg, _, errpos = RS.decode(bytearray(cw.tobytes()))
            except ReedSolomonError:     # 誤りが多すぎて訂正できない
                return None, -1
            data = np.frombuffer(bytes(msg), np.uint8)
            total += len(errpos)
        out[i::4] = data                 # 元の並びに戻す
    return out, total

# ============================================================
# 3. ビタビ復号
# ============================================================
@njit(cache=True)
def _parity(x):
    x ^= x >> 4
    x ^= x >> 2
    x ^= x >> 1
    return x & 1

@njit(cache=True)
def viterbi(soft, g0, g1, inv1):
    """軟判定ビット（+が1）と符号の多項式 → 復号したビット（0/1の配列）"""
    n = len(soft) // 2                   # 1ビットにつき符号ビットが2つ
    e0 = np.empty(128, np.float32)       # 1つ目の符号ビットの期待値（+1/-1）
    e1 = np.empty(128, np.float32)       # 2つ目の符号ビットの期待値
    for r in range(128):              # 7ビットのレジスタ値ごとの期待出力
        e0[r] = 1.0 if _parity(r & g0) else -1.0  # タップの位置のXOR
        e1[r] = 1.0 if (_parity(r & g1) ^ inv1) else -1.0  # inv1：反転の有無

    # 状態 = 直近6ビット（64通り）。metric はそこへ至る経路のもっともらしさ
    metric = np.zeros(64, np.float32)
    new = np.empty(64, np.float32)       # 次の時刻の metric
    dec = np.empty((n, 64), np.uint8)    # 時刻・状態ごとに選んだ遷移元
    for t in range(n):
        s0 = soft[2 * t]                 # 受信した1つ目の軟判定値
        s1 = soft[2 * t + 1]             # 受信した2つ目の軟判定値
        for ns in range(64):          # 遷移先の状態ごとに、2つの遷移元を比較
            ra = ns                   # 押し出されるビット x=0
            rb = 64 | ns              # x=1
            # 期待出力と軟判定値の積を足す（一致するほど大きくなる）
            ma = metric[ns >> 1] + s0 * e0[ra] + s1 * e1[ra]
            mb = metric[(ns >> 1) | 32] + s0 * e0[rb] + s1 * e1[rb]
            if mb > ma:  # 大きいほうの経路を残し、選んだ側を記録
                new[ns] = mb
                dec[t, ns] = 1
            else:
                new[ns] = ma
                dec[t, ns] = 0
        m = new.max()  # 値が大きくなりすぎないよう最大値を引く
        for i in range(64):
            metric[i] = new[i] - m

    out = np.empty(n, np.uint8)       # トレースバック
    st = 0
    for i in range(64):                  # 最後の時刻で最ももっともらしい状態
        if metric[i] > metric[st]:
            st = i
    # 記録した選択を後ろからたどり、ビットを決める
    for t in range(n - 1, -1, -1):
        out[t] = st & 1  # 状態の最下位ビットが、その時刻の入力
        st = (st >> 1) | (int(dec[t, st]) << 5)  # 1つ前の時刻の状態に戻る
    return out

# ============================================================
# 4. 位相不確定性の総当たり
# ============================================================
# 畳み込み符号の多項式はレジスタの向きの流儀で表記が変わるので両方試す
POLYS = [(0x4F, 0x6D), (0x6D, 0x4F), (0x79, 0x5B), (0x5B, 0x79)]
VARIANTS = [(p, inv, rot, off)
            for p in POLYS for inv in (0, 1) for rot in range(8) for off in (0, 1)]

def to_soft(sym, rot, off):
    """シンボル（N行2列）を回転・共役 → 軟判定ビットの配列"""
    i, q = sym[:, 0], sym[:, 1]                # I の列と Q の列
    # rot 0〜3 は90度ずつの回転、4〜7 はさらに Q の符号を反転（共役）
    if rot >= 4:
        q = -q
    for _ in range(rot % 4):                   # 90度回すと (I, Q) → (-Q, I)
        i, q = -q, i
    soft = np.empty(2 * len(i), np.float32)    # I, Q, I, Q ... の順に並べる
    soft[0::2] = i
    soft[1::2] = q
    return soft[off:]                          # off=1 なら1ビットずらす

def nrzm_decode(bits):
    """NRZ-M のビット列 → 元のビット列（隣り合うビットのXOR）"""
    # NRZ-M：1 なら前のビットから反転、0 ならそのまま、という形で送られている
    return np.concatenate([bits[:1], bits[1:] ^ bits[:-1]])

def run_variant(sym, var, nrzm=False):
    (g0, g1), inv, rot, off = var
    bits = viterbi(to_soft(sym, rot, off), g0, g1, inv)
    # Meteor-M N2-3・N2-4 は、ビタビ復号の後に NRZ-M を外す必要がある
    return nrzm_decode(bits) if nrzm else bits

def find_asm(bits, max_err=3):
    """ビットの配列 → 同期語が見つかった位置の配列"""
    bp = bits.astype(np.int16) * 2 - 1  # 0/1 を -1/+1 にする
    # 同期語をずらしながら掛けて足す（一致したビット数 - 違ったビット数）
    corr = np.correlate(bp, ASM_BITS.astype(np.int16) * 2 - 1, mode="valid")
    # 1ビット違うごとに相関は2下がる。max_err ビットまでの誤りを許す
    return np.nonzero(corr >= 32 - 2 * max_err)[0]

def count_pairs(pos):
    s = set(pos.tolist())
    return sum(1 for p in s if p + FRAME_BITS in s)

def detect_variant(sym, win=20000, nrzm=False):
    """シンボル → 同期語がよく見つかる (多項式, 反転, 回転, ずれ) の組"""
    best, best_n = None, 0
    for var in VARIANTS:                       # すべての組み合わせを試す
        # 先頭の一部だけ復号し、同期語が正しい間隔で見つかった数を数える
        bits = run_variant(sym[:win], var, nrzm)
        n = count_pairs(find_asm(bits, max_err=2))
        if n > best_n:                         # いちばん多い組み合わせを残す
            best, best_n = var, n
    return best

def frame_positions(bits):
    """ビットの配列 → フレームの先頭位置のリスト（欠けた同期語は補う）"""
    pos = sorted(set(find_asm(bits).tolist()))  # 見つかった位置（重複なし）
    kept = []                                   # 先頭として採用した位置
    for p in pos:                              # 近すぎる候補は除外
        if not kept or p - kept[-1] >= FRAME_BITS - 64:
            kept.append(p)
    filled = []                                 # 欠けた位置を補った結果
    for a, b in zip(kept, kept[1:] + [None]):   # 隣り合う2つの位置を見る
        filled.append(a)
        # 間隔がフレームの長さの整数倍なら、間のフレームも補う
        if b is not None and (b - a) % FRAME_BITS == 0:
            filled.extend(range(a + FRAME_BITS, b, FRAME_BITS))
    return filled

# ============================================================
# 5. ストリーム全体 → CADU
# ============================================================
def decode_frames(sym, seg=1 << 18, verbose=True, nrzm=False):
    """区間ごとにビタビ復号し、RS訂正済みVCDUを順に返す"""
    pre, post = 1024, FRAME_BITS + 2048
    n = len(sym)
    var = None
    stats = defaultdict(int)
    a, next_ok = 0, 0
    while a < n:
        s, e = max(0, a - pre), min(n, a + seg + post)
        chunk = sym[s:e]
        bits, pos = None, []
        if var is not None:                       # 前回の組み合わせをまず試す
            bits = run_variant(chunk, var, nrzm)
            pos = frame_positions(bits)
        if len(pos) < 2:                          # ダメなら総当たりで再判定
            pos = []
            for w in range(0, max(1, len(chunk) - 20000), 50000):
                v = detect_variant(chunk[w:], nrzm=nrzm)
                if v is not None:
                    var = v
                    bits = run_variant(chunk, var, nrzm)
                    pos = frame_positions(bits)
                    if verbose:
                        (g0, g1), inv, rot, off = var
                        print(f"  [{a/n*100:5.1f}%] 同期: 多項式=({g0:#x},{g1:#x}) "
                              f"反転={inv} 回転={rot} オフセット={off}")
                    break
        good = [p for p in pos if a <= s + p < a + seg and s + p >= next_ok
                and p + FRAME_BITS <= len(bits)]
        # 区間の途中で同期が途切れたら(位相スリップなど)、最後のフレームの直後から再判定
        if good and s + good[-1] + 3 * FRAME_BITS < min(a + seg, n):
            a = s + good[-1] + FRAME_BITS
        else:
            a += seg
        for p in good:
            next_ok = s + p + FRAME_BITS - 64
            frame = np.packbits(bits[p + 32:p + FRAME_BITS]) ^ PN
            vcdu, nerr = rs_correct(frame)
            stats["frames"] += 1
            if vcdu is None:
                stats["rs_fail"] += 1
                continue
            stats["rs_fixed"] += nerr
            yield vcdu
    if verbose:
        print(f"  フレーム {stats['frames']} 個 / RS訂正不能 {stats['rs_fail']} 個 / "
              f"訂正バイト数 {stats['rs_fixed']}")

# ============================================================
# 6. VCDU → M_PDU → CCSDSパケット
# ============================================================
class PacketAssembler:
    def __init__(self):
        self.buf = b""          # 組み立て途中のバイト列
        self.synced = False     # パケットの先頭がわかっているか
        self.last_count = None  # 前のフレームの通し番号

    def feed(self, counter, fhp, data):
        """VCDUの通し番号・先頭位置・データ → 完成したパケットを順に返す"""
        # counter はフレームの通し番号。飛んでいたらフレームが抜けている
        if self.last_count is not None and counter != (self.last_count + 1) & 0xFFFFFF:
            # フレームが抜けたら途中のパケットは捨てる
            self.buf, self.synced = b"", False    # フレームが抜けたら途中のパケットは捨てる
        self.last_count = counter
        if fhp == 0x7FF:                          # このフレームに新しいパケットの先頭は無い
            if self.synced:
                self.buf += data            # 全体が前のパケットの続き
                yield from self._extract()  # できあがったパケットを返す
            return
        if self.synced:                           # 前のパケットの続きを完成させる
            self.buf += data[:fhp]
            yield from self._extract()
        # fhp の位置から、新しいパケットが始まる
        self.buf, self.synced = data[fhp:], True
        yield from self._extract()

    def _extract(self):
        """バッファから、長さのそろったパケットを順に取り出して返す"""
        while len(self.buf) >= 6:           # ヘッダ（6バイト）があるあいだ
            # ヘッダ6バイト + データ長（長さフィールドは実際の長さ - 1）
            total = 6 + ((self.buf[4] << 8) | self.buf[5]) + 1
            if len(self.buf) < total:       # まだ全部そろっていない
                break
            yield self.buf[:total]          # 1つ分を取り出して返す
            self.buf = self.buf[total:]     # 残りは次に回す

def packets_from_vcdus(vcdus):
    """VCDU（892バイト）の並び → CCSDSパケット（バイト列）を順に返す"""
    asm = defaultdict(PacketAssembler)      # 仮想チャネルごとに組み立てる
    for v in vcdus:
        vcid = v[1] & 0x3F                  # 仮想チャネルの番号（6ビット）
        if vcid == 63:                            # 空きフレーム
            continue
        # フレームの通し番号（3バイト）
        counter = (int(v[2]) << 16) | (int(v[3]) << 8) | int(v[4])
        # 最初のパケットの先頭位置
        fhp = ((int(v[8]) & 0x07) << 8) | int(v[9])
        # 10バイト目以降のデータを、そのチャネルの組み立て役に渡す
        yield from asm[vcid].feed(counter, fhp, bytes(v[10:VCDU_LEN]))

# ============================================================
# 7. 画像パケット(JPEG風)のデコード
# ============================================================
def _huff_table(bits, vals):
    table, code, k = {}, 0, 0
    for length in range(1, 17):
        for _ in range(bits[length - 1]):
            table[format(code, f"0{length}b")] = vals[k]
            code += 1
            k += 1
        code <<= 1
    return table

DC_TABLE = _huff_table([0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], list(range(12)))
AC_VALS = [
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
    0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08, 0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0,
    0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28,
    0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
    0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
    0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
    0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
    0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5,
    0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2,
    0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
    0xf9, 0xfa]
AC_TABLE = _huff_table([0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d], AC_VALS)

STD_QT = np.array([
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99], np.float64)

def _zigzag():
    order = sorted(((x, y) for x in range(8) for y in range(8)),
                   key=lambda p: (p[0] + p[1], p[1] if (p[0] + p[1]) % 2 == 0 else p[0]))
    return np.array([x * 8 + y for x, y in order])

ZIGZAG = _zigzag()                     # ジグザグ順 k → 8x8 内の位置

_c = np.array([np.sqrt(0.125)] + [0.5] * 7)
DCT_MAT = _c[:, None] * np.cos((2 * np.arange(8)[None, :] + 1) * np.arange(8)[:, None] * np.pi / 16)

def quant_table(qf):
    """品質係数 qf → 量子化テーブル（64個の配列）"""
    # 品質係数 qf から、標準の量子化テーブルの倍率を決める（JPEGとほぼ同じ式）
    f = 5000.0 / qf if 20 < qf < 50 else 200.0 - 2.0 * qf
    # 標準のテーブル STD_QT を倍率で拡大・縮小する（1より小さくしない）
    return np.maximum(np.round(f / 100.0 * STD_QT), 1)

def _signed(bits, s):
    v = int(bits, 2)
    return v if v >= (1 << (s - 1)) else v - (1 << s) + 1

def _read_code(bits, pos, table):
    for length in range(2, 17):
        sym = table.get(bits[pos:pos + length])
        if sym is not None:
            return sym, pos + length
    raise ValueError("bad huffman code")

def decode_image_packet(comp, qf):
    """圧縮データと品質係数 → 8行112列の画素（0〜255）。壊れていたら None"""
    q = quant_table(qf)                    # 量子化テーブル（64個）
    # 圧縮データを '0101...' の文字列にして、先頭から読み進める
    bits = format(int.from_bytes(comp, "big"), f"0{len(comp) * 8}b")
    pos, prev_dc = 0, 0                    # 読んでいる位置と、前のDC成分
    strip = np.zeros((8, 8 * MCU_PER_PACKET), np.uint8)  # 8行112列の画素
    try:
        for m in range(MCU_PER_PACKET):    # 8行8列のブロック（MCU）を14個
            coef = np.zeros(64)            # このブロックのDCT係数
            # DC成分：前のブロックとの差がハフマン符号で入っている
            cat, pos = _read_code(bits, pos, DC_TABLE)  # 差のビット数を読む
            diff = _signed(bits[pos:pos + cat], cat) if cat else 0  # 差の値
            pos += cat
            prev_dc += diff                # 前のブロックの値に差を足す
            coef[0] = prev_dc * q[0]  # 量子化を戻す
            # AC成分：「0の続く数」と「値」の組がハフマン符号で入っている
            k = 1                          # 次に埋める係数の番号
            while k < 64:
                rs, pos = _read_code(bits, pos, AC_TABLE)
                # 上位4ビットが0の続く数、下位4ビットが値のビット数
                run, size = rs >> 4, rs & 0x0F
                if rs == 0x00:  # EOB（残りはすべて0）
                    break
                if rs == 0xF0:                    # 0が16個
                    k += 16
                    continue
                k += run                   # 0が続いた分だけ進む
                if k > 63:                 # 64個を超えたらデータが壊れている
                    raise ValueError("coef overflow")
                # ジグザグ順で k 番目の係数。量子化を戻して置く
                coef[ZIGZAG[k]] = _signed(bits[pos:pos + size], size) * q[ZIGZAG[k]]
                pos += size
                k += 1
            # 逆DCTで画素の値に戻し、128を足して0〜255の範囲にする
            block = DCT_MAT.T @ coef.reshape(8, 8) @ DCT_MAT + 128
            # 丸めて0〜255に収め、m番目のブロックの位置に置く
            strip[:, m * 8:(m + 1) * 8] = np.clip(np.round(block), 0, 255)
    except (ValueError, IndexError):       # 途中でデータが壊れていたら
        return None
    return strip

# ============================================================
# 8. 画像の組み立て
# ============================================================
def build_images(packets):
    """パケットの並び → APID → 画像（2次元配列）の辞書"""
    segs = defaultdict(list)          # apid → [(ms, mcun, strip)]
    for p in packets:
        apid = ((p[0] & 0x07) << 8) | p[1]  # チャネルを表す番号（64〜69）
        # 画像以外のパケットや、短すぎるパケットは使わない
        if not 64 <= apid <= 69 or len(p) < 6 + 14:
            continue
        pl = p[6:]                           # ヘッダ6バイトを除いた中身
        ms = int.from_bytes(pl[2:6], "big")  # 撮影時刻（1日の中のミリ秒）
        mcun, qf = pl[8], pl[13]  # ライン内の位置（MCU番号）と品質係数
        if mcun % MCU_PER_PACKET or mcun >= MCU_PER_LINE:  # 位置がおかしい
            continue
        strip = decode_image_packet(pl[14:], qf)  # 14バイト目からが圧縮データ
        if strip is not None:
            segs[apid].append((ms, mcun, strip))

    images = {}
    for apid, items in segs.items():
        lines, last = [], None                  # mcun が戻ったら新しいライン
        for ms, mcun, strip in items:
            if last is None or mcun <= last:     # 新しいラインを用意する
                lines.append([ms, np.zeros((8, MCU_PER_LINE * 8), np.uint8)])
            # ラインの中の mcun 番目の位置に、展開した画素を置く
            lines[-1][1][:, mcun * 8:mcun * 8 + strip.shape[1]] = strip
            last = mcun
        # 時刻の間隔から抜けたラインを黒で埋める
        t = np.array([l[0] for l in lines], np.float64)
        dt = np.median(np.diff(t)) if len(t) > 1 else 1
        rows = []
        for i, (ms, img) in enumerate(lines):
            if i and dt > 0:
                gap = int(round((ms - lines[i - 1][0]) / dt)) - 1
                rows.extend([np.zeros_like(img)] * max(0, min(gap, 1000)))
            rows.append(img)
        # ...（時刻の間隔から抜けたラインを黒で埋める処理は省略）
        images[apid] = np.vstack(rows)          # ラインを縦に積んで1枚にする
    return images

# ============================================================
# 9. I/Q 読み込み
# ============================================================
RAW_FORMATS = {                        # 生I/Qファイルの形式 → (numpy型, 中心値, スケール)
    "cf32": (np.float32, 0.0, 1.0),    # GQRX / GNU Radio / SatDump の f32
    "cs16": (np.int16, 0.0, 32768.0),  # SatDump の s16 など
    "cs8": (np.int8, 0.0, 128.0),      # HackRF
    "cu8": (np.uint8, 127.5, 128.0),   # rtl_sdr
}

def read_wav_iq(path):
    """2ch WAV(I=左, Q=右)を読む。PCM 8/16/24/32bit と float32 に対応"""
    with open(path, "rb") as f:
        head = f.read(12)
        if head[:4] != b"RIFF" or head[8:12] != b"WAVE":
            raise ValueError("WAV ファイルではありません")
        fmt = None
        while True:
            h = f.read(8)
            if len(h) < 8:
                raise ValueError("data チャンクが見つかりません")
            cid, size = h[:4], int.from_bytes(h[4:], "little")
            if cid == b"fmt ":
                b = f.read(size)
                tag = int.from_bytes(b[0:2], "little")
                ch = int.from_bytes(b[2:4], "little")
                fs = int.from_bytes(b[4:8], "little")
                bits = int.from_bytes(b[14:16], "little")
                if tag == 0xFFFE and size >= 26:                 # WAVE_FORMAT_EXTENSIBLE
                    tag = int.from_bytes(b[24:26], "little")
                fmt = (tag, ch, fs, bits)
            elif cid == b"data":
                offset = f.tell()
                break
            else:
                f.seek(size + (size & 1), 1)
    if fmt is None:
        raise ValueError("fmt チャンクが見つかりません")
    tag, ch, fs, bits = fmt
    if ch != 2:
        raise ValueError(f"チャンネル数が {ch} です。I/Q 録音は 2ch のはずです(音声録音では?)")
    if tag == 3 and bits == 32:
        x = np.memmap(path, np.float32, "r", offset)
        center, scale = 0.0, 1.0
    elif tag == 1 and bits == 16:
        x, center, scale = np.memmap(path, np.int16, "r", offset), 0.0, 32768.0
    elif tag == 1 and bits == 8:
        x, center, scale = np.memmap(path, np.uint8, "r", offset), 127.5, 128.0
    elif tag == 1 and bits in (24, 32):
        b = np.memmap(path, np.uint8, "r", offset)
        nb = bits // 8                                           # 上位16bitだけ使えば十分
        b = b[: len(b) // nb * nb].reshape(-1, nb)
        v = (b[:, nb - 1].astype(np.int8).astype(np.int32) << 8) | b[:, nb - 2]
        x, center, scale = v, 0.0, 32768.0
    else:
        raise ValueError(f"未対応の WAV 形式です(format={tag}, {bits}bit)")
    return x, center, scale, fs

def load_iq(path, raw_format=None, rate=None, swap=False,
            offset=0.0, decim=1):
    """ファイルのパス → (複素数の配列, サンプルレート [Hz])。WAVか生のIQ"""
    if path.lower().endswith(".wav"):
        # WAVは左右のチャネルが I, Q。中心値・倍率・サンプルレートも返る
        x, center, scale, fs = read_wav_iq(path)
        if rate:
            fs = rate
    else:
        if raw_format is None or rate is None:
            raise ValueError("生I/Qファイルには --format と --rate の指定が必要です")
        # 生のIQ：形式（cs8 など）から、型・中心値・倍率を決める
        dt, center, scale = RAW_FORMATS[raw_format]
        # memmap：ファイルを全部読まずに、配列として少しずつ参照する
        x, fs = np.memmap(path, dt, "r"), rate
    n = len(x) // 2                    # I, Q が交互なので、サンプル数は半分
    # 間引く前のローパス（間引いた後のサンプルレートの40%まで、正負とも通す）
    taps = np.sinc(0.8 / decim * (np.arange(129) - 64)) * np.hamming(129)
    taps = (taps / taps.sum()).astype(np.float32)
    iq = np.empty(n // decim + 1, np.complex64)  # 結果を入れる複素数の配列
    tail = np.zeros(0, np.complex64)   # 前のブロックの末尾（つなぎ用）
    k = 0                              # 書き込んだサンプル数
    step = 1 << 22                     # 一度に変換するサンプル数（約400万）
    for a in range(0, n, step):  # メモリ節約のため少しずつ変換
        # 整数を実数にし、中心を0、振幅を -1〜+1 にそろえる
        blk = (np.asarray(x[2 * a: 2 * min(n, a + step)], np.float32) - center) / scale
        i, q = blk[0::2], blk[1::2]    # 偶数番目が I、奇数番目が Q
        if swap:                       # I と Q が逆の録音に対応する
            i, q = q, i
        z = (i + 1j * q).astype(np.complex64)  # I + jQ の複素数にする
        if offset:  # 中心から offset [Hz] ずれた位置の信号を、0Hz へ移す
            # 各サンプルで戻す位相
            ph = -2 * np.pi * offset / fs * np.arange(a, a + len(z))
            z *= np.exp(1j * ph).astype(np.complex64)
        if decim > 1:  # ローパスを通しながら、decim サンプルに1つだけ残す
            zz = np.concatenate([tail, z])
            i0 = a - len(tail) + len(taps) - 1  # 最初の出力のサンプル番号
            win = np.lib.stride_tricks.sliding_window_view(zz, len(taps))
            z = (win[(-i0) % decim::decim] @ taps).astype(np.complex64)
            tail = zz[-(len(taps) - 1):]
        iq[k:k + len(z)] = z
        k += len(z)
    return iq[:k], fs / decim

# ============================================================
# 10. 復調(I/Q → 軟判定シンボル)
# ============================================================
#   a. DC除去と粗い周波数補正:信号を4乗すると 4×搬送波 に輝線が立つ(RRC整形した
#      QPSK・OQPSK のどちらでも)。区間ごとに測ってドップラーを追う
#   b. RRC 整合フィルタ(Meteor はロールオフ 0.6)
#   c. Gardner 法でシンボルタイミング、コスタスループで位相を追従

def rrc_taps(sps, alpha=0.6, span=8):
    """1シンボルのサンプル数 → RRCフィルタの係数（実数の配列）"""
    t = np.arange(-span * sps, span * sps + 1) / sps            # シンボル単位の時刻
    h = np.empty(len(t))                   # 係数を入れる配列
    for k, x in enumerate(t):              # 時刻ごとに、RRCの式で値を求める
        if abs(x) < 1e-8:                  # 時刻0：一般の式が 0/0 になる
            h[k] = 1 - alpha + 4 * alpha / np.pi
        elif abs(abs(x) - 1 / (4 * alpha)) < 1e-8:  # 分母が0になる時刻
            h[k] = alpha / np.sqrt(2) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * alpha))
                                         + (1 - 2 / np.pi) * np.cos(np.pi / (4 * alpha)))
        else:                              # それ以外の時刻：RRCの一般の式
            h[k] = ((np.sin(np.pi * x * (1 - alpha)) + 4 * alpha * x * np.cos(np.pi * x * (1 + alpha)))
                    / (np.pi * x * (1 - (4 * alpha * x) ** 2)))
    # 係数の2乗の和を1にそろえる（通しても電力が変わらないように）
    return (h / np.sqrt((h ** 2).sum())).astype(np.float32)

def filt(x, taps):
    """IQとフィルタ係数 → フィルタを通したIQ（同じ長さ）"""
    return (np.convolve(x.real, taps, "same") + 1j * np.convolve(x.imag, taps, "same")).astype(np.complex64)

def estimate_carrier(x, fs, max_offset, block_sec=0.25):
    """IQ → 区間ごとの (中心位置, 搬送波の周波数 [Hz], 信号ありか)"""
    n_blk = int(block_sec * fs)               # 1区間のサンプル数（0.25秒分）
    nfft = 1 << int(np.ceil(np.log2(n_blk)))  # FFTの点数（2のべき乗）
    f = np.fft.fftfreq(nfft, 1 / fs)          # FFTの各点の周波数 [Hz]
    centers, freqs = [], []
    for a in range(0, len(x) - n_blk + 1, n_blk):  # 区間ごとに調べる
        y = x[a:a + n_blk].astype(np.complex128)  # 4乗しても精度が保てる型に
        # 電力を1にそろえる
        y = y / (np.sqrt(np.mean(np.abs(y) ** 2)) + 1e-12)
        # 4乗するとQPSKの位相（90度単位）が消え、搬送波の4倍の周波数に線が残る
        P = np.abs(np.fft.fft(y ** 4, nfft)) ** 2  # 4乗した信号の電力
        m = np.abs(f) <= 4 * max_offset            # 探す範囲（ずれの4倍まで）
        score = P[m]
        k = np.argmax(score)                       # いちばん強い線の位置
        fc = f[m][k] / 4  # 4で割って搬送波の周波数に戻す
        snr = score[k] / (np.median(P) + 1e-12)  # 輝線が周りの何倍か
        centers.append(a + n_blk / 2)              # 区間の中心のサンプル番号
        freqs.append(fc if snr > 30 else np.nan)   # 線が弱い区間は信号なし
    centers, freqs = np.array(centers), np.array(freqs)
    ok = ~np.isnan(freqs)
    if ok.sum() >= 5:                                     # 外れ値をメディアンで除去
        med = np.array([np.nanmedian(freqs[max(0, i - 3):i + 4]) for i in range(len(freqs))])
        ok &= np.abs(freqs - med) < 300
    # ...（外れ値の除去は省略）
    return centers, freqs, ok

def mix_down(x, fs, centers, freqs, ok):
    """推定した搬送波(区間の間は線形補間)を打ち消す"""
    if ok.sum() == 0:
        return x, 0.0
    out = np.empty_like(x)
    ph0 = 0.0
    step = 1 << 22
    for a in range(0, len(x), step):
        idx = np.arange(a, min(len(x), a + step))
        fi = np.interp(idx, centers[ok], freqs[ok])
        ph = ph0 + np.cumsum(2 * np.pi * fi / fs)
        out[idx] = x[idx] * np.exp(-1j * ph).astype(np.complex64)
        ph0 = ph[-1] % (2 * np.pi)
    return out, np.nanmean(freqs[ok])

def agc(x, fs, win_sec=0.05):
    """ゆっくり変わる振幅を正規化"""
    n = max(1, int(win_sec * fs))
    p = np.abs(x) ** 2
    nb = len(p) // n
    if nb < 2:
        return x / (np.sqrt(p.mean()) + 1e-12)
    rms = np.sqrt(p[:nb * n].reshape(nb, n).mean(1)) + 1e-12
    g = np.interp(np.arange(len(x)), (np.arange(nb) + 0.5) * n, 1 / rms)
    return (x * g).astype(np.complex64)

@njit(cache=True)
def _cubic(x, t):
    """複素信号 x の時刻 t(サンプル単位、小数)を3次ラグランジュ補間"""
    i = int(np.floor(t))
    mu = t - i
    if i < 1 or i + 2 >= len(x):
        return 0j
    xm1, x0, x1, x2 = x[i - 1], x[i], x[i + 1], x[i + 2]
    c0 = -mu * (mu - 1) * (mu - 2) / 6
    c1 = (mu + 1) * (mu - 1) * (mu - 2) / 2
    c2 = -(mu + 1) * mu * (mu - 2) / 2
    c3 = (mu + 1) * mu * (mu - 1) / 6
    return c0 * xm1 + c1 * x0 + c2 * x1 + c3 * x2

def _loop_gains(bw, zeta=0.707):
    th = bw / (zeta + 1 / (4 * zeta))
    d = 1 + 2 * zeta * th + th * th
    return 4 * zeta * th / d, 4 * th * th / d

@njit(cache=True)
def _track(x, sps, oqpsk, kp_c, ki_c, kp_t, ki_t, lock_out):
    """IQ（整合フィルタ後） → シンボル（N行2列：I, Q の軟判定値）"""
    nmax = int(len(x) / sps) + 4           # 出力できるシンボル数の上限
    out = np.empty((nmax, 2), np.float32)  # 結果（1行が1シンボルの I, Q）
    # 次に読む時刻（サンプル単位、小数）
    t = 4.0 * sps
    period = sps                           # 1シンボルの長さ（少しずつ直す）
    phase, freq = 0.0, 0.0                 # 搬送波の位相と、1シンボルの回転量
    prev_on = 0j                           # 1つ前のシンボル点
    prev_late = 0j                         # 1つ前の、半シンボル後の点
    amp = 1.0                              # 信号の大きさの平均
    k = 0                                  # 出力したシンボルの数
    while t + period + 3 < len(x) and k < nmax:
        rot = np.exp(-1j * phase)  # 推定した位相のずれを打ち消す
        on = _cubic(x, t) * rot                 # シンボル点（補間して読む）
        late = _cubic(x, t + period / 2) * rot  # 半シンボル後
        if oqpsk:
            si, sq = on.real, late.imag
            e_t = prev_late.real * (prev_on.real - on.real) + on.imag * (prev_late.imag - late.imag)
        else:
            si, sq = on.real, on.imag
            e_t = prev_late.real * (prev_on.real - on.real) + prev_late.imag * (prev_on.imag - on.imag)
        # 振幅の追従(誤差を正規化するため)
        # 信号の大きさを、ゆっくり追いかけて平均する
        amp = 0.999 * amp + 0.001 * np.sqrt(0.5 * (si * si + sq * sq))
        a = amp + 1e-9  # 0で割らないよう小さな値を足す
        # コスタス：最も近い理想の点からの位相のずれ
        e_c = (np.sign(si) * sq - np.sign(sq) * si) / a  # QPSK コスタス誤差
        e_t = e_t / (a * a)  # 信号の大きさによらない値にする
        if e_c > 1.0:
            e_c = 1.0
        elif e_c < -1.0:
            e_c = -1.0
        if e_t > 1.0:
            e_t = 1.0
        elif e_t < -1.0:
            e_t = -1.0
        # PI制御：誤差を積分して周波数を、誤差そのもので位相を直す
        # ...（誤差を -1〜+1 に制限する処理は省略）
        freq += ki_c * e_c
        phase += freq + kp_c * e_c
        # タイミング:周期を少しずつ合わせる
        # タイミングも同じように、1シンボルの長さと次の時刻を直す
        period += ki_t * e_t * sps
        if period > sps * 1.01:
            period = sps * 1.01
        elif period < sps * 0.99:
            period = sps * 0.99
        # ...（period を sps の 99〜101% に制限する処理は省略）
        t += period + kp_t * e_t * sps
        # 振幅でならした値を、軟判定シンボルとして出力する
        out[k, 0] = si / a * 0.5
        out[k, 1] = sq / a * 0.5
        prev_on, prev_late = on, late           # 次の誤差の計算に使う
        k += 1
    lock_out[0] = freq                          # 最後の周波数のずれを返す
    return out[:k]                              # 使った行だけを返す

def demodulate(iq, fs, symrate=72000, mode="qpsk", max_offset=5000,
               costas_bw=0.005, timing_bw=0.001, verbose=True):
    sps = fs / symrate
    if sps < 2:
        raise ValueError(f"サンプルレート {fs} Hz は低すぎます({2 * symrate} Hz 以上必要)")
    iq = iq - iq.mean()
    taps = rrc_taps(sps)
    centers, freqs, ok = estimate_carrier(filt(iq, taps), fs, max_offset)
    if verbose:
        if ok.any():
            print(f"  搬送波: {np.nanmin(freqs[ok]):+.0f} 〜 {np.nanmax(freqs[ok]):+.0f} Hz"
                  f"(信号あり区間 {ok.mean() * 100:.0f}%)")
        else:
            print("  警告: 搬送波が見つかりません(変調方式やサンプルレートを確認)")
    x, _ = mix_down(iq, fs, centers, freqs, ok)
    x = agc(filt(x, taps), fs)
    kp_c, ki_c = _loop_gains(costas_bw)
    kp_t, ki_t = _loop_gains(timing_bw)
    info = np.zeros(2)
    sym = _track(x, sps, mode == "oqpsk", kp_c, ki_c, kp_t, ki_t, info)
    if verbose:
        print(f"  シンボル {len(sym):,} 個(1シンボルあたり {sps:.3f} サンプル)")
    return sym

def save_constellation(sym, path, size=256):
    """コンステレーション(シンボル点の散布図)を PNG に保存。4つの塊に分かれていれば正常"""
    s = sym[len(sym) // 2: len(sym) // 2 + 50000]
    h, _, _ = np.histogram2d(s[:, 1], s[:, 0], bins=size, range=[[-1.2, 1.2], [-1.2, 1.2]])
    img = (255 * np.log1p(h) / (np.log1p(h).max() + 1e-9)).astype(np.uint8)[::-1]
    Image.fromarray(img).save(path)

# ============================================================
# メイン
# ============================================================
def main():
    ap = argparse.ArgumentParser(description="Meteor-M LRPT decoder (I/Q録音 または .s → PNG)")
    ap.add_argument("input", help="I/Q録音(.wav / 生I/Q)または軟判定シンボル .s(int8, I/Q交互)")
    ap.add_argument("-o", "--out", default="lrpt_out", help="出力フォルダ")
    ap.add_argument("-m", "--mode", choices=["qpsk", "oqpsk"], default="qpsk",
                    help="変調方式(Meteor-M2: qpsk / M2-3・M2-4: oqpsk)")
    ap.add_argument("-r", "--rate", type=float, help="サンプルレート [Hz](生I/Qでは必須、WAVでは上書き)")
    ap.add_argument("-f", "--format", choices=sorted(RAW_FORMATS), help="生I/Qの形式")
    ap.add_argument("--symrate", type=float, default=72000, help="シンボルレート(既定 72000)")
    ap.add_argument("--max-offset", type=float, default=5000, help="探す周波数ずれの範囲 ±[Hz]")
    ap.add_argument("--swap-iq", action="store_true", help="I と Q を入れ替える")
    ap.add_argument("--offset", type=float, default=0, help="信号が中心から何Hzずれた位置にあるか")
    ap.add_argument("--decim", type=int, default=1, help="読み込むときに間引く比率(例: 2MHz → 250kHz なら 8)")
    ap.add_argument("--save-s", metavar="FILE", help="復調したシンボルを .s として保存")
    args = ap.parse_args()

    t0 = time.time()
    if args.input.lower().endswith(".s"):
        raw = np.fromfile(args.input, np.int8)
        sym = (raw[: len(raw) // 2 * 2].reshape(-1, 2).astype(np.float32) / 127.0)
    else:
        # Step 1：IQ（複素数の配列）とサンプルレート
        iq, fs = load_iq(args.input, args.format, args.rate, args.swap_iq,
                         args.offset, args.decim)
        print(f"I/Q {len(iq):,} サンプル @ {fs:,.0f} Hz({len(iq) / fs:.0f}秒)")
        # Step 2〜4：搬送波の補正 → 整合フィルタ → 同期。シンボル（N行2列）
        sym = demodulate(iq, fs, args.symrate, args.mode, args.max_offset)
        del iq
        os.makedirs(args.out, exist_ok=True)
        save_constellation(sym, os.path.join(args.out, "constellation.png"))
        if args.save_s:
            np.clip(np.round(sym * 127), -127, 127).astype(np.int8).tofile(args.save_s)
            print(f"  シンボルを保存 → {args.save_s}")
    print(f"シンボル数 {len(sym):,}({len(sym) / args.symrate:.0f}秒分)")

    # Step 5〜8：誤り訂正したVCDU（O-QPSK の N2-3・N2-4 は NRZ-M も外す）
    vcdus = decode_frames(sym, nrzm=args.mode == "oqpsk")
    packets = packets_from_vcdus(vcdus)  # Step 9：CCSDSパケット（バイト列）
    images = build_images(packets)       # Step 10〜11：APID → 画像（2次元配列）

    os.makedirs(args.out, exist_ok=True)
    if not images:
        print("  画像パケットが取れませんでした(--mode や --swap-iq を変えて試してください)")
    for apid, img in sorted(images.items()):
        path = os.path.join(args.out, f"apid{apid}.png")
        Image.fromarray(img).save(path)
        print(f"  APID {apid}: {img.shape[1]}x{img.shape[0]} → {path}")
    if len(images) >= 3:
        r, g, b = sorted(images)[:3][::-1]
        h = min(images[r].shape[0], images[g].shape[0], images[b].shape[0])
        rgb = np.dstack([images[r][:h], images[g][:h], images[b][:h]])
        path = os.path.join(args.out, f"rgb_{r}_{g}_{b}.png")
        Image.fromarray(rgb).save(path)
        print(f"  合成 R={r} G={g} B={b} → {path}")
    print(f"完了 {time.time() - t0:.1f}秒")

if __name__ == "__main__":
    main()
