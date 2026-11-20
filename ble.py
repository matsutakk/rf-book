"""BLE (LE 1M PHY) のアドバタイジングパケットを IQ データから復調するサンプル

hackrf_transfer -r ble_ch37.iq -f 2402000000 -s 4000000 -n 40000000 で記録したファイルを想定
（リアルタイム受信版は ble.py）
"""
import sys
import numpy as np

SAMPLE_RATE = 4_000_000           # サンプルレート [Hz]（1秒に400万サンプル）
SYMBOL_RATE = 1_000_000           # シンボルレート [Hz]（1秒に100万bit）
SPS = SAMPLE_RATE // SYMBOL_RATE  # 1bitあたりのサンプル数 = 4
# アドバタイジングチャネルの番号（37は2402MHz）
CHANNEL = 37

# アドバタイジング用のアクセスアドレス（全機器共通）
ADV_ACCESS_ADDRESS = 0x8E89BED6
# PDUヘッダの下位4bit → PDUの種類の名前
PDU_TYPES = {
    0x0: "ADV_IND",
    0x1: "ADV_DIRECT_IND",
    0x2: "ADV_NONCONN_IND",
    0x3: "SCAN_REQ",
    0x4: "SCAN_RSP",
    0x5: "CONNECT_IND",
    0x6: "ADV_SCAN_IND",
}
# AD構造の種類の番号 → 名前（よく使うものだけ）
AD_TYPES = {
    0x01: "Flags",
    0x03: "UUID16 List",
    0x08: "Short Name",
    0x09: "Complete Name",
    0x0A: "Tx Power",
    0x16: "Service Data",
    0xFF: "Manufacturer Data",
}


def load_iq(path):
    """IQファイルのパス → 複素数の配列（-1〜+1、平均0）"""
    # HackRFのファイルは符号付き8bitの I, Q, I, Q, ... の並び
    raw = np.fromfile(path, dtype=np.int8).astype(np.float32)
    x = (raw[0::2] + 1j * raw[1::2]) / 128.0  # I + jQ にし、-1〜+1 にそろえる
    return x - np.mean(x)                     # 平均（DC成分）を引いて中心を0にする


def detect_bursts(x, win=40, factor=4.0, min_len=80 * SPS):
    """IQ（複素数の配列） → 電波が出ている区間 (開始, 終了) のリスト"""
    env = np.convolve(np.abs(x), np.ones(win) / win, mode="same")  # 包絡線
    active = env > np.median(env) * factor  # 雑音レベルの4倍を超える区間
    edges = np.diff(active.astype(np.int8))  # 立ち上がり+1、立ち下がり-1
    starts = np.where(edges == 1)[0]         # 立ち上がりのサンプル番号
    ends = np.where(edges == -1)[0]          # 立ち下がりのサンプル番号
    # 記録の開始時に途中だったバーストは除く
    if len(ends) and len(starts) and ends[0] < starts[0]:
        ends = ends[1:]
    # 80bitより短い区間は雑音とみなして捨てる
    return [(s, e) for s, e in zip(starts, ends) if e - s > min_len]


def fm_demod(x):
    """IQ（複素数の配列） → 位相差 [rad] の配列（長さは1つ少ない）"""
    # 隣り合うサンプルの位相差。瞬時周波数に比例する
    return np.angle(x[1:] * np.conj(x[:-1]))


def slice_bits(freq):
    """位相差の配列 → 1bitに1つずつ判定した0/1の配列"""
    # 周波数オフセットを打ち消す（+側と-側の中心を0にそろえる）
    freq = freq - (np.percentile(freq, 90) + np.percentile(freq, 10)) / 2
    # 1シンボル内のどのサンプルで判定するかは、振れ幅が最大になる位置を選ぶ
    n = len(freq) // SPS  # bit数
    # 1行が1bit（4サンプル）の表にし、列ごとに振れ幅の平均をとる
    eye = np.abs(freq[:n * SPS].reshape(n, SPS)).mean(axis=0)
    offset = int(np.argmax(eye))  # 振れ幅がいちばん大きい列（0〜3）
    return (freq[offset::SPS] > 0).astype(np.uint8)  # 正なら1


def int_to_bits(value, n):
    """整数 → 下位ビットから並べた n 個の0/1の配列"""
    # LSB first（下位ビットから並べる）
    return np.array([(value >> i) & 1 for i in range(n)], dtype=np.uint8)


def find_access_address(bits, max_errors=2):
    """0/1の配列 → PDUの先頭の位置。見つからなければ None"""
    # プリアンブル（AAの最下位ビットが0なら 01010101）＋ AA の40bitを探す
    preamble = 0x55 if ADV_ACCESS_ADDRESS & 1 else 0xAA
    pattern = np.concatenate([int_to_bits(preamble, 8),
                              int_to_bits(ADV_ACCESS_ADDRESS, 32)])
    # 1bitずつずらし、不一致がmax_errors以下の位置を探す
    for i in range(len(bits) - len(pattern)):
        if np.count_nonzero(bits[i:i + 40] != pattern) <= max_errors:
            return i + 40  # PDUの先頭位置
    return None


def bits_to_bytes(bits):
    """0/1の配列（LSB first） → バイト列"""
    n = len(bits) // 8           # バイト数（端数のbitは捨てる）
    weights = 1 << np.arange(8)  # 各bitの重み 1, 2, 4, ..., 128（LSB first）
    # 1行が1バイト（8bit）の表にし、重みを掛けて足す
    return bytes((bits[:n * 8].reshape(n, 8) * weights).sum(axis=1).astype(np.uint8))


def dewhiten(data, channel):
    """バイト列とチャネル番号 → ホワイトニングを解除したバイト列"""
    lfsr = channel | 0x40  # x^7 + x^4 + 1 を、ビット順を反転した形で実装
    out = bytearray()
    for byte in data:
        for i in range(8):  # 1バイトの中を下位ビットから順に
            if lfsr & 1:  # 疑似乱数のビットが1なら、データを反転
                byte ^= 1 << i
                lfsr ^= 0x88  # 帰還（x^7 + x^4 + 1）
            lfsr >>= 1        # レジスタを1つ進める
        out.append(byte)
    return bytes(out)


def ble_crc(pdu, init=0x555555):
    """PDUのバイト列 → CRCの3バイト（受信したCRCと比べる）"""
    # initのビット順を反転してから、LSB firstで1bitずつ処理する
    state = int(f"{init:024b}"[::-1], 2)
    for byte in pdu:
        for _ in range(8):
            bit = (state ^ byte) & 1  # データのビットとレジスタの出力のXOR
            byte >>= 1                # 次のデータのビットへ
            state >>= 1               # レジスタを1つ進める
            if bit:
                state |= 1 << 23      # 帰還したビットを最上位に入れる
                state ^= 0x5A6000  # 生成多項式（ビット反転した形）
    return state.to_bytes(3, "little")


def read_packet(bits, pos, channel):
    """0/1の配列とPDUの先頭位置 → (PDU, CRC) のバイト列の組。無理なら None"""
    # まずヘッダの2バイトだけホワイトニングを解除して長さを知る
    header = dewhiten(bits_to_bytes(bits[pos:pos + 16]), channel)
    length = header[1] & 0x3F  # 2バイト目の下位6bitが長さ
    n = 2 + length + 3  # ヘッダ + ペイロード + CRC
    # 長さが分かったので、全体をあらためて解除する
    data = dewhiten(bits_to_bytes(bits[pos:pos + n * 8]), channel)
    if len(data) < n:   # バーストの終わりで途切れている
        return None
    return data[:2 + length], data[2 + length:n]


def parse_pdu(pdu):
    """PDUのバイト列 → 種類・アドレス・AD構造などの辞書"""
    h = pdu[0]  # ヘッダの1バイト目（種類とアドレスの種別）
    adv = {
        "type": PDU_TYPES.get(h & 0x0F, f"0x{h & 0x0F:X}"),  # 下位4bit
        "tx_random": (h >> 6) & 1,  # TxAddビット
        "length": pdu[1],                    # ペイロードの長さ [バイト]
        "adv_addr": format_addr(pdu[2:8]),   # 送信元の機器のアドレス
        "ad": [],                            # AD構造のリスト（あとで入れる）
    }
    if adv["type"] in ("SCAN_REQ", "CONNECT_IND"):
        # 送信元（スキャンする側）の後ろに、相手の機器のアドレスが続く
        adv["target"] = format_addr(pdu[8:14])
    else:
        adv["ad"] = parse_ad(pdu[8:])  # アドレスの後ろはAdvData
    return adv


def format_addr(b):
    """6バイト（LSB first） → 「AA:BB:CC:DD:EE:FF」形式の文字列"""
    # LSB firstなので、逆順に並べて表示する
    return ":".join(f"{v:02X}" for v in reversed(b))


def parse_ad(data):
    """AdvDataのバイト列 → (種類の名前, データのバイト列) のリスト"""
    ads, i = [], 0  # ads：結果、i：読んでいる位置
    # 「長さ、種類、データ」の組を先頭から順に読む
    while i + 1 < len(data) and data[i] > 0:
        # 長さは「種類＋データ」のバイト数
        length, ad_type = data[i], data[i + 1]
        ads.append((AD_TYPES.get(ad_type, f"0x{ad_type:02X}"),
                    data[i + 2:i + 1 + length]))
        i += 1 + length  # 次の組へ（長さの1バイト ＋ 長さ）
    return ads


def format_ad(name, value):
    """AD構造の名前とデータ → 表示用の文字列"""
    if name in ("Complete Name", "Short Name"):  # 名前は文字列として表示
        return value.decode("utf-8", errors="replace")
    if name == "Manufacturer Data":  # 先頭2バイトが会社ID（リトルエンディアン）
        return f"company=0x{int.from_bytes(value[:2], 'little'):04X} data={value[2:].hex()}"
    if name == "Tx Power":           # 送信電力：符号付き1バイト [dBm]
        return f"{int.from_bytes(value, 'little', signed=True)} dBm"
    return value.hex()               # それ以外は16進で表示


def main(path, channel=CHANNEL):
    x = load_iq(path)                        # Step 1：IQ（複素数の配列）
    bursts = detect_bursts(x)                # Step 2：(開始, 終了) のリスト
    print(f"bursts: {len(bursts)}")
    for s, e in bursts:
        freq = fm_demod(x[s:e])              # Step 3：位相差（実数の配列）
        bits = slice_bits(freq)              # Step 4：0/1の配列
        pos = find_access_address(bits)      # Step 5：PDUの先頭の位置
        if pos is None:
            continue
        r = read_packet(bits, pos, channel)  # Step 6：(PDU, CRC) のバイト列
        if r is None:
            continue
        pdu, crc = r
        if ble_crc(pdu) != crc:              # Step 7：CRCの検査
            continue
        adv = parse_pdu(pdu)                 # Step 8：解析結果の辞書
        addr_type = "random" if adv["tx_random"] else "public"
        # バーストの開始時刻 [ミリ秒]、PDUの種類、アドレスを表示する
        print(f"t={s / SAMPLE_RATE * 1e3:9.3f}ms {adv['type']:15s} "
              f"{adv['adv_addr']} ({addr_type}) CRC OK")
        if "target" in adv:
            print(f"    -> {adv['target']}")
        for name, value in adv["ad"]:
            print(f"    {name:18s}: {format_ad(name, value)}")


# ---- 以下は動作確認用：アドバタイジングパケットを GFSK 変調した IQ ファイルを作る ----

def build_packet(pdu_type, adv_addr, ad, channel=CHANNEL, tx_random=1):
    """PDUの種類・アドレス・AdvData → 空中を飛ぶ順の0/1の配列"""
    # ペイロード：アドレス（LSB firstなので逆順）＋ AdvData
    payload = bytes(reversed(bytes.fromhex(adv_addr.replace(":", "")))) + ad
    # ヘッダ：1バイト目が種類とTxAdd、2バイト目が長さ
    pdu = bytes([pdu_type | (tx_random << 6), len(payload)]) + payload
    air = dewhiten(pdu + ble_crc(pdu), channel)  # ホワイトニングは XOR なので同じ関数で掛けられる
    # プリアンブル ＋ アクセスアドレス ＋ PDUとCRC（すべてLSB first）
    bits = np.concatenate([int_to_bits(0xAA, 8), int_to_bits(ADV_ACCESS_ADDRESS, 32),
                           np.unpackbits(np.frombuffer(air, np.uint8), bitorder="little")])
    return bits


def gfsk_modulate(bits, bt=0.5, h=0.5):
    """0/1の配列 → GFSK変調したIQ（複素数の配列、振幅1）"""
    t = np.arange(-1.5 * SPS, 1.5 * SPS + 1) / SPS                  # 前後1.5bit分の時刻
    g = np.exp(-2 * np.pi ** 2 * bt ** 2 * t ** 2 / np.log(2))  # ガウスフィルタの形
    g /= g.sum()                             # 係数の和を1にそろえる
    nrz = np.repeat(2.0 * bits - 1, SPS)     # 0/1 を -1/+1 にし、1bitを4サンプルに
    freq = np.convolve(nrz, g, mode="same")  # ガウスフィルタで周波数の変化をなめらかに
    # 周波数を足し合わせると位相になる（変調指数 h = 0.5）
    return np.exp(1j * np.cumsum(np.pi * h * freq / SPS))


def make_test_iq(path, seconds=1.0, cfo=40e3, snr_db=20, seed=0):
    """テスト用のIQファイルを作る（3種類のパケットを6回、雑音と周波数ずれ付き）"""
    rng = np.random.default_rng(seed)
    packets = [
        build_packet(0x2, "5C:3A:91:7E:22:D4",
                     bytes.fromhex("02011a") + bytes.fromhex("0aff4c0010053b1c6e3f8a")),
        build_packet(0x0, "C4:7F:0E:31:A8:96",
                     bytes.fromhex("020106") + bytes.fromhex("0b09") + b"BLE-Sensor" +
                     bytes.fromhex("020a04")),
        build_packet(0x0, "E2:15:6B:9C:40:1F",
                     bytes.fromhex("020106") + bytes.fromhex("03030918") +
                     bytes.fromhex("051609186e09")),
    ]
    # 複素ガウス雑音（信号の振幅1に対してSNRが snr_db になる大きさ）
    x = (rng.normal(size=int(seconds * SAMPLE_RATE)) +
         1j * rng.normal(size=int(seconds * SAMPLE_RATE))) * 10 ** (-snr_db / 20) / np.sqrt(2)
    pos = 20_000  # 最初のパケットを置く位置（サンプル番号）
    for k in range(6):
        wave = gfsk_modulate(packets[k % len(packets)])
        n = np.arange(len(wave))
        # 周波数ずれ cfo [Hz] と、パケットごとに違う初期位相 k を付けて足す
        x[pos:pos + len(wave)] += wave * np.exp(1j * (2 * np.pi * cfo / SAMPLE_RATE * n + k))
        pos += 120_000 + int(rng.integers(0, 20_000))  # 30〜35ミリ秒ほど空ける
    x *= 0.4  # 8bitに収まるよう小さくする
    # HackRFと同じ、符号付き8bitの I, Q, I, Q, ... の並びで書き出す
    out = np.empty(2 * len(x), dtype=np.int8)
    out[0::2] = np.clip(np.round(x.real * 127), -128, 127)
    out[1::2] = np.clip(np.round(x.imag * 127), -128, 127)
    out.tofile(path)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--make-test":
        make_test_iq(sys.argv[2])
    else:
        main(sys.argv[1] if len(sys.argv) > 1 else "data/ble_ch37.iq")
