"""IEEE 802.15.4 (2.4GHz O-QPSK) を IQ データから復調するサンプル

hackrf_transfer -r zigbee_ch20.iq -f 2450000000 -s 4000000 で記録したファイルを想定
"""
import sys
import numpy as np

SAMPLE_RATE = 4_000_000         # サンプルレート [Hz]（1秒に400万サンプル）
CHIP_RATE = 2_000_000           # チップレート [Hz]（1秒に200万チップ）
SPC = SAMPLE_RATE // CHIP_RATE  # 1チップあたりのサンプル数 = 2

# IEEE 802.15.4 2.4GHzのシンボル → チップ対応表（左端がc0）
CHIP_SEQS = [
    "11011001110000110101001000101110",  # 0x0
    "11101101100111000011010100100010",  # 0x1
    "00101110110110011100001101010010",  # 0x2
    "00100010111011011001110000110101",  # 0x3
    "01010010001011101101100111000011",  # 0x4
    "00110101001000101110110110011100",  # 0x5
    "11000011010100100010111011011001",  # 0x6
    "10011100001101010010001011101101",  # 0x7
    "10001100100101100000011101111011",  # 0x8
    "10111000110010010110000001110111",  # 0x9
    "01111011100011001001011000000111",  # 0xA
    "01110111101110001100100101100000",  # 0xB
    "00000111011110111000110010010110",  # 0xC
    "01100000011101111011100011001001",  # 0xD
    "10010110000001110111101110001100",  # 0xE
    "11001001011000000111011110111000",  # 0xF
]
# 文字の"1"/"0"を+1/-1の数に変える（16行32列の配列）
CHIPS = np.array([[1 if c == "1" else -1 for c in s] for s in CHIP_SEQS])


def load_iq(path):
    """IQファイルのパス → 複素数の配列（-1〜+1、平均0）"""
    # HackRFのファイルは符号付き8bitのI, Q, I, Q, ...の並び
    raw = np.fromfile(path, dtype=np.int8).astype(np.float32)
    x = (raw[0::2] + 1j * raw[1::2]) / 128.0  # I+jQにし、-1〜+1にそろえる
    return x - np.mean(x)                     # 平均（DC成分）を引いて中心を0にする


def detect_bursts(x, win=200, factor=4.0, min_len=800):
    """IQ（複素数の配列） → 電波が出ている区間 (開始, 終了) のリスト"""
    env = np.convolve(np.abs(x), np.ones(win) / win, mode="same")  # 包絡線
    active = env > np.median(env) * factor   # 雑音レベルの4倍を超える区間
    edges = np.diff(active.astype(np.int8))  # 立ち上がり+1、立ち下がり-1
    starts = np.where(edges == 1)[0]         # 立ち上がりのサンプル番号
    ends = np.where(edges == -1)[0]          # 立ち下がりのサンプル番号
    if len(ends) and len(starts) and ends[0] < starts[0]:
        ends = ends[1:]  # 記録の開始時に途中だったバーストは除く
    # 短すぎる区間（800サンプル以下）は雑音とみなして捨てる
    return [(s, e) for s, e in zip(starts, ends) if e - s > min_len]


def oqpsk_modulate(chips):
    """チップ列（0/1の配列） → O-QPSKの送信波形（複素数の配列）"""
    # 偶数チップをI、奇数チップをQに載せ、half-sineパルスで整形する
    # 2チップ分の山形のパルス
    pulse = np.sin(np.pi * np.arange(2 * SPC) / (2 * SPC))
    # I成分（最後のパルスがはみ出す分も確保）
    i = np.zeros(len(chips) * SPC + 2 * SPC)
    q = np.zeros_like(i)                      # Q成分
    for k, c in enumerate(chips):
        start = k * SPC  # QはIに対して1チップ遅れる
        if k % 2 == 0:
            i[start:start + 2 * SPC] += c * pulse
        else:
            q[start:start + 2 * SPC] += c * pulse
    return i + 1j * q


def fm_demod(x):
    """IQ（複素数の配列） → 位相差 [rad] の配列（長さは1つ少ない）"""
    # 隣り合うサンプルの位相差。瞬時周波数に比例する
    return np.angle(x[1:] * np.conj(x[:-1]))


def make_templates():
    """（入力なし） → 16シンボル分の参照パターン（+1/-1、16行32列の配列）"""
    # 送信波形を作り、受信側と同じ方法で「位相差の符号列」に変換する
    # 先頭の2シンボルは、直前のシンボルの影響をそろえるための前置き
    wave = oqpsk_modulate(np.concatenate([CHIPS[0], CHIPS[0], *CHIPS]))
    templates = []
    for sym in range(16):
        start = (2 + sym) * 32 * SPC  # シンボルsymの波形の開始位置
        d = fm_demod(wave[start - 1:start + 32 * SPC])
        # 1チップ分（2サンプル）の位相差を足して、その符号をとる
        templates.append(np.sign(d[0::2] + d[1::2]))
    return np.array(templates)


TEMPLATES = make_templates()  # 16行32列：シンボルごとの参照パターン
# SHR：プリアンブル（シンボル0が8個）＋ SFD（0xA7を下位4bitから：0x7, 0xA）
SHR = np.concatenate([TEMPLATES[s] for s in [0] * 8 + [0x7, 0xA]])


def to_chip_signs(burst, offset):
    """バーストのIQと区切り位置（0か1） → チップごとの符号（+1/-1）"""
    d = fm_demod(burst)
    d = d - np.median(d)  # 周波数オフセットを打ち消す
    n = (len(d) - offset) // 2  # チップ数
    # offsetから2サンプルずつ足して、1チップ分の位相差にする
    f = d[offset:offset + 2 * n:2] + d[offset + 1:offset + 2 * n:2]
    return np.sign(f)


def find_shr(burst):
    """バーストのIQ → (一致度, 符号の配列, PHRの開始位置) の組"""
    best = (-1, None, None)  # これまでで一致度がいちばん高かったもの
    for offset in range(SPC):  # チップの区切りを0, 1サンプル目の両方で試す
        signs = to_chip_signs(burst, offset)
        # 相関：符号の列とSHRのパターンがどれだけ一致するか（最大は len(SHR)）
        corr = np.correlate(signs, SHR, mode="valid")
        k = int(np.argmax(corr))  # いちばん一致した位置（SHRの先頭）
        if corr[k] > best[0]:
            best = (corr[k], signs, k + len(SHR))  # SHRの直後がPHR
    return best


def despread(signs, pos, n_symbols):
    """符号の配列、開始位置、シンボル数 → (シンボルのリスト, 次の位置)"""
    symbols = []
    for _ in range(n_symbols):
        window = signs[pos:pos + 32]  # 1シンボル = 32チップ
        if len(window) < 32:          # バーストの終わりで途切れている
            break
        # 16個の参照パターンとの内積のうち、いちばん大きいものがそのシンボル
        symbols.append(int(np.argmax(TEMPLATES @ window)))
        pos += 32
    return symbols, pos


def symbols_to_bytes(symbols):
    """シンボル（0〜15）のリスト → バイト列"""
    # 1バイト = 下位4bit → 上位4bitの順に2シンボル
    return bytes(lo | (hi << 4) for lo, hi in zip(symbols[0::2], symbols[1::2]))


def read_ppdu(signs, pos):
    """符号の配列とPHRの開始位置 → PSDUのバイト列"""
    phr, pos = despread(signs, pos, 2)        # PHR：1バイト = 2シンボル
    length = symbols_to_bytes(phr)[0] & 0x7F  # 下位7bitがPSDUの長さ [バイト]
    psdu_syms, _ = despread(signs, pos, length * 2)  # 1バイトにつき2シンボル
    return symbols_to_bytes(psdu_syms)


def crc16_kermit(data):
    """バイト列 → CRC-16の値（整数）"""
    crc = 0x0000  # 初期値は0
    for b in data:
        crc ^= b  # 1バイトを下位8bitにXORする
        for _ in range(8):
            # 最下位bitが1なら、
            # 右にずらして生成多項式（ビット反転した形）をXOR
            crc = (crc >> 1) ^ 0x8408 if crc & 1 else crc >> 1
    return crc


def check_fcs(psdu):
    """PSDUのバイト列 → 末尾2バイトのFCSが合っていれば True"""
    # FCSを除いた部分のCRCと、末尾2バイト（リトルエンディアン）を比べる
    return (len(psdu) >= 3 and
            crc16_kermit(psdu[:-2]) == int.from_bytes(psdu[-2:], "little"))


def parse_mac(psdu):
    """PSDUのバイト列 → MACヘッダの各フィールドとペイロードの辞書"""
    # フレーム制御フィールド（16bit）
    fcf = int.from_bytes(psdu[0:2], "little")
    mac = {
        # 0：ビーコン、1：データ、2：ACK、3：コマンド
        "frame_type": fcf & 0x7,
        "security": (fcf >> 3) & 1,      # 1ならMAC層で暗号化
        "ack_request": (fcf >> 5) & 1,   # 1なら受信側にACKを求める
        "pan_id_comp": (fcf >> 6) & 1,   # 1なら送信元のPAN IDを省略
        # 宛先アドレスの形式（2：短縮16bit、3：64bit）
        "dst_mode": (fcf >> 10) & 0x3,
        "src_mode": (fcf >> 14) & 0x3,   # 送信元アドレスの形式
        "seq": psdu[2],                  # シーケンス番号
    }
    # ここでは「短縮アドレス + PAN ID圧縮」の典型的なデータフレームのみ扱う
    mac["dst_pan"] = int.from_bytes(psdu[3:5], "little")   # 宛先のPAN ID
    # 宛先の短縮アドレス
    mac["dst_addr"] = int.from_bytes(psdu[5:7], "little")
    # 送信元の短縮アドレス
    mac["src_addr"] = int.from_bytes(psdu[7:9], "little")
    # 末尾のFCS 2バイトを除く
    mac["payload"] = psdu[9:-2]
    return mac


def parse_nwk(payload):
    """MACのペイロード → NWKヘッダの各フィールドと暗号化部分の辞書"""
    fcf = int.from_bytes(payload[0:2], "little")  # NWKのフレーム制御（16bit）
    nwk = {
        "frame_type": fcf & 0x3,                 # 0：データ、1：コマンド
        "protocol_version": (fcf >> 2) & 0xF,    # Zigbee PROなら2
        "security": (fcf >> 9) & 1,              # 1ならNWK層で暗号化
        # 1なら送信元の64bitアドレスあり
        "src_ieee_present": (fcf >> 12) & 1,
        "dst_addr": int.from_bytes(payload[2:4], "little"),  # 最終的な宛先
        "src_addr": int.from_bytes(payload[4:6], "little"),  # 最初の送信元
        "radius": payload[6],                    # あと何回中継できるか
        "seq": payload[7],                       # NWKのシーケンス番号
    }
    pos = 8  # ここから先は、フラグによってあるかないかが変わる
    if nwk["src_ieee_present"]:
        nwk["src_ieee"] = int.from_bytes(payload[pos:pos + 8], "little")
        pos += 8
    if nwk["security"]:
        # フレームカウンタ：送るたびに増える番号（制御1バイトの次の4バイト）
        nwk["frame_counter"] = int.from_bytes(payload[pos + 1:pos + 5], "little")
        # 補助セキュリティヘッダ（制御1 + カウンタ4 + 送信元8 + 鍵番号1）
        pos += 14
    # 残りは暗号化されたペイロード（と認証コード）
    nwk["encrypted"] = payload[pos:]
    return nwk


def main(path):
    x = load_iq(path)                       # Step 1：IQ（複素数の配列）
    bursts = detect_bursts(x)               # Step 2：(開始, 終了) のリスト
    print(f"bursts: {len(bursts)}")
    for s, e in bursts:
        burst = x[max(0, s - 400):e + 400]  # 前後に少し余裕を持たせる
        # Step 3〜5：参照パターンとの相関でSHRを探す
        score, signs, pos = find_shr(burst)
        if score < 0.6 * len(SHR):          # IEEE 802.15.4のパケットではない
            continue
        psdu = read_ppdu(signs, pos)        # Step 6〜7：PSDUのバイト列
        if not check_fcs(psdu):             # Step 8：FCSの検査
            print(f"sample={s} SHR score={score} FCS NG")
            continue
        print(f"sample={s} SHR score={score} len={len(psdu)} FCS OK")
        print("  PSDU:", psdu.hex(" "))
        mac = parse_mac(psdu)               # Step 9：MACヘッダの辞書
        # parse_mac/parse_nwk が想定するのは短縮アドレスのデータフレームのみ（ACK などは飛ばす）
        if mac["frame_type"] != 1 or mac["dst_mode"] != 2 or len(mac["payload"]) < 8:
            print(f"  MAC: type={mac['frame_type']} seq={mac['seq']} (データフレーム以外のため解析を省略)")
            continue
        nwk = parse_nwk(mac["payload"])     # Step 10：NWKヘッダの辞書
        # MACとNWKの主なフィールドを表示する
        print(f"  MAC: type={mac['frame_type']} seq={mac['seq']} "
              f"PAN=0x{mac['dst_pan']:04X} dst=0x{mac['dst_addr']:04X} src=0x{mac['src_addr']:04X}")
        print(f"  NWK: type={nwk['frame_type']} ver={nwk['protocol_version']} "
              f"dst=0x{nwk['dst_addr']:04X} src=0x{nwk['src_addr']:04X} radius={nwk['radius']} "
              f"seq={nwk['seq']} "
              + (f"ieee=0x{nwk['src_ieee']:016X} " if "src_ieee" in nwk else "")
              + f"counter={nwk.get('frame_counter')} encrypted={nwk['encrypted'].hex(' ')}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/zigbee_ch20.iq")
