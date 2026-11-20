"""ADS-B (1090MHz Mode S Extended Squitter) を IQ データから復調するサンプル

hackrf_transfer -r adsb.iq -f 1090000000 -s 2000000 で記録したファイルを想定
"""
import sys
import numpy as np

SAMPLE_RATE = 2_000_000         # サンプルレート [Hz]（1秒に200万サンプル）
SPB = SAMPLE_RATE // 1_000_000  # 1bitのサンプル数 = 2（1bit = 1マイクロ秒）
HALF = SPB // 2                 # 0.5マイクロ秒あたりのサンプル数 = 1

# プリアンブル：0, 1.0, 3.5, 4.5マイクロ秒にパルス
# （0.5マイクロ秒単位で16スロット）
PREAMBLE = np.array([1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0], dtype=float)
PREAMBLE_LEN = len(PREAMBLE) * HALF  # 8マイクロ秒分のサンプル数

# 6bitの値（0〜63）を文字に変換する表。1〜26がA〜Z、32が空白、48〜57が0〜9
ADSB_CHARSET = "#ABCDEFGHIJKLMNOPQRSTUVWXYZ##### ###############0123456789######"


def load_iq(path):
    """IQファイルのパス → 複素数の配列（-1〜+1、平均0）"""
    # HackRFのファイルは符号付き8bitの I, Q, I, Q, ... の並び
    raw = np.fromfile(path, dtype=np.int8).astype(np.float32)
    x = (raw[0::2] + 1j * raw[1::2]) / 128.0  # I + jQ にし、-1〜+1 にそろえる
    return x - np.mean(x)                     # 平均（DC成分）を引いて中心を0にする


def magnitude(x):
    """IQ（複素数の配列） → 振幅（実数の配列、同じ長さ）"""
    return np.abs(x)  # 複素数の大きさ sqrt(I^2 + Q^2)


def find_preambles(mag, factor=3.0):
    """振幅の配列 → フレームの先頭位置（サンプル番号）のリスト"""
    # 0.5マイクロ秒単位のパターンをサンプル単位に引き伸ばし、平均を引く
    ref = np.repeat(PREAMBLE, HALF)
    # 平均を0にし、全体が明るいだけの区間に反応しないようにする
    ref = ref - ref.mean()
    # 相関：各位置で振幅とパターンを掛けて足す。プリアンブルと形が合うほど大きい
    corr = np.correlate(mag, ref, mode="valid")
    # 同じ長さの区間ごとの平均振幅（信号の強さ）
    level = np.convolve(mag, np.ones(len(ref)) / len(ref), mode="valid")
    # 振幅の中央値。ほとんどが無信号なので雑音の目安になる
    noise = np.median(mag)
    # 形が合い、しかも雑音より十分強い位置を候補にする
    cand = np.where((corr > factor * noise) & (level > 2 * noise))[0]
    # 近接する候補は相関が最大のものだけ残す
    starts, last = [], -10**9  # starts：結果、last：直前に採用した位置
    for i in cand:
        if i - last > PREAMBLE_LEN:  # 前の採用位置から十分離れた候補だけ見る
            # そこからプリアンブル1つ分の範囲で、相関がいちばん大きい位置
            j = i + int(np.argmax(corr[i:i + PREAMBLE_LEN]))
            starts.append(j)
            last = j
    return starts


def demod_ppm(mag, start, n_bits=112):
    """振幅の配列と先頭位置 → 112bit（0/1の配列）。足りなければ None"""
    s = start + PREAMBLE_LEN           # データ部の先頭（プリアンブルの直後）
    seg = mag[s:s + n_bits * SPB]      # データ部（112bit分）の振幅
    if len(seg) < n_bits * SPB:        # ファイルの終わりで途切れている
        return None
    seg = seg.reshape(n_bits, SPB)  # 1行が1bit（前半, 後半）の表にする
    first = seg[:, :HALF].sum(axis=1)   # 各bitの前半0.5マイクロ秒の振幅
    second = seg[:, HALF:].sum(axis=1)  # 各bitの後半0.5マイクロ秒の振幅
    return (first > second).astype(np.uint8)  # 前半にパルスがあれば1


def bits_to_int(bits):
    """0/1の配列 → 整数（先頭が上位ビット）"""
    val = 0
    for b in bits:
        val = (val << 1) | int(b)  # 左に1桁ずらして、次のbitを最下位に入れる
    return val


def bits_to_hex(bits):
    """0/1の配列 → 16進の文字列"""
    # 4bitで16進1桁。桁数をそろえ、先頭の0も残す
    return f"{bits_to_int(bits):0{len(bits) // 4}X}"


def crc24(bits):
    """112bitの配列 → CRCの余り（整数）。正しいフレームなら0"""
    # 生成多項式 0xFFF409（x^24 の項を含めると 0x1FFF409）
    GENERATOR = 0x1FFF409
    msg = bits_to_int(bits)  # 112bit全体を1つの整数にする
    n = len(bits)            # bit数（112）
    for i in range(n - 24):  # 先頭から、余り24bitの手前まで1bitずつ進む
        # 筆算の割り算：先頭が1なら、生成多項式を位置をそろえてXORする
        if msg & (1 << (n - 1 - i)):
            msg ^= GENERATOR << (n - 25 - i)
    return msg & 0xFFFFFF    # 下位24bitが余り


def decode_frame(bits):
    """112bitの配列 → 解析結果の辞書（df, icao, tc, me と、TCごとの値）"""
    msg = {
        "df": bits_to_int(bits[0:5]),      # DF：フレームの種類（17がADS-B）
        "icao": bits_to_hex(bits[8:32]),   # ICAOアドレス（機体ごとの24bit）
        "tc": bits_to_int(bits[32:37]),    # TC：MEの中身の種類
        "me": bits[32:88],                 # ME：本体の56bit
    }
    tc = msg["tc"]
    me = msg["me"]
    if 1 <= tc <= 4:        # 1〜4：便名
        msg["callsign"] = decode_callsign(me)
    elif 9 <= tc <= 18:     # 9〜18：高度と位置
        msg.update(decode_airborne_position(me))
    elif tc == 19:          # 19：速度
        msg.update(decode_velocity(me))
    return msg


def decode_callsign(me):
    """ME（56bitの配列） → 便名の文字列"""
    # TCとカテゴリの8bitのあと、6bitずつ8文字が並ぶ
    chars = [ADSB_CHARSET[bits_to_int(me[8 + i * 6:14 + i * 6])] for i in range(8)]
    return "".join(chars).replace("#", "").strip()  # 未使用の文字と空白を除く


def decode_airborne_position(me):
    """ME → 高度 [ft]、偶数/奇数の別、CPRの緯度・経度（0〜1）の辞書"""
    alt_bits = list(me[8:20])  # 高度の12bit（途中にQビットを含む）
    q = alt_bits.pop(7)  # Qビット：1なら25ft単位
    # Qビットを抜いた11bitの値を25倍し、1000を引くと高度 [ft] になる
    alt = bits_to_int(alt_bits) * 25 - 1000 if q else None
    return {
        "altitude": alt,
        # 0：偶数形式、1：奇数形式
        "cpr_odd": int(me[21]),
        # 帯の中の緯度（17bit → 0〜1）
        "cpr_lat": bits_to_int(me[22:39]) / 131072,
        # 帯の中の経度（17bit → 0〜1）
        "cpr_lon": bits_to_int(me[39:56]) / 131072,
    }


def decode_velocity(me):
    """ME → 速さ [kt]、進行方向 [度]、昇降率 [ft/分] の辞書"""
    st = bits_to_int(me[5:8])  # サブタイプ1, 2が対地速度
    if st not in (1, 2):
        return {}
    v_ew = bits_to_int(me[14:24]) - 1  # 「速さ+1」の値（0は情報なし）
    v_ns = bits_to_int(me[25:35]) - 1  # 南北方向も同じ
    if me[13]:  # 方向ビット：1なら西向き
        v_ew = -v_ew
    if me[24]:  # 方向ビット：1なら南向き
        v_ns = -v_ns
    vr = (bits_to_int(me[37:46]) - 1) * 64  # 昇降率：64ft/分 単位
    if me[36]:  # 符号ビット：1なら降下
        vr = -vr
    return {
        # 東西・南北の成分を合わせた速さ
        "speed": float(np.hypot(v_ew, v_ns)),
        # 北から時計回りの角度
        "heading": float(np.degrees(np.arctan2(v_ew, v_ns)) % 360),
        "vertical_rate": vr,
    }


NZ = 15  # 赤道から極までの緯度帯の数（CPRの決まり）


def cpr_nl(lat):
    """緯度 [度] → その緯度での経度方向の帯の数（整数）"""
    if abs(lat) >= 87.0:  # 極の近くは帯が1つだけ
        return 1
    # 高緯度ほど経線の間隔が狭いので、帯の数が減る（CPRで決まった式）
    a = 1 - np.cos(np.pi / (2 * NZ))
    b = np.cos(np.radians(abs(lat))) ** 2
    return int(np.floor(2 * np.pi / np.arccos(1 - a / b)))


def cpr_global(even, odd, newest_odd):
    """偶数・奇数の (lat, lon)（0〜1） → (緯度, 経度) [度]。無理なら None"""
    lat_e, lon_e = even  # 偶数形式の帯の中の位置
    lat_o, lon_o = odd   # 奇数形式の帯の中の位置
    # 緯度：どの帯にいるかを求めてから、帯の中の位置を足す
    # 帯の番号 j：偶数（60帯）と奇数（59帯）の区切りのずれから求める
    j = np.floor(59 * lat_e - 60 * lat_o + 0.5)
    lat0 = 360 / 60 * ((j % 60) + lat_e)  # 偶数形式：1帯 = 6度
    lat1 = 360 / 59 * ((j % 59) + lat_o)  # 奇数形式：1帯 = 360/59度
    lat0 = lat0 - 360 if lat0 >= 270 else lat0  # 南半球は負の緯度にする
    lat1 = lat1 - 360 if lat1 >= 270 else lat1
    if cpr_nl(lat0) != cpr_nl(lat1):
        return None  # 偶数・奇数で緯度帯が違う場合は計算できない
    # 経度：その緯度での帯の数を使って同様に計算する
    lat = lat1 if newest_odd else lat0  # 新しいほうのメッセージを使う
    nl = cpr_nl(lat)                    # この緯度での経度方向の帯の数
    m = np.floor(lon_e * (nl - 1) - lon_o * nl + 0.5)  # 帯の数は nl と nl-1
    if newest_odd:
        ni = max(nl - 1, 1)                   # 奇数形式の帯の数
        lon = 360 / ni * ((m % ni) + lon_o)   # 帯の番号 + 帯の中の位置
    else:
        ni = max(nl, 1)                       # 偶数形式の帯の数
        lon = 360 / ni * ((m % ni) + lon_e)
    lon = lon - 360 if lon >= 180 else lon  # 西経は負の経度にする
    return float(lat), float(lon)


def update_aircraft(aircraft, msg, t):
    """decode_frame の結果 msg を、ICAOアドレスごとの辞書 aircraft に足す"""
    # 初めて見る機体なら、空の航跡で登録する
    ac = aircraft.setdefault(msg["icao"], {"track": []})
    if "callsign" in msg:
        ac["callsign"] = msg["callsign"]
    if "speed" in msg:
        ac["speed"], ac["heading"] = msg["speed"], msg["heading"]
    if "cpr_lat" in msg:
        if msg["altitude"] is not None:
            ac["altitude"] = msg["altitude"]
        # 偶数・奇数の最新のものを、受信時刻 t [秒] と一緒に覚えておく
        key = "odd" if msg["cpr_odd"] else "even"
        ac[key] = (msg["cpr_lat"], msg["cpr_lon"], t)
        # 両方そろい、受信時刻の差が10秒未満なら位置を計算する
        if "odd" in ac and "even" in ac and abs(ac["odd"][2] - ac["even"][2]) < 10:
            pos = cpr_global(ac["even"][:2], ac["odd"][:2], newest_odd=msg["cpr_odd"])
            if pos:
                ac["track"].append(pos)  # 航跡に (緯度, 経度) を足す


def plot_map(aircraft, path="adsb_map.png"):
    """aircraft の航跡を地図に描き、PNGで保存する"""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 8))
    for icao, ac in aircraft.items():
        if not ac["track"]:  # 位置が1つも求まっていない機体は描かない
            continue
        # (緯度, 経度) の列を、緯度と経度の配列に分ける
        lat, lon = np.array(ac["track"]).T
        ax.plot(lon, lat, ".-")        # 横が経度、縦が緯度
        # 最後の位置に便名（なければICAOアドレス）を書く
        ax.annotate(ac.get("callsign", icao), (lon[-1], lat[-1]))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    # 経度1度の長さは緯度によって縮むので、縦横比で補正する
    ax.set_aspect(1 / np.cos(np.radians(np.mean(ax.get_ylim()))))
    fig.savefig(path, dpi=150)


def main(path):
    x = load_iq(path)             # Step 1：IQ（複素数の配列）
    mag = magnitude(x)            # Step 2：振幅（実数の配列）
    starts = find_preambles(mag)  # Step 3：フレームの先頭位置のリスト
    print(f"preamble candidates: {len(starts)}")
    aircraft, n_ok = {}, 0        # aircraft：ICAOアドレス → 航空機の情報の辞書
    for s in starts:
        bits = demod_ppm(mag, s)  # Step 4：112bit（0/1の配列）
        # Step 5：DFが17か18で、CRCの余りが0のものだけを使う
        if bits is None or bits_to_int(bits[0:5]) not in (17, 18) or crc24(bits) != 0:
            continue
        n_ok += 1                 # 正しく受信できたフレームの数
        msg = decode_frame(bits)  # Step 6〜9：各フィールドを取り出した辞書
        update_aircraft(aircraft, msg, s / SAMPLE_RATE)  # Step 10〜11：集計
    print(f"CRC OK: {n_ok}")
    # 位置が求まった機体ごとに、最後の状態を1行で表示する
    for icao, ac in aircraft.items():
        if not ac["track"]:
            continue
        lat, lon = ac["track"][-1]
        print(f"{icao}  {ac.get('callsign', '-'):8s} {ac.get('altitude', 0):5d}ft "
              f"{ac.get('speed', 0):4.0f}kt {ac.get('heading', 0):4.0f}deg  "
              f"({lat:.5f}, {lon:.5f})  points={len(ac['track'])}")
    plot_map(aircraft)            # Step 11：航跡を地図に描く


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/adsb.iq")
