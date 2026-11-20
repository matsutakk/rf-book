"""Thread (IEEE 802.15.4 + 6LoWPAN) のフレームを IQ データから読み解くサンプル

物理層（SHR 検出・逆拡散・FCS 検査）は zigbee.py と共通。
hackrf_transfer -r thread_ch11.iq -f 2405000000 -s 4000000 で記録したファイルを想定

  python thread.py data/thread_ch11.iq
  python thread.py --test   # テスト用 IQ を生成して復号する
"""
import sys
import numpy as np

from zigbee import (CHIPS, SAMPLE_RATE, check_fcs, crc16_kermit, detect_bursts,
                    find_shr, load_iq, oqpsk_modulate, read_ppdu, SHR)

try:  # 鍵 ID モード 2 の復号に使う（pip install cryptography）。なくても他の処理は動く
    from cryptography.exceptions import InvalidTag
    from cryptography.hazmat.primitives.ciphers.aead import AESCCM
except ImportError:
    AESCCM = None

MLE_PORT = 19788
# 鍵 ID モード 2（MLE Announce など）用に Thread の規格で公開されている鍵と拡張アドレス
MODE2_KEY = bytes.fromhex("78581686fdb4580fb092546aecbd1566")
MODE2_EXT_ADDR = bytes.fromhex("3506feb823d48712")
FRAME_TYPES = {0: "Beacon", 1: "Data", 2: "Ack", 3: "Command"}
MAC_COMMANDS = {0x01: "Association Request", 0x02: "Association Response",
                0x04: "Data Request", 0x07: "Beacon Request"}
MLE_COMMANDS = {0: "Link Request", 1: "Link Accept", 2: "Link Accept And Request",
                3: "Link Reject", 4: "Advertisement", 5: "Update", 6: "Update Request",
                7: "Data Request", 8: "Data Response", 9: "Parent Request",
                10: "Parent Response", 11: "Child ID Request", 12: "Child ID Response",
                13: "Child Update Request", 14: "Child Update Response",
                15: "Announce", 16: "Discovery Request", 17: "Discovery Response"}
MIC_LEN = {0: 0, 1: 4, 2: 8, 3: 16, 4: 0, 5: 4, 6: 8, 7: 16}
KEY_ID_LEN = {0: 0, 1: 1, 2: 5, 3: 9}


# ---------------------------------------------------------------- MAC 層

def ext_addr(b):
    # 拡張アドレス（8 バイト）は下位バイトから送られるので、逆順にして表示する
    return ":".join(f"{v:02x}" for v in b[::-1])


def pan_id_fields(version, dst_mode, src_mode, comp):
    """宛先 PAN ID・送信元 PAN ID がそれぞれヘッダに入っているかを返す"""
    if version < 2:  # 2003/2006 形式
        dst = dst_mode != 0
        src = src_mode != 0 and not (comp and dst_mode != 0)
        return dst, src
    # 2015 形式（規格の表を簡略化したもの）
    if dst_mode == 0 and src_mode == 0:
        return comp, False
    if src_mode == 0:
        return not comp, False
    if dst_mode == 0:
        return False, not comp
    if dst_mode == 3 and src_mode == 3:
        return not comp, False
    return True, not comp


def parse_mac(psdu):
    fcf = int.from_bytes(psdu[0:2], "little")
    mac = {
        "frame_type": fcf & 0x7,
        "security": (fcf >> 3) & 1,
        "pending": (fcf >> 4) & 1,
        "ack_request": (fcf >> 5) & 1,
        "pan_id_comp": (fcf >> 6) & 1,
        "seq_suppress": (fcf >> 8) & 1,
        "ie_present": (fcf >> 9) & 1,
        "dst_mode": (fcf >> 10) & 0x3,
        "version": (fcf >> 12) & 0x3,
        "src_mode": (fcf >> 14) & 0x3,
    }
    pos = 2
    if not (mac["version"] == 2 and mac["seq_suppress"]):
        mac["seq"] = psdu[pos]
        pos += 1

    has_dst_pan, has_src_pan = pan_id_fields(mac["version"], mac["dst_mode"],
                                             mac["src_mode"], mac["pan_id_comp"])
    if has_dst_pan:
        mac["dst_pan"] = int.from_bytes(psdu[pos:pos + 2], "little")
        pos += 2
    pos = read_address(psdu, pos, mac["dst_mode"], mac, "dst")
    if has_src_pan:
        mac["src_pan"] = int.from_bytes(psdu[pos:pos + 2], "little")
        pos += 2
    pos = read_address(psdu, pos, mac["src_mode"], mac, "src")

    end = len(psdu) - 2  # FCS を除く
    if mac["security"]:
        pos = read_aux_security(psdu, pos, mac)
        end -= MIC_LEN[mac["sec_level"]]
    if mac["ie_present"]:
        pos = read_header_ies(psdu, pos, end, mac)
    mac["header"] = psdu[:pos]  # 暗号化の検証に使う（認証のみの範囲）
    mac["payload"] = psdu[pos:end]
    mac["mic"] = psdu[end:-2]
    return mac


def read_address(psdu, pos, mode, mac, key):
    if mode == 2:  # 短縮アドレス
        mac[key] = f"0x{int.from_bytes(psdu[pos:pos + 2], 'little'):04x}"
        mac[key + "_raw"] = psdu[pos:pos + 2]
        return pos + 2
    if mode == 3:  # 拡張アドレス
        mac[key] = ext_addr(psdu[pos:pos + 8])
        mac[key + "_raw"] = psdu[pos:pos + 8]
        return pos + 8
    return pos


def read_aux_security(psdu, pos, mac):
    ctrl = psdu[pos]
    mac["sec_level"] = ctrl & 0x7
    mac["key_id_mode"] = (ctrl >> 3) & 0x3
    counter_suppress = mac["version"] == 2 and (ctrl >> 5) & 1
    pos += 1
    if not counter_suppress:
        mac["frame_counter"] = int.from_bytes(psdu[pos:pos + 4], "little")
        pos += 4
    key_id = psdu[pos:pos + KEY_ID_LEN[mac["key_id_mode"]]]
    if key_id:
        mac["key_source"] = key_id[:-1]
        mac["key_index"] = key_id[-1]
    return pos + len(key_id)


def decrypt_mode2(mac):
    """鍵 ID モード 2 のフレームを、Thread の規格で決まっている既知の鍵で復号する

    ノンス = 拡張アドレス（固定値）8B + フレームカウンタ 4B（ビッグエンディアン）+ セキュリティレベル 1B
    """
    if AESCCM is None or mac.get("key_id_mode") != 2 or mac.get("key_source") != b"\xff" * 4:
        return None
    nonce = MODE2_EXT_ADDR + mac["frame_counter"].to_bytes(4, "big") + bytes([mac["sec_level"]])
    try:
        return AESCCM(MODE2_KEY, tag_length=len(mac["mic"])).decrypt(
            nonce, mac["payload"] + mac["mic"], mac["header"])
    except InvalidTag:
        return None


def read_header_ies(psdu, pos, end, mac):
    # Header IE: 2 バイトの記述子（長さ 7bit・ID 8bit・種別 1bit）+ 内容
    mac["header_ies"] = []
    while pos + 2 <= end:
        desc = int.from_bytes(psdu[pos:pos + 2], "little")
        length, element_id = desc & 0x7F, (desc >> 7) & 0xFF
        pos += 2
        if element_id in (0x7E, 0x7F):  # 終端（この後は Payload IE または本体）
            break
        mac["header_ies"].append((element_id, psdu[pos:pos + length]))
        pos += length
    return pos


# ---------------------------------------------------------------- 6LoWPAN / UDP / MLE

def iid_from_mac(raw):
    # MAC アドレスから IPv6 のインターフェース識別子（下位 64bit）を作る
    if len(raw) == 8:
        iid = bytearray(raw[::-1])
        iid[0] ^= 0x02  # U/L ビットを反転
        return bytes(iid)
    return bytes([0, 0, 0, 0xff, 0xfe, 0]) + raw[::-1]


def ipv6_str(b):
    words = [int.from_bytes(b[i:i + 2], "big") for i in range(0, 16, 2)]
    # 連続する 0 の並びのうち最長のものを "::" にまとめる
    best, run = (0, 0), (0, 0)
    for i, w in enumerate(words + [1]):
        if w == 0:
            run = (run[0], run[1] + 1) if run[1] else (i, 1)
        else:
            best = max(best, run, key=lambda r: r[1])
            run = (0, 0)
    head = ":".join(f"{w:x}" for w in words[:best[0]])
    tail = ":".join(f"{w:x}" for w in words[best[0] + best[1]:])
    return f"{head}::{tail}" if best[1] > 1 else ":".join(f"{w:x}" for w in words)


def unicast_address(p, pos, mode, ctx, mac_raw):
    prefix = bytes([0xfe, 0x80]) + bytes(6) if not ctx else bytes(8)
    if mode == 0:  # 128bit をそのまま
        return p[pos:pos + 16], pos + 16
    if mode == 1:  # 下位 64bit
        return prefix + p[pos:pos + 8], pos + 8
    if mode == 2:  # 下位 16bit
        return prefix + bytes([0, 0, 0, 0xff, 0xfe, 0]) + p[pos:pos + 2], pos + 2
    return prefix + iid_from_mac(mac_raw), pos  # MAC アドレスから復元


def multicast_address(p, pos, mode):
    if mode == 0:
        return p[pos:pos + 16], pos + 16
    if mode == 1:  # ffXX::00XX:XXXX:XXXX
        return bytes([0xff, p[pos]]) + bytes(9) + p[pos + 1:pos + 6], pos + 6
    if mode == 2:  # ffXX::00XX:XXXX
        return bytes([0xff, p[pos]]) + bytes(11) + p[pos + 1:pos + 4], pos + 4
    return bytes([0xff, 0x02]) + bytes(13) + p[pos:pos + 1], pos + 1  # ff02::XX


def parse_iphc(p, mac):
    """6LoWPAN の IPHC（圧縮した IPv6 ヘッダ）を展開する"""
    iphc = int.from_bytes(p[0:2], "big")
    tf, nh, hlim = (iphc >> 11) & 3, (iphc >> 10) & 1, (iphc >> 8) & 3
    cid, sac, sam = (iphc >> 7) & 1, (iphc >> 6) & 1, (iphc >> 4) & 3
    m, dac, dam = (iphc >> 3) & 1, (iphc >> 2) & 1, iphc & 3
    pos = 2 + cid
    pos += {0: 4, 1: 3, 2: 1, 3: 0}[tf]  # トラフィッククラス・フローラベル
    ip = {}
    if not nh:
        ip["next_header"] = p[pos]
        pos += 1
    if hlim == 0:
        ip["hop_limit"] = p[pos]
        pos += 1
    else:
        ip["hop_limit"] = {1: 1, 2: 64, 3: 255}[hlim]

    src, pos = unicast_address(p, pos, sam, sac, mac.get("src_raw", b""))
    if m:
        dst, pos = multicast_address(p, pos, dam)
    else:
        dst, pos = unicast_address(p, pos, dam, dac, mac.get("dst_raw", b""))
    ip["src"], ip["dst"] = ipv6_str(src), ipv6_str(dst)

    if nh and p[pos] & 0xF8 == 0xF0:  # UDP の圧縮ヘッダ（NHC）
        ip["next_header"] = 17
        pos = parse_udp_nhc(p, pos, ip)
    elif ip.get("next_header") == 17:
        ip["src_port"] = int.from_bytes(p[pos:pos + 2], "big")
        ip["dst_port"] = int.from_bytes(p[pos + 2:pos + 4], "big")
        pos += 8
    ip["payload"] = p[pos:]
    return ip


def parse_udp_nhc(p, pos, ip):
    nhc = p[pos]
    ports, c = nhc & 3, (nhc >> 2) & 1
    pos += 1
    if ports == 0:
        ip["src_port"] = int.from_bytes(p[pos:pos + 2], "big")
        ip["dst_port"] = int.from_bytes(p[pos + 2:pos + 4], "big")
        pos += 4
    elif ports == 1:
        ip["src_port"] = int.from_bytes(p[pos:pos + 2], "big")
        ip["dst_port"] = 0xF000 | p[pos + 2]
        pos += 3
    elif ports == 2:
        ip["src_port"] = 0xF000 | p[pos]
        ip["dst_port"] = int.from_bytes(p[pos + 1:pos + 3], "big")
        pos += 3
    else:
        ip["src_port"] = 0xF0B0 | (p[pos] >> 4)
        ip["dst_port"] = 0xF0B0 | (p[pos] & 0xF)
        pos += 1
    return pos if c else pos + 2  # チェックサム


def parse_lowpan(p, mac):
    """MAC ペイロードの先頭（ディスパッチ）を見て 6LoWPAN のヘッダを順に外す"""
    info = {}
    if p and p[0] & 0xC0 == 0x80:  # メッシュヘッダ
        v, f = (p[0] >> 5) & 1, (p[0] >> 4) & 1
        p = p[1 + (2 if v else 8) + (2 if f else 8):]
        info["mesh"] = True
    if p and p[0] & 0xF8 == 0xC0:  # 最初の断片（FRAG1）
        info["frag"] = f"FRAG1 size={int.from_bytes(p[0:2], 'big') & 0x7FF} tag=0x{p[2] << 8 | p[3]:04x}"
        p = p[4:]
    elif p and p[0] & 0xF8 == 0xE0:  # 2 番目以降の断片（FRAGN）は IPHC を含まない
        info["frag"] = f"FRAGN size={int.from_bytes(p[0:2], 'big') & 0x7FF} tag=0x{p[2] << 8 | p[3]:04x} offset={p[4] * 8}"
        return info
    if p and p[0] & 0xE0 == 0x60:  # IPHC
        info["ip"] = parse_iphc(p, mac)
    return info


def parse_mle(p):
    if p[0] == 0xFF:  # 暗号化なし（Discovery など）
        return {"secured": False, "command": MLE_COMMANDS.get(p[1], f"0x{p[1]:02x}")}
    if p[0] == 0x00:  # 暗号化あり：補助セキュリティヘッダだけ読める
        ctrl = p[1]
        return {"secured": True, "frame_counter": int.from_bytes(p[2:6], "little"),
                "key_id_mode": (ctrl >> 3) & 3}
    return {}


def parse_beacon(p):
    # Thread 1.1 のビーコン：スーパーフレーム仕様 2B・GTS 1B・保留アドレス 1B の後に Thread 独自の情報
    body = p[4:]
    if len(body) < 26 or body[0] != 0x03:
        return {}
    return {"version": body[1] >> 4, "joinable": body[1] & 1,
            "network_name": body[2:18].rstrip(b"\x00").decode(errors="replace"),
            "ext_pan_id": body[18:26].hex()}


# ---------------------------------------------------------------- 判定と表示

def describe(mac):
    """1 フレームの中身を 1 行で表す"""
    kind = FRAME_TYPES.get(mac["frame_type"], str(mac["frame_type"]))
    if mac["frame_type"] == 2:
        return kind
    if not mac["security"]:
        return describe_payload(mac, kind, mac["payload"])
    sec = f"MAC 暗号化 level={mac['sec_level']} key_id_mode={mac['key_id_mode']} counter={mac.get('frame_counter')}"
    plain = decrypt_mode2(mac)
    if plain is None:
        return f"{kind} ({sec})"
    return f"{describe_payload(mac, kind, plain)}  [{sec} → 既知の鍵で復号]"


def describe_payload(mac, kind, p):
    if mac["frame_type"] == 3:
        return f"Command {MAC_COMMANDS.get(p[0], f'0x{p[0]:02x}')}" if p else kind
    if mac["frame_type"] == 0:
        b = parse_beacon(p)
        return f"Beacon Thread name='{b['network_name']}' xpanid={b['ext_pan_id']}" if b else kind
    if not p:
        return kind
    if p[0] & 0x3C == 0x08 and len(p) > 8:  # Zigbee NWK（プロトコル版 2）の可能性
        return f"{kind} (Zigbee NWK の可能性)"
    info = parse_lowpan(p, mac)
    if "ip" not in info:
        return f"{kind} 6LoWPAN {info.get('frag', 'unknown dispatch')}"
    ip = info["ip"]
    text = f"{kind} {info['frag'] + ' ' if 'frag' in info else ''}IPv6 {ip['src']} -> {ip['dst']} hlim={ip['hop_limit']}"
    if "src_port" in ip:
        text += f" UDP {ip['src_port']}->{ip['dst_port']}"
        if ip["dst_port"] == MLE_PORT and ip["payload"]:
            mle = parse_mle(ip["payload"])
            if mle.get("secured"):
                text += f" MLE (暗号化 counter={mle['frame_counter']})"
            elif mle:
                text += f" MLE {mle['command']}"
    return text


def main(path):
    x = load_iq(path)
    bursts = detect_bursts(x)
    print(f"bursts: {len(bursts)}")
    for s, e in bursts:
        burst = x[max(0, s - 400):e + 400]
        score, signs, pos = find_shr(burst)
        if score < 0.6 * len(SHR):
            continue
        psdu = read_ppdu(signs, pos)
        if not check_fcs(psdu):
            print(f"t={s / SAMPLE_RATE:8.3f}s FCS NG")
            continue
        mac = parse_mac(psdu)
        pan = f"PAN=0x{mac['dst_pan']:04x} " if "dst_pan" in mac else ""
        print(f"t={s / SAMPLE_RATE:8.3f}s len={len(psdu):3d} seq={mac.get('seq', '-')} v{mac['version']} "
              f"{pan}{mac.get('src', '-')} -> {mac.get('dst', '-')}")
        print(f"    {describe(mac)}")


# ---------------------------------------------------------------- テスト用 IQ の生成

def build_ppdu_chips(psdu_wo_fcs):
    psdu = psdu_wo_fcs + crc16_kermit(psdu_wo_fcs).to_bytes(2, "little")
    data = bytes(4) + bytes([0xA7, len(psdu)]) + psdu  # プリアンブル・SFD・PHR・PSDU
    symbols = [n for b in data for n in (b & 0xF, b >> 4)]
    return np.concatenate([CHIPS[s] for s in symbols])


def make_test_iq(path, seed=1):
    rng = np.random.default_rng(seed)
    ext_router = bytes.fromhex("1122334455667788")[::-1]
    ext_child = bytes.fromhex("8a7b6c5d4e3f2a1b")[::-1]
    pan = (0xFACE).to_bytes(2, "little")
    mle_sec = bytes.fromhex("0015") + (1234).to_bytes(4, "little") + bytes(4) + b"\x01"
    frames = [
        # MLE Advertisement（ルーター → ff02::1、MAC 暗号化なし・MLE 暗号化あり）
        bytes.fromhex("41d8") + b"\x10" + pan + b"\xff\xff" + ext_router
        + bytes.fromhex("7f3b01f04d4c4d4c0000") + mle_sec + rng.bytes(20) + rng.bytes(4),
        # MAC 暗号化されたデータフレーム（子機 → ルーター、2006 形式・拡張アドレス）
        bytes.fromhex("69dc") + b"\x11" + pan + ext_router + ext_child
        + bytes.fromhex("0d") + (5678).to_bytes(4, "little") + b"\x02" + rng.bytes(30) + rng.bytes(4),
        # Ack
        bytes.fromhex("0200") + b"\x11",
        # MLE Discovery Request（暗号化なし）
        bytes.fromhex("41d8") + b"\x12" + pan + b"\xff\xff" + ext_child
        + bytes.fromhex("7f3b01f04d4c4d4c0000") + bytes.fromhex("ff10") + rng.bytes(8),
    ]
    sig = [np.zeros(4000, complex)]
    for f in frames:
        sig += [oqpsk_modulate(build_ppdu_chips(f)), np.zeros(8000, complex)]
    x = np.concatenate(sig)
    x *= np.exp(2j * np.pi * 30e3 * np.arange(len(x)) / SAMPLE_RATE)  # 周波数オフセット
    x = 0.4 * x + 0.02 * (rng.standard_normal(len(x)) + 1j * rng.standard_normal(len(x)))
    iq = np.empty(2 * len(x), np.int8)
    iq[0::2] = np.clip(x.real * 127, -127, 127)
    iq[1::2] = np.clip(x.imag * 127, -127, 127)
    iq.tofile(path)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        make_test_iq("thread_test.iq")
        main("thread_test.iq")
    else:
        main(sys.argv[1] if len(sys.argv) > 1 else "data/thread_ch11.iq")
