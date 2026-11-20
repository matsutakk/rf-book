"""GPS L1 C/A を IQ データから復調し、受信機の位置を求めるサンプル

hackrf_transfer -r gps.iq -f 1575420000 -s 4000000 -p 1 -a 0 -l 40 -g 40 -n 720000000
で記録したファイルを想定（-p 1 はアクティブアンテナへの給電。180秒分）

  python gps.py gps.iq --skip 25 --seconds 60   # 25秒目から60秒分を処理する
"""
import numpy as np

L1_FREQ = 1_575_420_000   # L1の搬送波周波数 [Hz]
CHIP_RATE = 1_023_000  # C/Aコードのチップレート [Hz]（1秒に1023000チップ）
CODE_LEN = 1023        # C/Aコード1周期のチップ数（1周期 = 1ミリ秒）
C = 299_792_458.0          # 光速 [m/s]
MU = 3.986005e14           # 地球の重力定数 [m^3/s^2]
OMEGA_E = 7.2921151467e-5  # 地球の自転角速度 [rad/s]

# C/Aコード生成用 G2 の出力タップ（PRN 1〜32）。衛星ごとにG2のどの2セルを使うか
G2_TAPS = {
    1: (2, 6), 2: (3, 7), 3: (4, 8), 4: (5, 9), 5: (1, 9), 6: (2, 10), 7: (1, 8), 8: (2, 9),
    9: (3, 10), 10: (2, 3), 11: (3, 4), 12: (5, 6), 13: (6, 7), 14: (7, 8), 15: (8, 9),
    16: (9, 10), 17: (1, 4), 18: (2, 5), 19: (3, 6), 20: (4, 7), 21: (5, 8), 22: (6, 9),
    23: (1, 3), 24: (4, 6), 25: (5, 7), 26: (6, 8), 27: (7, 9), 28: (8, 10), 29: (1, 6),
    30: (2, 7), 31: (3, 8), 32: (4, 9),
}


# ---------------------------------------------------------------- IQ の読み込み

def load_iq(path, seconds=None, fs=4_000_000, skip=0):
    """IQファイル（skip 秒目から seconds 秒分） → 複素数の配列（平均0）"""
    # 読む個数（I, Qで2個ずつ）。-1ならファイルの最後まで
    count = -1 if seconds is None else int(seconds * fs) * 2
    # 符号付き8bitのI, Q, I, Q, ...を、skip秒分（バイト数）飛ばして読む
    raw = np.fromfile(path, dtype=np.int8, count=count, offset=int(skip * fs) * 2)
    x = np.empty(len(raw) // 2, dtype=np.complex64)  # 32bitの複素数で省メモリ
    x.real, x.imag = raw[0::2], raw[1::2]  # 偶数番目がI、奇数番目がQ
    return x - np.mean(x)                   # 平均（DC成分）を引いて中心を0にする


def load_1bit(path, seconds=None, fs=1_000_000):
    """gps-sdr-sim の1bit形式のファイル → 複素数の配列"""
    # gps-sdr-sim の -b 1 形式（1バイトに I,Q 4組、MSB から）
    count = -1 if seconds is None else int(seconds * fs) // 4  # 1バイトに4サンプル
    bits = np.unpackbits(np.fromfile(path, dtype=np.uint8, count=count)).astype(np.float32)
    bits = 2 * bits - 1                  # 0/1 を -1/+1 にする
    return bits[0::2] + 1j * bits[1::2]  # 偶数番目がI、奇数番目がQ


# ---------------------------------------------------------------- C/A コード

def ca_code(prn):
    """PRN番号（1〜32） → C/Aコード（+1/-1、1023個の配列）"""
    g1 = [1] * 10        # G1：10セルのシフトレジスタ（初期値はすべて1）
    g2 = [1] * 10        # G2：同じく10セル
    a, b = G2_TAPS[prn]  # この衛星でG2から取り出す2セルの番号
    chips = []
    for _ in range(CODE_LEN):  # 1023チップ作る
        # G1の出力とG2の2セルのXOR
        chips.append(g1[9] ^ g2[a - 1] ^ g2[b - 1])
        f1 = g1[2] ^ g1[9]  # G1の帰還（セル3と10）
        # G2の帰還（セル2, 3, 6, 8, 9, 10）
        f2 = g2[1] ^ g2[2] ^ g2[5] ^ g2[7] ^ g2[8] ^ g2[9]
        g1 = [f1] + g1[:9]  # 右へシフトし、先頭に帰還を入れる
        g2 = [f2] + g2[:9]
    return 1 - 2 * np.array(chips)  # 0 → +1, 1 → -1


def sample_code(code, fs, n, offset_chips=0.0, rate=CHIP_RATE):
    """C/Aコード → サンプル n 個分に並べたコード（+1/-1の配列）"""
    # 各サンプルの時刻に当たるチップの番号（1023で一周する）
    idx = np.floor(offset_chips + np.arange(n) * rate / fs).astype(int) % CODE_LEN
    return code[idx]


# ---------------------------------------------------------------- 捕捉

def acquire(x, fs, prn, dopplers=np.arange(-5000, 5001, 250), n_ms=40):
    """IQとPRN番号 → ドップラー、コードの開始位置、metric などの辞書"""
    n = int(fs * 1e-3)     # 1ミリ秒（コード1周期）のサンプル数
    t = np.arange(n) / fs  # 1ミリ秒の中の各サンプルの時刻 [秒]
    # 相関をFFTで計算するため、コードのFFTの複素共役を用意する
    code_fft = np.conj(np.fft.fft(sample_code(ca_code(prn), fs, n)))
    # 行：ドップラーの候補、列：コードのずれ
    grid = np.zeros((len(dopplers), n))
    for i, fd in enumerate(dopplers):
        # n_ms個の1ミリ秒について足し合わせ、雑音を平均で減らす
        for k in range(n_ms):
            # k番目の1msを取り出し、ドップラーの分だけ周波数を戻す
            seg = x[k * n:(k + 1) * n] * np.exp(-2j * np.pi * fd * (t + k * 1e-3))
            # 全コード位相の相関を一度に求め、電力を足し合わせる
            grid[i] += np.abs(np.fft.ifft(np.fft.fft(seg) * code_fft)) ** 2
    # いちばん強い (行, 列)
    i, j = np.unravel_index(np.argmax(grid), grid.shape)
    metric = grid[i, j] / np.mean(grid)  # ピークが平均の何倍か
    # j はコードの先頭が来るサンプル位置（相関のずれ）
    return {"prn": prn, "doppler": float(dopplers[i]), "code_start": int(j),
            "metric": float(metric), "grid": grid}


# ---------------------------------------------------------------- 追尾

def track(x, fs, acq, n_ms):
    """IQと捕捉結果 → 1msごとのPrompt（複素数）とコード周期の開始位置の辞書"""
    code = ca_code(acq["prn"])
    n = int(fs * 1e-3)     # 1ミリ秒（コード1周期）のサンプル数
    t = np.arange(n) / fs  # 1ミリ秒の中の各サンプルの時刻 [秒]
    # 搬送波の周波数のずれ [Hz]。捕捉の結果から始めて追いかける
    fd = acq["doppler"]
    phase = 0.0  # ブロック先頭での搬送波の位相
    start = float(acq["code_start"])  # コード周期が始まるサンプル位置（小数）
    prev = None                       # 1つ前のPrompt（FLLで使う）
    prompts, starts = [], []          # 結果：1ミリ秒ごとのPromptと開始位置
    for _ in range(n_ms):
        s = int(start)     # 今回の1ミリ秒の先頭（整数のサンプル番号）
        if s + n > len(x):  # ファイルの終わり
            break
        # 搬送波を取り除く
        base = x[s:s + n] * np.exp(-1j * (phase + 2 * np.pi * fd * t))
        rate = CHIP_RATE * (1 + fd / L1_FREQ)  # ドップラーで少し変わる
        frac = (s - start) * rate / fs  # サンプル s でのコード位相 [chip]
        # 0.5チップ早い・ちょうど・遅いコードとの相関（1ms分）
        early = np.sum(base * sample_code(code, fs, n, frac + 0.5, rate))
        prompt = np.sum(base * sample_code(code, fs, n, frac, rate))
        late = np.sum(base * sample_code(code, fs, n, frac - 0.5, rate))
        prompts.append(prompt)  # 航法ビットはPromptの符号に載っている
        starts.append(start)

        # PLL：航法ビットで符号が反転しても誤差が出ないよう atan(Q/I) を使う
        err = np.arctan(prompt.imag / prompt.real) if prompt.real != 0 else 0.0
        # FLL：前回との位相の変化から周波数のずれを推定する
        ferr = 0.0
        if prev is not None:
            dot = (prompt * np.conj(prev)).real    # 位相の変化のcos成分
            cross = (prompt * np.conj(prev)).imag  # 位相の変化のsin成分
            # 1ミリ秒での位相の変化 [rad] を、周波数 [Hz] に直す
            ferr = np.arctan(cross / dot) / (2 * np.pi * 1e-3) if dot != 0 else 0.0
        prev = prompt
        # DLL：early と late の差からコードのずれを推定する
        e, l = np.abs(early), np.abs(late)  # early, late の相関の大きさ
        # earlyが強ければコードが早く来ている（正）、
        # lateが強ければ遅れている（負）
        dll = 0.5 * (e - l) / (e + l + 1e-12)  # [chip]

        # 次の周期の開始位置：1周期分進め、DLLのずれで少し前後させる
        # （係数は、弱い信号でも雑音に振り回されないよう小さめにしている）
        next_start = start + CODE_LEN / rate * fs - 0.05 * dll * fs / rate
        # 搬送波の位相と周波数を、PLLとFLLの誤差で少しずつ直す
        phase += 2 * np.pi * fd * (int(next_start) - s) / fs + 0.03 * err
        fd += 0.08 * err + 0.005 * ferr
        start = next_start
    return {"prn": acq["prn"], "prompts": np.array(prompts), "starts": np.array(starts)}


# ---------------------------------------------------------------- 航法ビット

def to_nav_bits(prompts):
    """1msごとのPrompt → (航法ビットの0/1の配列, ビット境界の位置)"""
    # 1ビット = 20ms。符号が変わる位置からビットの境界を探す
    p = np.sign(prompts.real)  # 1ミリ秒ごとの符号（+1/-1）
    flips = np.nonzero(p[1:] != p[:-1])[0] + 1  # 符号が変わった位置
    # 20で割った余りで最も多いものが、ビットの境界
    edge = int(np.bincount(flips % 20, minlength=20).argmax())
    n = (len(p) - edge) // 20  # 取り出せるビット数
    # 境界から20msずつ足し合わせ、符号でビットを決める
    sums = prompts.real[edge:edge + n * 20].reshape(n, 20).sum(axis=1)
    return (sums > 0).astype(np.uint8), edge


# サブフレームの先頭の8bit
PREAMBLE = np.array([1, 0, 0, 0, 1, 0, 1, 1], dtype=np.uint8)


def parity_ok(word, d29, d30):
    """30bitのワードと直前のD29, D30 → (パリティが合うか, データ24bit)"""
    d = word[:24] ^ d30  # 前のワードの D30 が 1 ならデータは反転している
    # パリティ6bitのそれぞれについて、
    # (使う前のワードのbit, XORするデータの位置)
    idx = [
        (d29, [1, 2, 3, 5, 6, 10, 11, 12, 13, 14, 17, 18, 20, 23]),
        (d30, [2, 3, 4, 6, 7, 11, 12, 13, 14, 15, 18, 19, 21, 24]),
        (d29, [1, 3, 4, 5, 7, 8, 12, 13, 14, 15, 16, 19, 20, 22]),
        (d30, [2, 4, 5, 6, 8, 9, 13, 14, 15, 16, 17, 20, 21, 23]),
        (d30, [1, 3, 5, 6, 7, 9, 10, 14, 15, 16, 17, 18, 21, 22, 24]),
        (d29, [3, 5, 6, 8, 9, 10, 11, 13, 15, 19, 22, 23, 24]),
    ]
    # 各パリティ = 前のワードのD29かD30と、決まった位置のデータのXOR
    calc = [prev ^ (np.sum(d[np.array(ix) - 1]) & 1) for prev, ix in idx]
    return np.array_equal(calc, word[24:30]), d


def find_subframes(bits):
    """航法ビットの配列 → (先頭の位置, パリティを除いた240bit) のリスト"""
    subframes = []
    for pol in (0, 1):  # 極性がわからないので、そのままと反転の両方を試す
        b = bits ^ pol
        # 1サブフレーム = 10ワード（30bitずつ）= 300bit
        for i in range(len(b) - 300):
            # プリアンブルを探し、そこから10ワードのパリティをすべて確かめる
            if not np.array_equal(b[i:i + 8], PREAMBLE):
                continue
            # words：データ24bitのリスト、d29, d30：前のワードの末尾2bit
            words, d29, d30, ok = [], 0, 0, True
            for w in range(10):  # 10ワードを順に検査する
                good, data = parity_ok(b[i + 30 * w:i + 30 * w + 30], d29, d30)
                if not good:
                    ok = False
                    break
                words.append(data)
                # このワードの末尾2ビットを、次のワードの検査に使う
                d29, d30 = b[i + 30 * w + 28], b[i + 30 * w + 29]
            if ok:  # 10ワードすべて合格したら、サブフレームとして採用
                subframes.append((i, np.concatenate(words)))
        if subframes:  # 片方の極性で見つかれば、もう片方は試さない
            break
    return subframes


def field(data, *ranges, signed=False, scale=1.0):
    """240bitの配列と位置の範囲 → その範囲の値（整数か、単位を掛けた実数）"""
    # ranges は 300bit 中の位置（1始まり）。
    # パリティを除いた 240bit 配列から取り出す
    bits = []
    for a, b in ranges:
        for p in range(a, b + 1):
            # 何ワード目の何ビット目か（各ワードのパリティ6bitは除いてある）
            w, k = divmod(p - 1, 30)
            bits.append(data[w * 24 + k])
    v = int("".join(map(str, bits)), 2)  # 0/1の並びを2進数として整数にする
    if signed and bits[0]:  # 2の補数
        v -= 1 << len(bits)
    return v * scale if scale != 1.0 else v  # 単位（LSBの大きさ）を掛ける


def parse_subframe(data):
    """240bitの配列 → TOW、サブフレームID、軌道や時計の値の辞書"""
    # HOW（2ワード目）：TOWは6秒単位、IDはサブフレームの番号
    sf = {"tow": field(data, (31, 47)) * 6, "id": field(data, (50, 52))}
    pi = np.pi  # 角度の値は「半円（π）」単位で送られてくる
    # 位置はICDの300bit中の番号。scaleは1単位の大きさ（2**-31 など）
    if sf["id"] == 1:  # サブフレーム1：週番号と衛星時計の補正
        sf.update(week=field(data, (61, 70)),
                  tgd=field(data, (197, 204), signed=True, scale=2**-31),
                  toc=field(data, (219, 234), scale=16),
                  af2=field(data, (241, 248), signed=True, scale=2**-55),
                  af1=field(data, (249, 264), signed=True, scale=2**-43),
                  af0=field(data, (271, 292), signed=True, scale=2**-31))
    elif sf["id"] == 2:  # サブフレーム2：軌道（エフェメリスの前半）
        sf.update(crs=field(data, (69, 84), signed=True, scale=2**-5),
                  dn=field(data, (91, 106), signed=True, scale=2**-43 * pi),
                  m0=field(data, (107, 114), (121, 144), signed=True, scale=2**-31 * pi),
                  cuc=field(data, (151, 166), signed=True, scale=2**-29),
                  e=field(data, (167, 174), (181, 204), scale=2**-33),
                  cus=field(data, (211, 226), signed=True, scale=2**-29),
                  sqrt_a=field(data, (227, 234), (241, 264), scale=2**-19),
                  toe=field(data, (271, 286), scale=16))
    elif sf["id"] == 3:  # サブフレーム3：軌道（エフェメリスの後半）
        sf.update(cic=field(data, (61, 76), signed=True, scale=2**-29),
                  omega0=field(data, (77, 84), (91, 114), signed=True, scale=2**-31 * pi),
                  cis=field(data, (121, 136), signed=True, scale=2**-29),
                  i0=field(data, (137, 144), (151, 174), signed=True, scale=2**-31 * pi),
                  crc=field(data, (181, 196), signed=True, scale=2**-5),
                  omega=field(data, (197, 204), (211, 234), signed=True, scale=2**-31 * pi),
                  omega_dot=field(data, (241, 264), signed=True, scale=2**-43 * pi),
                  idot=field(data, (279, 292), signed=True, scale=2**-43 * pi))
    return sf


# ---------------------------------------------------------------- 衛星の位置

def sat_position(eph, t):
    """エフェメリスと時刻 t [秒] → (衛星の位置 ECEF [m], 時計のずれ [秒])"""
    a = eph["sqrt_a"] ** 2  # 軌道長半径 [m]
    tk = t - eph["toe"]     # エフェメリスの基準時刻からの経過時間 [秒]
    n = np.sqrt(MU / a ** 3) + eph["dn"]  # 平均運動 [rad/s]
    m = eph["m0"] + n * tk  # 平均近点角：一定の速さで回ったとしたときの角度
    e = eph["e"]            # 離心率（軌道の楕円のつぶれ具合）
    ek = m                  # 離心近点角の初期値
    for _ in range(10):     # E = M + e sin E を繰り返して解く
        ek = m + e * np.sin(ek)  # ケプラー方程式
    # 真近点角：楕円軌道上の実際の角度
    nu = np.arctan2(np.sqrt(1 - e ** 2) * np.sin(ek), np.cos(ek) - e)
    phi = nu + eph["omega"]  # 昇交点から測った角度
    # 摂動の補正項を加えた、角度・半径・傾き
    u = phi + eph["cus"] * np.sin(2 * phi) + eph["cuc"] * np.cos(2 * phi)
    r = a * (1 - e * np.cos(ek)) + eph["crs"] * np.sin(2 * phi) + eph["crc"] * np.cos(2 * phi)
    i = eph["i0"] + eph["idot"] * tk + eph["cis"] * np.sin(2 * phi) + eph["cic"] * np.cos(2 * phi)
    # 昇交点の経度（地球の自転を差し引いてECEFの向きにする）
    om = eph["omega0"] + (eph["omega_dot"] - OMEGA_E) * tk - OMEGA_E * eph["toe"]
    x, y = r * np.cos(u), r * np.sin(u)  # 軌道面上の位置
    # 軌道面上の (x, y) を、傾き i と向き om で回してECEFに直す
    pos = np.array([x * np.cos(om) - y * np.cos(i) * np.sin(om),
                    x * np.sin(om) + y * np.cos(i) * np.cos(om),
                    y * np.sin(i)])
    # 衛星時計の補正（相対論補正を含む）
    dt = t - eph["toc"]  # 時計の基準時刻からの経過時間 [秒]
    clk = eph["af0"] + eph["af1"] * dt + eph["af2"] * dt ** 2 - eph["tgd"]
    clk += -4.442807633e-10 * e * eph["sqrt_a"] * np.sin(ek)
    return pos, clk


# ---------------------------------------------------------------- 擬似距離と測位

def transmit_time(trk, sfs, n_rx, fs):
    """追尾結果、サブフレーム、受信のサンプル番号 → 衛星を出た時刻 [秒]"""
    # サンプル n_rx の時点で受信している信号が、衛星を出た時刻（GPS時刻）
    i0, _ = sfs[0]  # 最初のサブフレームの先頭（航法ビットの番号）
    sf = parse_subframe(sfs[0][1])
    first_ms = trk["edge"] + i0 * 20  # サブフレーム先頭のコード周期
    t0 = sf["tow"] - 6  # HOW の TOW は「次の」サブフレームの開始時刻
    # n_rx を含むコード周期の番号
    k = int(np.searchsorted(trk["starts"], n_rx) - 1)
    # サブフレームの開始時刻 + 経過した周期の数（1周期1ミリ秒）
    # + 周期の中の端数
    return t0 + (k - first_ms) * 1e-3 + (n_rx - trk["starts"][k]) / fs


def solve_position(sats, t_tx):
    """エフェメリスと送信時刻のリスト → (位置 ECEF [m], 時計のずれ [m])"""
    pos = np.zeros(3)  # 受信機の位置の初期値（地球の中心）
    bias = 0.0         # 受信機の時計のずれ（距離に換算 [m]）
    t_rx = max(t_tx) + 0.075  # 仮の受信時刻（ずれは bias で吸収）
    for _ in range(10):  # 位置を少しずつ直して、10回繰り返す
        # rows：連立方程式の係数、res：測った距離と計算した距離の差
        rows, res = [], []
        for eph, tt in zip(sats, t_tx):
            # 送信時刻での衛星の位置と時計のずれ
            sp, clk = sat_position(eph, tt)
            tau = np.linalg.norm(sp - pos) / C  # 電波が届くまでの時間 [秒]
            th = OMEGA_E * tau  # 伝搬中の地球の自転を補正
            sp = np.array([sp[0] * np.cos(th) + sp[1] * np.sin(th),
                           -sp[0] * np.sin(th) + sp[1] * np.cos(th), sp[2]])
            rho = C * (t_rx - tt + clk)  # 擬似距離
            # いまの推定位置から衛星までの距離
            dist = np.linalg.norm(sp - pos)
            # 位置を少し動かしたときの距離の変化（方向ベクトル）と、時計の項
            rows.append(np.append((pos - sp) / dist, 1.0))
            res.append(rho - dist - bias)
        # 最小二乗法で修正量を求める
        delta = np.linalg.lstsq(np.array(rows), np.array(res), rcond=None)[0]
        pos += delta[:3]  # 位置の修正量 (x, y, z)
        bias += delta[3]  # 時計のずれの修正量
    return pos, bias


def ecef_to_lla(p):
    """ECEF座標 [m] → (緯度 [度], 経度 [度], 高さ [m])"""
    a, f = 6378137.0, 1 / 298.257223563  # WGS84の楕円体
    e2 = f * (2 - f)                     # 離心率の2乗
    lon = np.arctan2(p[1], p[0])         # 経度はx, yの向きだけで決まる
    r = np.hypot(p[0], p[1])             # 地軸からの距離
    lat = np.arctan2(p[2], r * (1 - e2))  # 緯度の初期値
    # 緯度と高さは互いに依存するので、繰り返して求める
    for _ in range(10):
        # その緯度での楕円体の曲率半径
        n = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
        h = r / np.cos(lat) - n                     # 楕円体からの高さ
        lat = np.arctan2(p[2], r * (1 - e2 * n / (n + h)))
    return np.degrees(lat), np.degrees(lon), h


# ---------------------------------------------------------------- まとめ

def main(path, fs=4_000_000, one_bit=False, seconds=60, skip=0,
         dopplers=np.arange(-40000, 40001, 500)):
    # Step 1：IQ（複素数の配列）
    x = load_1bit(path, seconds, fs) if one_bit else load_iq(path, seconds, fs, skip)
    print(f"samples: {len(x)} ({len(x) / fs:.1f} s)")
    found = []  # 見つかった衛星の捕捉結果（辞書）のリスト
    # Step 2〜3：32機すべてのC/Aコードで探し、見つかった衛星（辞書）だけ残す
    for prn in range(1, 33):
        acq = acquire(x, fs, prn, dopplers)
        if acq["metric"] > 3.5:  # ピークが平均の3.5倍を超えたら「見つかった」
            print(f"PRN {prn:2d}: doppler={acq['doppler']:+8.0f} Hz "
                  f"code_start={acq['code_start']:5d} metric={acq['metric']:.1f}")
            found.append(acq)

    # sats：エフェメリス、trks：追尾結果、sf_list：サブフレーム（衛星ごと）
    sats, trks, sf_list = [], [], []
    for acq in found:
        # Step 4：1msごとのPromptと、コード周期の開始位置
        trk = track(x, fs, acq, int(len(x) / fs * 1000) - 2)
        bits, edge = to_nav_bits(trk["prompts"])  # Step 5：航法ビット
        trk["edge"] = edge
        sfs = find_subframes(bits)  # Step 6：(位置, 240bit) のリスト
        parsed = [parse_subframe(d) for _, d in sfs]  # 各サブフレームの値の辞書
        eph = {}  # Step 7：エフェメリス（名前 → 値）
        for sf in parsed:
            eph.update({k: v for k, v in sf.items() if k not in ("tow", "id")})
        ids = sorted({sf["id"] for sf in parsed})  # 受信できたサブフレームの番号
        print(f"PRN {acq['prn']:2d}: bits={len(bits)} subframes={ids}")
        # サブフレーム1〜3がそろった
        if {"sqrt_a", "omega0", "af0"} <= eph.keys():
            sats.append(eph)
            trks.append(trk)
            sf_list.append(sfs)

    if len(sats) < 4:  # 未知数は位置 x, y, z と時計のずれの4つ
        print("衛星が4機未満のため測位できません")
        return
    # Step 8〜9：同じ受信の瞬間 n_rx での、各衛星の送信時刻
    n_rx = int(min(t["starts"][-2] for t in trks))
    t_tx = [transmit_time(t, s, n_rx, fs) for t, s in zip(trks, sf_list)]
    # Step 10：受信機の位置と時計のずれ
    pos, bias = solve_position(sats, t_tx)
    lat, lon, h = ecef_to_lla(pos)  # 地球中心の座標を緯度・経度・高さに
    print(f"position: lat={lat:.6f} lon={lon:.6f} height={h:.1f} m")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="GPS L1 C/A の捕捉・追尾・測位")
    ap.add_argument("path", nargs="?", default="data/gps.iq")
    ap.add_argument("--skip", type=float, default=0, help="先頭から飛ばす秒数")
    ap.add_argument("--seconds", type=float, default=60, help="処理する秒数")
    ap.add_argument("--1bit", dest="one_bit", action="store_true",
                    help="gps-sdr-sim の 1bit・1MHz 形式")
    args = ap.parse_args()
    if args.one_bit:
        main(args.path, fs=1_000_000, one_bit=True, seconds=args.seconds,
             dopplers=np.arange(-5000, 5001, 250))
    else:
        main(args.path, seconds=args.seconds, skip=args.skip)
