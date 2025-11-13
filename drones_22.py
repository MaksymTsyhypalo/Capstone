# -*- coding: utf-8 -*-
import os
import numpy as np
import soundfile as sf
from collections import deque
import matplotlib.pyplot as plt
import time

# ================== KONFIG ==================
CHAN_OF_MIC = [0,1,2,3,4,5,6,7]   
C = 343.0                         
RADIUS_M = 0.13                   


LR_PAIRS = [(5,1), (4,2), (6,8)]  
UD_PAIRS = [(3,7), (2,8), (4,6)]  
WAV_PATH = "Drone_2_8ch_24bit_96000Hz.wav"
START_SEC = None                  
DURATION_SEC = None               
WIN_SEC = 0.5                     
HOP_SEC = 0.5                     
ENERGY_SILENCE = 1e-8            


SENS_GAIN = 1.8                   
Y_SENS_GAIN = 3.8                 

FRAME_TIME_S = 0.5                


PAD_X = 0.10
PAD_Y_BOTTOM = 0.10
PAD_Y_TOP = 0.90
XLIM = (-1.0 - PAD_X,        1.0 + PAD_X)
YLIM = (-1.0 - PAD_Y_BOTTOM, 1.0 + PAD_Y_TOP)



DX_THR = 0.020            
DY_THR = 0.010            


LR_SUPPRESS = 1.2         
VERT_WEIGHT = 1.8         
HYST_FACTOR_Y = 0.80      


EMA_ALPHA_X = 0.35        
EMA_ALPHA_Y = 0.30        

# ================== GCC-PHAT / TDoA ==================
def gcc_phat_cc(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Liczy skorelowaną funkcję cross-correlation GCC-PHAT
    między dwoma sygnałami a i b.
    """
    n = a.shape[0] + b.shape[0]
    A = np.fft.rfft(a, n=n)
    B = np.fft.rfft(b, n=n)
    R = A * np.conj(B)
    R /= (np.abs(R) + 1e-12)
    cc = np.fft.irfft(R, n=n)
    m = n // 2
    return np.concatenate((cc[-m:], cc[:m+1]))

def max_tau_seconds(mic_a: int, mic_b: int) -> float:
    """
    Maksymalny czas opóźnienia między dwoma mikrofonami
    ~ (średnica pierścienia / prędkość dźwięku).
    """
    return (2.0 * RADIUS_M) / C

def tdoa_tau_seconds(seg: np.ndarray, ch_a: int, ch_b: int, sr: int, max_tau: float) -> float:
    """
    Szacuje TDoA (tau) między kanałami ch_a i ch_b w danym segmencie.
    Zwraca czas opóźnienia w sekundach, obcięty do [-max_tau, max_tau].
    """
    cc = gcc_phat_cc(seg[:, ch_a], seg[:, ch_b])
    center = cc.size // 2
    max_shift = int(round(max_tau * sr))
    lo = max(0, center - max_shift)
    hi = min(cc.size - 1, center + max_shift)
    window = cc[lo:hi+1]
    k = int(np.argmax(window))
    idx = lo + k
    if 0 < idx < cc.size - 1:
        y0, y1, y2 = cc[idx-1], cc[idx], cc[idx+1]
        denom = (y0 - 2*y1 + y2)
        delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
    else:
        delta = 0.0
    lag_samples = (idx - center) + delta
    tau = lag_samples / float(sr)
    tau = max(-max_tau, min(max_tau, tau))
    return tau

def axis_median_tau(seg: np.ndarray, pairs: list[tuple[int,int]], sr: int) -> float:
    """
    Liczy medianę TDoA po kilku parach mikrofonów dla danej osi (LR lub UD).
    """
    taus = []
    for ma, mb in pairs:
        ch_a = CHAN_OF_MIC[ma-1]
        ch_b = CHAN_OF_MIC[mb-1]
        taus.append(tdoa_tau_seconds(seg, ch_a, ch_b, sr, max_tau_seconds(ma, mb)))
    return float(np.median(taus))

# ================== FILTR / PASMO ==================
def simple_bandpass(X, sr, low=120.0, high=3000.0):
    """
    Prosty filtr 'pasmowy':
    - high-pass przez różniczkę (usuwa DC i bardzo niskie),
    - low-pass przez średnią ruchomą (usuwa bardzo wysokie).
    """
    if low and low > 0:
        X = np.concatenate([X[:1], np.diff(X, axis=0)])
    if high and high > 0:
        k = max(1, int(sr / (2.0 * high)))
        if k > 1:
            ker = np.ones((k,), dtype=np.float32) / k
            Y = np.empty_like(X)
            for ch in range(X.shape[1]):
                Y[:, ch] = np.convolve(X[:, ch], ker, mode="same")
            X = Y
    return X

# ================== POZYCJA (x,y) ==================
def pos_vector_from_tdoa(seg: np.ndarray, sr: int) -> tuple[float,float]:
    """
    Na podstawie TDoA z par LR i UD wyznacza znormalizowaną pozycję (x,y) w [-1,1].
    x > 0 → prawo, x < 0 → lewo
    y > 0 → góra (bliżej/przód), y < 0 → dół (dalej/tył)
    """
    tau_lr = axis_median_tau(seg, LR_PAIRS, sr)
    tau_ud = axis_median_tau(seg, UD_PAIRS, sr)
    scale = (2.0 * RADIUS_M) / C
    x = -tau_lr / scale
    y = -tau_ud / scale
    x *= SENS_GAIN
    y *= SENS_GAIN * Y_SENS_GAIN
    x = float(np.clip(x, -1, 1))
    y = float(np.clip(y, -1, 1))
    return x, y

# ================== DECYZJE (z priorytetem pionu) ==================
def classify_with_vertical_priority(dx: float, dy: float, prev_label: str | None) -> str:
    """
    Na podstawie delta pozycji dx, dy decyduje:
    - 'prawo', 'lewo', 'góra', 'dół' lub 'brak ruchu'
    z priorytetem osi pionowej (góra/dół).
    """
    adx, ady = abs(dx), abs(dy)

    # histereza dla pionu
    dy_thr_eff = DY_THR
    if prev_label in ("góra", "dół", "prawo/góra", "prawo/dół", "lewo/góra", "lewo/dół"):
        dy_thr_eff *= HYST_FACTOR_Y  # łatwiej utrzymać pion

    # 1) czysty pion, gdy poziom słaby
    if (ady >= dy_thr_eff) and (adx <= DX_THR * LR_SUPPRESS):
        return "góra" if dy > 0 else "dół"

    # 2) kolizja: pion vs poziom – pion ma dodatkową wagę
    if (ady >= dy_thr_eff) and (adx >= DX_THR):
        if (ady * VERT_WEIGHT) > adx:
            return "góra" if dy > 0 else "dół"
        else:
            return "prawo" if dx > 0 else "lewo"

    # 3) tylko pion przekracza próg
    if ady >= dy_thr_eff:
        return "góra" if dy > 0 else "dół"

    # 4) tylko poziom przekracza próg
    if adx >= DX_THR:
        return "prawo" if dx > 0 else "lewo"

    return "brak ruchu"

def vote_majority(q):
    """
    Głosowanie większościowe po ostatnich etykietach w kolejce q.
    Przy remisie – preferuje pion (góra/dół).
    """
    if not q:
        return "brak ruchu"
    counts = {}
    for l in q:
        counts[l] = counts.get(l, 0) + 1
    order = ["góra","dół","prawo","lewo","prawo/góra","prawo/dół","lewo/góra","lewo/dół","brak ruchu"]
    return max(order, key=lambda k: counts.get(k,0))

def rms(x: np.ndarray) -> float:
    """
    Liczy RMS (root mean square) – miarę energii sygnału.
    """
    return float(np.sqrt(np.mean(x**2)))

# ================== UI STATE ==================
window_running = True
def on_close(event):
    """
    Callback do matplotlib – ustawiany gdy okno zostanie zamknięte.
    """
    global window_running
    window_running = False

# ================== MAIN ==================
def main():
    global window_running

    if not os.path.exists(WAV_PATH):
        raise FileNotFoundError(WAV_PATH)

    data, sr = sf.read(WAV_PATH, always_2d=True)
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float32) / (2**23)
    if data.shape[1] < 8:
        raise ValueError(f"Plik ma {data.shape[1]} kanałów, potrzebne >= 8.")

    # ewentualne cięcie fragmentu
    if START_SEC is not None:
        s = int(round(START_SEC * sr))
        e = data.shape[0] if DURATION_SEC is None else min(data.shape[0], s + int(round(DURATION_SEC * sr)))
        data = data[s:e]

    # uporządkowanie kanałów wg CHAN_OF_MIC
    order = [CHAN_OF_MIC[m-1] for m in range(1,9)]
    X = data[:, order]

    # filtr pasmowy
    X = simple_bandpass(X, sr, 120.0, 3000.0)

    N = X.shape[0]
    win = int(round(WIN_SEC * sr))
    hop = int(round(HOP_SEC * sr))

    label_hist = deque(maxlen=3)

    print(f"Plik: {WAV_PATH}  sr={sr} Hz  czas={N/sr:.1f}s\nWynik:")

    # interactive mode – wykres
    plt.ion()
    fig, ax = plt.subplots()
    fig.canvas.mpl_connect('close_event', on_close)

    x_przed_br, y_przed_br = None, None

    # EMA pozycji
    ema_x = None
    ema_y = None

    prev_label = None
    sec = 0.0

    for start in range(0, N - win + 1, hop):
        sec += HOP_SEC
        seg = X[start:start+win, :]

        # CISZA
        if rms(seg) < ENERGY_SILENCE:
            label_hist.append("brak ruchu")
            print(f"{sec:.1f} s = brak ruchu (cisza)")
            if not window_running:
                break

            if x_przed_br is not None and y_przed_br is not None:
                plt.cla()
                ax.set_xlim(*XLIM)
                ax.set_ylim(*YLIM)
                ax.set_aspect("auto")
                ax.axhline(0, color="0.85", linewidth=0.8)
                ax.axvline(0, color="0.85", linewidth=0.8)
                ax.plot(x_przed_br, y_przed_br, "ro")
                plt.draw()
                plt.pause(FRAME_TIME_S)
            else:
                time.sleep(FRAME_TIME_S)

            ema_x = ema_y = None
            prev_label = "brak ruchu"
            continue

        # pozycja z TDoA
        x, y = pos_vector_from_tdoa(seg, sr)

        # EMA osobno dla osi
        if ema_x is None:
            ema_x, ema_y = x, y
            dx, dy = 0.0, 0.0
        else:
            ema_x = (1-EMA_ALPHA_X)*ema_x + EMA_ALPHA_X*x
            ema_y = (1-EMA_ALPHA_Y)*ema_y + EMA_ALPHA_Y*y
            dx = ema_x - (x_przed_br if x_przed_br is not None else ema_x)
            dy = ema_y - (y_przed_br if y_przed_br is not None else ema_y)

        # klasyfikacja z priorytetem pionu
        label = classify_with_vertical_priority(dx, dy, prev_label)
        prev_label = label
        label_hist.append(label)
        voted = vote_majority(label_hist)
        print(f"{sec:.1f} s = {voted}")

        if not window_running:
            break

        # rysowanie
        plt.cla()
        ax.set_xlim(*XLIM)
        ax.set_ylim(*YLIM)
        ax.set_aspect("auto")  # widok z boku
        ax.axhline(0, color="0.85", linewidth=0.8)
        ax.axvline(0, color="0.85", linewidth=0.8)

        ax.plot(ema_x, ema_y, "ro")
        if (x_przed_br is not None) and (y_przed_br is not None):
            ax.plot([x_przed_br, ema_x], [y_przed_br, ema_y], "k--")

        x_przed_br, y_przed_br = ema_x, ema_y

        plt.draw()
        plt.pause(FRAME_TIME_S)

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
