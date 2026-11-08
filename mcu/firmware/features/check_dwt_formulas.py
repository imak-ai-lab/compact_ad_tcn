"""Check, on the host, the index formulas main.c uses for the wavelet transforms against pywt.

  dwt 'per':   o-th output = sum_j h[j] * xt[(F/2 + 2o - j) mod N2], N2 = N + (N odd), xt = x + [x[-1]] if odd
  idwt 'per':  for t in 0..M-1, i = F/4 - 1 + t:
                 y[(2t - 1) mod 2M] = sum_{j<F/2} f[2j]   * c[(i - j) mod M]   (a with rec_lo + d with rec_hi)
                 y[2t]              = sum_{j<F/2} f[2j+1] * c[(i - j) mod M]
  dwt 'symmetric': o-th output = sum_j h[j] * xs[1 + 2o - j], xs = half-sample symmetric extension, floor((N+F-1)/2) outputs
Exit code 1 if any difference exceeds 1e-12. No board and no compiler needed.
"""
import sys

import numpy as np
import pywt

TOL = 1e-12
rng = np.random.default_rng(1)
worst = 0.0


def dwt_per(x, w):
    h_lo, h_hi, F = np.array(w.dec_lo), np.array(w.dec_hi), w.dec_len
    xt = np.append(x, x[-1]) if len(x) % 2 else x
    n2 = len(xt)
    ca, cd = np.zeros(n2 // 2), np.zeros(n2 // 2)
    for o in range(n2 // 2):
        idx = [(F // 2 + 2 * o - j) % n2 for j in range(F)]
        ca[o], cd[o] = h_lo @ xt[idx], h_hi @ xt[idx]
    return ca, cd


def idwt_per(a, d, w):
    lo, hi, F = np.array(w.rec_lo), np.array(w.rec_hi), w.rec_len
    M = len(d)
    a = a[:M]
    y = np.zeros(2 * M)
    for t in range(M):
        i = F // 4 - 1 + t
        idx = [(i - j) % M for j in range(F // 2)]
        y[(2 * t - 1) % (2 * M)] = lo[0::2] @ a[idx] + hi[0::2] @ d[idx]
        y[2 * t] = lo[1::2] @ a[idx] + hi[1::2] @ d[idx]
    return y


def dwt_sym(x, w):
    h_lo, h_hi, F = np.array(w.dec_lo), np.array(w.dec_hi), w.dec_len
    N = len(x)
    xs = lambda m: x[-1 - m] if m < 0 else (x[2 * N - 1 - m] if m >= N else x[m])  # noqa: E731
    n_out = (N + F - 1) // 2
    ca, cd = np.zeros(n_out), np.zeros(n_out)
    for o in range(n_out):
        v = np.array([xs(1 + 2 * o - j) for j in range(F)])
        ca[o], cd[o] = h_lo @ v, h_hi @ v
    return ca, cd


for name, level in (("sym8", 6), ("db4", 4)):
    w = pywt.Wavelet(name)
    for n in (3248, 1968, 3264, 203 * 16):
        x = rng.standard_normal(n)
        ref = pywt.wavedec(x, name, mode="per", level=level)
        a, details = x, []
        for _ in range(level):
            a, d = dwt_per(a, w)
            details.append(d)
        worst = max(worst, np.abs(a - ref[0]).max(), *[np.abs(p - q).max() for p, q in zip(details[::-1], ref[1:])])
        y = a
        for d in details[::-1]:
            y = idwt_per(y, d, w)
        worst = max(worst, np.abs(y[:n] - pywt.waverec(ref, name, mode="per")[:n]).max())

w = pywt.Wavelet("db4")
x = rng.standard_normal(64)
ref = pywt.wavedec(x, "db4", level=3)
a1, d1 = dwt_sym(x, w)
a2, d2 = dwt_sym(a1, w)
worst = max(worst, np.abs(d1 - ref[3]).max(), np.abs(d2 - ref[2]).max())
print(f"worst abs difference vs pywt: {worst:.3e} (tol {TOL}); lengths D1={len(d1)} D2={len(d2)}")
sys.exit(0 if worst < TOL else 1)
