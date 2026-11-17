"""Error of storing the oversampled grid in half precision between FFT axis passes (fp32 arithmetic inside a pass).
Forward: pad (image block of x / phihat, smooth object + white texture) -> 3 axis passes, each output rounded to the
storage format -> the gather reads the rounded grid.  Relative error of the grid vs f64; also after a W=7 gather.
Formats: fp32, fp16 (per-pass global scale so the max |value| maps to 2^14), bf16 (truncate mantissa to 8 bits)."""
import numpy as np
rng = np.random.default_rng(5)
N, sig = 80, 1.25
n = int(sig * N)
r = np.arange(-N // 2, N // 2)
X, Y, Z = np.meshgrid(r, r, r, indexing="ij")
obj = 1 / (1 + np.exp((np.sqrt((X / 34.) ** 2 + (Y / 30.) ** 2 + (Z / 36.) ** 2) - 1) * 10))
x = obj * (1 + 0.2 * (rng.standard_normal(obj.shape) + 1j * rng.standard_normal(obj.shape))) * np.exp(1j * 2.0 * np.sin(X / 9.))
def rnd(a, fmt):
    if fmt == "fp32": return a.astype(np.complex64).astype(complex)
    if fmt == "fp16":
        s = 2.0 ** 14 / np.abs(a).max()
        return (np.round((a.real * s)).astype(np.float16).astype(float) * 0 + (a.real * s).astype(np.float16).astype(float) + 1j * (a.imag * s).astype(np.float16).astype(float)) / s
    if fmt == "bf16":
        def bf(v):
            v32 = v.astype(np.float32).view(np.uint32)
            v32 = ((v32 + 0x7FFF + ((v32 >> 16) & 1)) & 0xFFFF0000).astype(np.uint32)
            return v32.view(np.float32).astype(float)
        return bf(a.real) + 1j * bf(a.imag)
g = np.zeros((n,) * 3, complex); idx = np.mod(r, n); g[np.ix_(idx, idx, idx)] = x
ref = np.fft.fftn(g)
for fmt in ["fp32", "fp16", "bf16"]:
    a = rnd(g, fmt)
    for axis in range(3):
        a = rnd(np.fft.fft(a, axis=axis), fmt)
    e = np.linalg.norm(a - ref) / np.linalg.norm(ref)
    # error weighted like a gather from the k-space ball (samples where the trajectory is): central sphere of radius N/2
    kk = np.fft.fftfreq(n) * n
    KX, KY, KZ = np.meshgrid(kk, kk, kk, indexing="ij")
    ball = KX ** 2 + KY ** 2 + KZ ** 2 <= (N / 2) ** 2
    eb = np.linalg.norm((a - ref)[ball]) / np.linalg.norm(ref[ball])
    # sample-weighted: density ~ 1/|k|^2 (centre-dense spiral/yarn)
    wgt = 1 / np.maximum(KX ** 2 + KY ** 2 + KZ ** 2, 1.0) * ball
    ew = np.sqrt((wgt * np.abs(a - ref) ** 2).sum() / (wgt * np.abs(ref) ** 2).sum())
    print(f"{fmt}: grid rel. error {e:.1e}; inside k-ball {eb:.1e}; centre-dense sample weighting {ew:.1e}")
