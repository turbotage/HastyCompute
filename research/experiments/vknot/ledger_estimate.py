"""Knot-mode memory plan (the formulas of hasty_recon::vknot::plan) for configurations not run: 2 mm full scan and the
1 mm heart box, K = 96 knots (6 × 16). Measured inputs: sample tables 210 B/sample (order 3, W 10; smoke4mm log),
31 virtual coils at ≥ 99.9 % (ph0001 calibration Gram), operator = coil maps + 7 encodes' segment diagonals + the fine-
grid slot (cb · L channels) + FFT workspace (≈ 1 GiB).
  python -I ledger_estimate.py"""
GIB = 2 ** 30
def plan(name, nv, nk, nvc, samples_enc, ne, L, ge, cb, keep=1):
    m = samples_enc / keep
    stack = nk * nv * 20 + 64
    op = nvc * nv * 8 + ne * L * nv * 8 + cb * L * ge * 8 + GIB
    img = nk * nv * 8
    data = nvc * m * 8
    tab = 210 * m
    dev_hostall = op + img + data + data + 2 * tab + 6 * 3 * nv * 4 * 2 + GIB  # host state, data staged, tables streamed
    dev_devstate = op + img + data + ne * data + ne * tab + 3 * stack + GIB  # device state, all resident (prev on host)
    host_hoststate = 4 * stack + 2 * GIB
    print(f"{name}: nv {nv/1e6:.2f} M, K {nk}, {nvc} coils, {m/1e6:.1f} M samples per encode; stack {stack/GIB:.1f} GiB")
    print(f"   leanest plan (host state, data staged, tables streamed): device {dev_hostall/GIB:.1f} GiB "
          f"[operator {op/GIB:.1f}, knot stack {img/GIB:.1f}, residual + staged data {2*data/GIB:.1f}, table slots {2*tab/GIB:.1f}], host state {host_hoststate/GIB:.1f} GiB")
    print(f"   all-device plan: device {dev_devstate/GIB:.1f} GiB")
# 2 mm full scan: 254 × 144 × 114, 53 144 readouts × 3990 ADC / 7 encodes; fine grid σ 1.25 ≈ 320 × 180 × 144
plan("2 mm full scan", 254 * 144 * 114, 96, 31, 53144 / 7 * 3990, 7, 10, 320 * 180 * 144, 4)
plan("2 mm full scan, --keep 8", 254 * 144 * 114, 96, 31, 53144 / 7 * 3990, 7, 10, 320 * 180 * 144, 4, keep=8)
# 1 mm heart box: 198 × 189 × 228, 9891 readouts × 5720 ADC / 7 encodes, 22 virtual coils (R0), L 11
plan("1 mm heart box", 198 * 189 * 228, 96, 22, 9891 / 7 * 5720, 7, 11, 250 * 240 * 288, 1)
plan("1 mm heart box, 4 × 16 knots", 198 * 189 * 228, 64, 22, 9891 / 7 * 5720, 7, 11, 250 * 240 * 288, 1)
