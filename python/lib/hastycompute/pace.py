"""GPU pacer for Python GPU loops (training, phantom tools): the same environment as the Rust `hasty_cuda::pace`.

    from hastycompute.pace import pace
    for it in ...:
        ...one GPU step...
        pace()            # a no-op unless HASTY_GPU_DUTY or HASTY_GPU_WATTS is set

- unset: full speed; `pace()` is one global check.
- HASTY_GPU_DUTY=f (0 < f < 1): the GPU may be busy a fraction f of the time: at each call torch.cuda.synchronize()
  (when torch is loaded) and a sleep of busy · (1 − f) / f, busy = wall time since the end of the last sleep.
- HASTY_GPU_WATTS=W: a background thread reads the board power every 0.5 s (pynvml if installed, else nvidia-smi),
  averages it over HASTY_GPU_WATTS_WINDOW s (default 3.5) and once per second steers the duty, duty ← duty · (W / P̄)^0.3.
- HASTY_GPU_PACE_LOG=1: print the duty / power every few seconds.
"""
import os
import subprocess
import sys
import threading
import time

_state = None  # None: not read yet; False: off; dict: active


def _power_reader():
    try:
        import pynvml  # noqa: PLC0415

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(int((os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",")[0]))
        return lambda: pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
    except Exception:
        idx = (os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",")[0]

        def smi():
            out = subprocess.run(["nvidia-smi", "-i", idx, "--query-gpu=power.draw", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
            return float(out.stdout.strip().split()[0])

        return smi


def _init():
    global _state
    duty = os.environ.get("HASTY_GPU_DUTY")
    watts = os.environ.get("HASTY_GPU_WATTS")
    d = float(duty) if duty else None
    w = float(watts) if watts else None
    if not ((d is not None and 0 < d < 1) or (w is not None and w > 0)):
        _state = False
        return
    st = {"duty": d if d is not None and 0 < d < 1 else 0.7, "last": time.monotonic(), "log": os.environ.get("HASTY_GPU_PACE_LOG", "0") != "0"}
    print(f"GPU pacer: {'target %.0f W' % w if w else 'duty %.2f' % st['duty']}", file=sys.stderr, flush=True)
    if w:
        read = _power_reader()

        win = float(os.environ.get("HASTY_GPU_WATTS_WINDOW", "3.5"))

        def loop():
            ring, tc, tl = [], time.monotonic(), time.monotonic()
            nwin = max(1, int(round(win / 0.5)))
            while True:
                time.sleep(0.5)
                try:
                    ring.append(read())
                except Exception:
                    continue
                ring = ring[-nwin:]
                mean = sum(ring) / len(ring)
                if time.monotonic() - tc >= 1.0 and 2 * len(ring) >= nwin:
                    st["duty"] = min(1.0, max(0.05, st["duty"] * (w / max(mean, 1.0)) ** 0.3))
                    tc = time.monotonic()
                if st["log"] and time.monotonic() - tl > 5:
                    print(f"GPU pacer: {mean:.0f} W over {win:.1f} s (target {w:.0f}), duty {st['duty']:.2f}", file=sys.stderr, flush=True)
                    tl = time.monotonic()

        threading.Thread(target=loop, daemon=True).start()
    _state = st


def pace():
    """pacing point: a no-op unless configured"""
    if _state is False:
        return
    if _state is None:
        _init()
        if _state is False:
            return
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()
    busy = time.monotonic() - _state["last"]
    d = _state["duty"]
    if d < 1.0:
        s = min(busy * (1.0 - d) / d, 5.0)
        if s > 1e-4:
            time.sleep(s)
    _state["last"] = time.monotonic()
