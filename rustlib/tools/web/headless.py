"""Run a page in headless Chrome (software rendering, no GPU) and wait for a JS expression to become truthy, via the
DevTools protocol over --remote-debugging-pipe (stdlib only). Prints the expression's value, console messages and
exceptions; exit code 0 when the value parses as JSON with "ok": true.

    python -I headless.py <url> [--expr "document.getElementById('selftest')?.textContent"] [--timeout 60]
                             [--chrome-flags "--disable-gpu"] [--screenshot out.png] [--size 1600x1000]   (software WebGPU: "--enable-unsafe-webgpu --use-webgpu-adapter=swiftshader --use-angle=swiftshader")

The Chrome profile lives in rustlib/target/chrome-profile.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

args = sys.argv[1:]
url = args[0]


def opt(n, d):
    return args[args.index(n) + 1] if n in args else d


expr = opt("--expr", "document.getElementById('selftest')?.textContent")
timeout = float(opt("--timeout", "60"))
extra = opt("--chrome-flags", "--disable-gpu").split()
profile = Path(__file__).resolve().parents[2] / "target/chrome-profile"
import fcntl  # noqa: E402

def high(fd):  # move a pipe end to fd ≥ 10 so dup2 onto 3 / 4 in the child cannot clobber the other end
    h = fcntl.fcntl(fd, fcntl.F_DUPFD, 10)
    os.close(fd)
    os.set_inheritable(h, True)
    return h


to_chrome_r, to_chrome_w = (high(f) for f in os.pipe())
from_chrome_r, from_chrome_w = (high(f) for f in os.pipe())
chrome = subprocess.Popen(
    ["google-chrome", "--headless=new", *extra, "--no-first-run", "--no-default-browser-check", f"--user-data-dir={profile}", "--remote-debugging-pipe", "about:blank"],
    pass_fds=(3, 4, to_chrome_r, from_chrome_w), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=open(profile.parent / "tmp" / "chrome.err", "wb"),
    preexec_fn=lambda: (os.dup2(to_chrome_r, 3), os.dup2(from_chrome_w, 4)),
)
os.close(to_chrome_r)
os.close(from_chrome_w)
wr = os.fdopen(to_chrome_w, "wb", buffering=0)
rd = os.fdopen(from_chrome_r, "rb", buffering=0)
buf = b""
mid = 0
events = []


def send(method, params=None, session=None):
    global mid
    mid += 1
    m = {"id": mid, "method": method, "params": params or {}}
    if session:
        m["sessionId"] = session
    wr.write(json.dumps(m).encode() + b"\0")
    return mid


def recv_until(want_id):
    global buf
    while True:
        while b"\0" not in buf:
            c = rd.read(65536)
            if not c:
                raise RuntimeError("chrome closed the pipe")
            buf += c
        msg, buf = buf.split(b"\0", 1)
        m = json.loads(msg)
        if m.get("id") == want_id:
            if "error" in m:
                raise RuntimeError(m["error"])
            return m.get("result", {})
        if m.get("method") in ("Runtime.consoleAPICalled", "Runtime.exceptionThrown"):
            events.append(m)


def call(method, params=None, session=None):
    return recv_until(send(method, params, session))


rc = 1
try:
    tid = call("Target.createTarget", {"url": "about:blank"})["targetId"]
    sid = call("Target.attachToTarget", {"targetId": tid, "flatten": True})["sessionId"]
    call("Runtime.enable", session=sid)
    call("Page.enable", session=sid)
    if opt("--size", None):
        w, h = (int(v) for v in opt("--size", "").split("x"))
        call("Emulation.setDeviceMetricsOverride", {"width": w, "height": h, "deviceScaleFactor": 1, "mobile": False}, session=sid)
    call("Page.navigate", {"url": url}, session=sid)
    t0 = time.time()
    value = None
    while time.time() - t0 < timeout:
        r = call("Runtime.evaluate", {"expression": expr, "returnByValue": True}, session=sid)
        value = r.get("result", {}).get("value")
        if value:
            try:
                j = json.loads(value)
                if "ok" in j:
                    break
            except (TypeError, ValueError):
                break
        time.sleep(0.25)
    for e in events:
        p = e["params"]
        if e["method"] == "Runtime.consoleAPICalled":
            print(f"console.{p['type']}:", " ".join(str(a.get("value", a.get("description", ""))) for a in p["args"]))
        else:
            d = p["exceptionDetails"]
            print("exception:", d.get("exception", {}).get("description", d.get("text")))
    print(value)
    if opt("--screenshot", None):
        import base64
        shot = call("Page.captureScreenshot", {"format": "png"}, session=sid)
        Path(opt("--screenshot", "")).write_bytes(base64.b64decode(shot["data"]))
        print("screenshot:", opt("--screenshot", ""))
    try:
        rc = 0 if json.loads(value).get("ok") else 1
    except (TypeError, ValueError, AttributeError):
        rc = 1
finally:
    try:
        call("Browser.close")
    except Exception:
        pass
    chrome.wait(timeout=10)
sys.exit(rc)
