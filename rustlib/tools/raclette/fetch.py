"""Fetch parts of the RACLETTE dataset (Dirix, Buoso, Kozerke; ETH Zurich; DOI 10.3929/ethz-c-000799752; data licence
CC BY-NC 4.0) from its public WebDAV share into the repository (data/hasty/external/raclette_data).

    python -I fetch.py meta                                   # meta.json of every subject, all variants (~1 MB)
    python -I fetch.py data <variant> <subject> [--gt]        # Data/ (MRI frames + coil maps), optionally GroundTruth/
    python -I fetch.py sizes <variant> <subject>              # remote sizes only

The public share token is read from the RACLETTE repository clone (data/hasty/external/raclette).
"""
import json
import re
import subprocess
import sys
from pathlib import Path

from webdav4.client import Client

REPO = Path(__file__).resolve().parents[3]  # HastyCompute
CLONE = REPO / "data/hasty/external/raclette"
OUT = REPO / "data/hasty/external/raclette_data"
URL = "https://libdrive.ethz.ch/public.php/webdav"
VARIANTS = ["7p_2Venc", "7p_2Venc_Breathing", "7p_2Venc_Super", "13p_2Venc_ICOSA"]


def client():
    src = subprocess.run(["git", "-C", str(CLONE), "show", "HEAD:Tutorials/DataDownload/tutorial_data_download.py"], capture_output=True, text=True, check=True).stdout
    token = re.search(r'REMOTE_TOKEN\s*=\s*"([^"]+)"', src).group(1)
    return Client(URL, auth=(token, ""))


def files(c, remote):
    out = []
    for it in c.ls(remote, detail=True):
        if it["type"] == "directory":
            out += files(c, it["name"])
        else:
            out.append((it["name"], it["content_length"]))
    return out


def get(c, remote, local, size):
    local.parent.mkdir(parents=True, exist_ok=True)
    if local.exists() and local.stat().st_size == size:
        return False
    c.download_file(remote, str(local))
    return True


def main():
    c = client()
    cmd = sys.argv[1]
    if cmd == "meta":
        for v in VARIANTS:
            subs = sorted(it["name"].split("/")[-1] for it in c.ls(f"/{v}", detail=True) if it["type"] == "directory")
            for s in subs:
                for it in c.ls(f"/{v}/{s}", detail=True):
                    if it["type"] == "file" and it["name"].endswith(".json"):
                        get(c, it["name"], OUT / v / s / Path(it["name"]).name, it["content_length"])
            print(f"{v}: {len(subs)} subjects")
        return
    v, s = sys.argv[2], sys.argv[3]
    parts = ["Data"] + (["GroundTruth"] if "--gt" in sys.argv or cmd == "sizes" else [])
    for p in parts:
        fl = files(c, f"/{v}/{s}/{p}")
        tot = sum(n for _, n in fl)
        print(f"{v}/{s}/{p}: {len(fl)} files, {tot / 1e9:.2f} GB")
        if cmd == "data":
            new = sum(get(c, r, OUT / v / s / p / Path(r).relative_to(f"/{v}/{s}/{p}" if r.startswith("/") else f"{v}/{s}/{p}"), n) for r, n in fl)
            print(f"  downloaded {new} new files")


if __name__ == "__main__":
    main()
