"""Client of the Rust hasty server (rustlib/crates/hasty-server, proto hasty.v2). Values use the block-compressed
hasty value format (hastycompute.value_format); numpy arrays / torch tensors, lists, tuples, dicts, scalars.

    from hastycompute.v2 import Client, show
    c = Client()                                  # HASTY_SERVER or https://localhost:8443, CA from data/hasty/certs
    i = c.put(np.random.rand(32, 64, 64), name="noise")
    x = c.get(i, slice="[0, :, :]")
    url = show(volume, name="recon")              # publish + switch open viewers to it
    url = show_flow(vel, mag, spacing=(2e-3,) * 3, dt=0.045, name="recon")   # 4D flow viewer (flow/)
Ids are 32-character hex strings. Stubs: ./generate.sh.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import grpc

from .. import value_format as vf
from . import hasty_pb2 as pb
from . import hasty_pb2_grpc as pbg

REPO = Path(__file__).resolve().parents[4]
DEFAULT_CA = REPO / "data/hasty/certs/ca.pem"


def _id(h: str):
    return pb.Id(value=bytes.fromhex(h))


def _hex(i) -> str:
    return i.value.hex()


def _entry(e):
    return {"id": _hex(e.id), "name": e.name, "summary": json.loads(e.summary or "null"), "metadata": json.loads(e.metadata or "{}"),
            "nbytes": e.nbytes, "created_ms": e.created_ms, "updated_ms": e.updated_ms}


class Client:
    def __init__(self, url: str | None = None, ca: str | os.PathLike | None = None):
        url = url or os.environ.get("HASTY_SERVER", "https://localhost:8443")
        self.url = url.rstrip("/")
        target = self.url.split("://", 1)[-1]
        opts = [("grpc.max_receive_message_length", 1 << 30), ("grpc.max_send_message_length", 1 << 30)]
        if self.url.startswith("http://"):
            self.channel = grpc.insecure_channel(target, options=opts)
        else:
            ca = Path(ca or os.environ.get("HASTY_CA", DEFAULT_CA))
            self.channel = grpc.secure_channel(target, grpc.ssl_channel_credentials(ca.read_bytes()), options=opts)
        self.stub = pbg.HastyStub(self.channel)

    def close(self):
        self.channel.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def put(self, value, name: str = "", metadata: dict | None = None, **encoding) -> str:
        """upload (streamed, compressed: codec="lz4"|"zstd"|"none", level, delta, block_size)"""
        head = pb.PutRequest(header=pb.PutHeader(name=name, metadata=json.dumps(metadata or {})))

        def msgs():
            yield head
            for c in vf.iter_encode(value, **encoding):
                yield pb.PutRequest(data=c)

        return _hex(self.stub.Put(msgs()).id)

    def write(self, id: str, value, slice: str = "", **encoding):
        """overwrite an entry, or the part of it selected by `slice`"""
        head = pb.PutRequest(header=pb.PutHeader(id=_id(id), slice=slice))

        def msgs():
            yield head
            for c in vf.iter_encode(value, **encoding):
                yield pb.PutRequest(data=c)

        self.stub.Put(msgs())

    def get(self, id: str, slice: str = "", codec: str = "lz4", delta: bool = False):
        stream = self.stub.Get(pb.GetRequest(id=_id(id), slice=slice, encoding=pb.Encoding(codec=codec, delta=delta)))
        return vf.decode_chunks(c.data for c in stream)

    def list(self) -> list[dict]:
        return [_entry(e) for e in self.stub.List(pb.Empty()).entries]

    def info(self, id: str) -> dict:
        return _entry(self.stub.Info(_id(id)))

    def delete(self, id: str):
        self.stub.Delete(_id(id))

    def set_metadata(self, id: str, metadata: dict) -> dict:
        return _entry(self.stub.SetMetadata(pb.MetadataRequest(id=_id(id), metadata=json.dumps(metadata))))

    def commands(self) -> dict:
        return {c.name: c.doc for c in self.stub.Commands(pb.Empty()).commands}

    def execute(self, command: str, inputs=(), options: dict | None = None, name: str = "") -> tuple[list[str], str]:
        r = self.stub.Execute(pb.ExecuteRequest(command=command, options=json.dumps(options or {}), inputs=[_id(i) for i in inputs], name=name))
        return [_hex(i) for i in r.outputs], r.message

    def show(self, id: str, view: str = "orthoslicer"):
        self.stub.Show(pb.ShowRequest(id=_id(id), view=view))

    def watch(self):
        """iterator of events {kind, id, name, message}"""
        for e in self.stub.Watch(pb.Empty()):
            yield {"kind": e.kind, "id": _hex(e.id) if e.id.value else "", "name": e.name, "message": e.message}

    def viewer_url(self, id: str, view: str = "orthoslicer") -> str:
        return f"{self.url}/{view}/index.html?id={id}"


_default = None


def show(value, name: str = "", view: str = "orthoslicer", **metadata) -> str:
    """publish `value` to the server (HASTY_SERVER or localhost) and switch open viewers to it; returns the URL"""
    global _default
    if _default is None:
        _default = Client()
    i = _default.put(value, name=name, metadata=metadata)
    _default.show(i, view)
    return _default.viewer_url(i, view)


def flow_value(vel, mag=None, spacing=(2e-3, 2e-3, 2e-3), dt=0.04, venc=None, mask=None, layout="hasty") -> dict:
    """The flow value of the flow viewer (schema: rustlib/crates/hasty-server/src/flow.rs).

    layout "hasty": numpy index order x first — vel [X, Y, Z, 3, F], mag [X, Y, Z, F], mask [X, Y, Z] (the arrays
    recon_p2np writes, Fortran npy); layout "c": already C-order [F, 3, Z, Y, X], [F, Z, Y, X], [Z, Y, X].
    spacing in m, dt in s per frame (the cycle is periodic), venc in m/s."""
    import numpy as np

    def c(a, nd):
        a = np.asarray(a)
        if a.ndim == nd - 1:  # one frame
            a = a[..., None] if layout == "hasty" else a[None]
        return np.ascontiguousarray(a.transpose(tuple(range(a.ndim))[::-1]) if layout == "hasty" else a)

    v = {"kind": "flow4d", "vel": c(vel, 5).astype(np.float32)}
    if mag is not None:
        v["mag"] = np.abs(c(mag, 4)).astype(np.float32)
    v["spacing"] = [float(x) for x in spacing]
    v["dt"] = float(dt)
    if venc is not None:
        v["venc"] = float(venc)
    if mask is not None:
        m = np.asarray(mask)
        v["mask"] = np.ascontiguousarray(m.transpose(tuple(range(m.ndim))[::-1]) if layout == "hasty" else m).astype(np.uint8)
    return v


def show_flow(vel, mag=None, spacing=(2e-3, 2e-3, 2e-3), dt=0.04, venc=None, mask=None, name="", layout="hasty", **metadata) -> str:
    """publish a 4D flow value (see flow_value) and switch open flow viewers to it; returns the viewer URL"""
    v = flow_value(vel, mag, spacing, dt, venc, mask, layout)
    return show(v, name=name, view="flow", kind="flow4d", **metadata)
