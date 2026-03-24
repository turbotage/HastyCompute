"""
Explicit integration test: push two tensors, add them on the server,
fetch the result, compare against local torch addition.

Usage:
  1. Build and run HastyTest (starts the gRPC server, prints READY)
  2. python test_add.py
"""

import atexit
import signal
import subprocess
import sys
import time
import pathlib
import torch

from hasty_client import HastyClient
from generic_value import GenericValue


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_server_exe() -> pathlib.Path:
    """Look for HastyTest in common build install locations."""
    candidates = [
        pathlib.Path(__file__).parents[2] / "out/build/Config-ClangDebug-Linux/test-build/HastyTest",
        pathlib.Path(__file__).parents[2] / "out/build/Config-ClangRelWithDebInfo-Linux/test-build/HastyTest",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find HastyTest binary. Build the project first, "
        "or start HastyTest manually and re-run with --no-server."
    )


def _kill_proc(proc: subprocess.Popen) -> None:
    """Terminate then kill a process, ignoring errors if already dead."""
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def start_server(exe: pathlib.Path) -> subprocess.Popen:
    proc = subprocess.Popen(
        [str(exe)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Register cleanup immediately — before waiting for READY — so that
    # even a timeout or KeyboardInterrupt cannot leave the process orphaned.
    atexit.register(_kill_proc, proc)

    deadline = time.time() + 15.0
    while time.time() < deadline:
        line = proc.stdout.readline()
        if "READY" in line:
            return proc
        if proc.poll() is not None:
            raise RuntimeError(
                f"HastyTest exited early:\n{proc.stderr.read()}"
            )
    raise TimeoutError("HastyTest did not print READY within 15 seconds")


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def test_tensor_add(client: HastyClient) -> None:
    print("Running test_tensor_add...")

    # Test with float32
    a = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    b = torch.tensor([10.0, 20.0, 30.0, 40.0], dtype=torch.float32)
    expected = a + b

    uuid_a = client.push_value(GenericValue.from_tensor(a))
    uuid_b = client.push_value(GenericValue.from_tensor(b))

    output_uuids = client.execute(0, [uuid_a, uuid_b])
    assert len(output_uuids) == 1, f"Expected 1 output, got {len(output_uuids)}"

    result_gv = client.fetch_value(output_uuids[0])
    assert result_gv.is_tensor(), "Result is not a tensor"
    result = result_gv.as_tensor()

    assert torch.allclose(result, expected), \
        f"Mismatch!\n  expected: {expected}\n  got:      {result}"
    print(f"  float32 add OK: {result.tolist()}")

    # Test with a 2-D complex64 tensor
    c = torch.randn(3, 4, dtype=torch.complex64)
    d = torch.randn(3, 4, dtype=torch.complex64)
    expected_cd = c + d

    uuid_c = client.push_value(GenericValue.from_tensor(c))
    uuid_d = client.push_value(GenericValue.from_tensor(d))

    [uuid_result] = client.execute(0, [uuid_c, uuid_d])
    result_cd = client.fetch_value(uuid_result).as_tensor()

    assert torch.allclose(result_cd, expected_cd), \
        "complex64 2-D add mismatch"
    print(f"  complex64 2-D add OK: shape {list(result_cd.shape)}")

    print("test_tensor_add PASSED")


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def _gib(n_bytes: int) -> float:
    return n_bytes / (1024 ** 3)


def _bw(n_bytes: int, elapsed_s: float) -> str:
    return f"{_gib(n_bytes) / elapsed_s:.2f} GiB/s"


def benchmark_large_add(client: HastyClient, size_gib: float = 3.0) -> None:
    item_bytes = 4  # float32
    n_elements = int(size_gib * (1024 ** 3) / item_bytes)
    actual_bytes = n_elements * item_bytes
    actual_gib = _gib(actual_bytes)

    print(f"\n{'='*60}")
    print(f"Benchmark: float32 add, {actual_gib:.2f} GiB each tensor")
    print(f"{'='*60}")

    a = torch.zeros(n_elements, dtype=torch.float32)
    b = torch.ones(n_elements, dtype=torch.float32)
    print("Tensors allocated.")

    t0 = time.perf_counter()
    uuid_a = client.push_value(GenericValue.from_tensor(a))
    push_a = time.perf_counter() - t0
    print(f"  push a:   {push_a:.3f} s  ({_bw(actual_bytes, push_a)})")

    t0 = time.perf_counter()
    uuid_b = client.push_value(GenericValue.from_tensor(b))
    push_b = time.perf_counter() - t0
    print(f"  push b:   {push_b:.3f} s  ({_bw(actual_bytes, push_b)})")

    t0 = time.perf_counter()
    [uuid_result] = client.execute(0, [uuid_a, uuid_b])
    add_elapsed = time.perf_counter() - t0
    print(f"  add:      {add_elapsed:.3f} s  ({_bw(actual_bytes, add_elapsed)})")

    # Free inputs from the bank now that execute is done
    client.delete_value(uuid_a)
    client.delete_value(uuid_b)
    del a, b  # also free Python-side tensors

    t0 = time.perf_counter()
    result_gv = client.fetch_value(uuid_result)
    fetch_elapsed = time.perf_counter() - t0
    print(f"  fetch:    {fetch_elapsed:.3f} s  ({_bw(actual_bytes, fetch_elapsed)})")

    client.delete_value(uuid_result)

    result = result_gv.as_tensor()
    assert result.shape == torch.Size([n_elements])
    print(f"\n  Result verified (shape={list(result.shape)}, dtype={result.dtype})")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    no_server = "--no-server" in sys.argv

    if not no_server:
        exe = _find_server_exe()
        print(f"Starting server: {exe}")
        start_server(exe)  # atexit cleanup registered inside
        print("Server ready.")

    # Also kill on SIGTERM so `kill <pid>` from a shell cleans up properly
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    address = "localhost:50051"
    #address = "unix:///tmp/hasty.sock"

    with HastyClient(address) as client:
        test_tensor_add(client)
        benchmark_large_add(client)
    print("\nAll tests PASSED")


if __name__ == "__main__":
    main()
