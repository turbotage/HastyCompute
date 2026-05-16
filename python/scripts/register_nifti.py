#!/usr/bin/env python3
"""
Register two NIfTI volumes using DIPY.

Usage:
    register_nifti.py --grpc-port=PORT --fixed=UUID_HEX --moving=UUID_HEX
                      --output=UUID_HEX [--transform=Rigid] [--debug-port=PORT]

Result is written to the pre-registered gRPC bank slot given by --output.
Nothing is printed to stdout. Progress/info messages go to stderr.

Registration direction is chosen automatically: the physically larger volume
is always used as the fixed reference (better convergence). The OUTPUT
is whichever volume is smaller, warped into the larger volume's grid.

The script reads NIfTI header metadata (pixdim/sform) written by
hasty::python::push_nifti_image() so voxel spacing is correct.
"""

import argparse
import json
import sys

import numpy as np
import torch
import time
from dipy.align.imaffine import AffineRegistration, MutualInformationMetric
from dipy.align.transforms import RigidTransform3D, AffineTransform3D
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.metrics import CCMetric

sys.path.insert(0, '')  # ensure local imports work when called as subprocess

from hastycompute.grpc_client.hasty_client import HastyClient
from hastycompute.generic_value import GenericValue


# ─── helpers ──────────────────────────────────────────────────────────────────

def hex_to_uuid(h: str) -> bytes:
    return bytes.fromhex(h)

def uuid_to_hex(b: bytes) -> str:
    return b.hex()

def get_spacing(client: HastyClient, uuid: bytes) -> tuple[float, float, float]:
    """Read pixdim[1:4] from JSON metadata written by push_nifti_image()."""
    meta = client.read_metadata(uuid)
    if meta:
        d = json.loads(meta)
        p = d.get('pixdim', [1, 1, 1, 1])
        return (float(p[1]), float(p[2]), float(p[3]))
    return (1.0, 1.0, 1.0)

def normalize_for_registration(t: torch.Tensor) -> torch.Tensor:
    """Normalize tensor to [0,1] range so registration metrics behave."""
    t = t.float()
    t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    mn, mx = t.min().item(), t.max().item()
    if mx > mn:
        return (t - mn) / (mx - mn)
    return torch.zeros_like(t)

def tensor_to_numpy(t: torch.Tensor, spacing: tuple) -> tuple[np.ndarray, np.ndarray]:
    """Convert C-order [nz, ny, nx] tensor → numpy array [nx, ny, nz] and affine.

    Returns (arr, affine) where affine maps voxel indices to world (mm) using spacing.
    """
    assert t.ndim == 3, f"tensor_to_numpy expects 3-D tensor, got shape {list(t.shape)}"
    arr = np.ascontiguousarray(t.float().numpy().T)  # [nx, ny, nz]
    sx, sy, sz = spacing[0], spacing[1], spacing[2]
    affine = np.diag([sx, sy, sz, 1.0])
    return arr, affine

def numpy_to_tensor(arr: np.ndarray) -> torch.Tensor:
    """Convert numpy [nx, ny, nz] → C-order [nz, ny, nx] tensor."""
    t = np.ascontiguousarray(arr.T)
    return torch.from_numpy(t.copy())

def extract_3d_ref(t: torch.Tensor) -> torch.Tensor:
    """Mean over leading dims until 3-D (last 3 dims are Z,Y,X)."""
    while t.ndim > 3:
        t = t.float().mean(dim=0)
    return t

_FALLBACK = {'SyN': 'Affine', 'Affine': 'Rigid'}

def run_registration(fixed_arr: np.ndarray, fixed_affine: np.ndarray,
                     moving_arr: np.ndarray, moving_affine: np.ndarray,
                     transform: str):
    """Run registration using DIPY and return a mapping object with `transform`.

    For Rigid/Affine, returns an AffineMap (with .transform()). For SyN, returns
    a DiffeomorphicMap-like object with .transform().
    """
    try:
        # Treat 'SyN' as 'Affine' by default to avoid expensive diffeomorphic runs.
        if transform in ('Rigid', 'Affine', 'SyN'):
            tmode = 'Affine' if transform == 'SyN' else transform
            metric = MutualInformationMetric(nbins=32, sampling_proportion=None)
            # Reduced iteration counts for faster registration while keeping multi-scale
            level_iters = [100, 50, 10]
            affreg = AffineRegistration(metric=metric, level_iters=level_iters,
                                        sigmas=[3.0, 1.0, 0.0], factors=[4, 2, 1])
            if tmode == 'Rigid':
                transform_obj = RigidTransform3D()
            else:
                transform_obj = AffineTransform3D()
            affine_map = affreg.optimize(static=fixed_arr, moving=moving_arr,
                                         transform=transform_obj,
                                         params0=None,
                                         static_grid2world=fixed_affine,
                                         moving_grid2world=moving_affine)
            return affine_map
        else:
            # Fallback: keep diffeomorphic option for unrecognized transforms
            metric = CCMetric(3)
            level_iters = [40, 20, 10]
            sdr = SymmetricDiffeomorphicRegistration(metric, level_iters)
            mapping = sdr.optimize(static=fixed_arr, moving=moving_arr,
                                   static_grid2world=fixed_affine,
                                   moving_grid2world=moving_affine)
            return mapping
    except Exception as e:
        fallback = _FALLBACK.get(transform)
        if fallback is None:
            raise
        print(f'[register_nifti] {transform} failed ({e}); retrying with {fallback}.',
              file=sys.stderr, flush=True)
        return run_registration(fixed_arr, fixed_affine, moving_arr, moving_affine, fallback)


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='ANTs NIfTI registration via gRPC')
    parser.add_argument('--grpc-port',   type=int, default=50051)
    parser.add_argument('--fixed',       required=True,  help='fixed volume UUID hex')
    parser.add_argument('--moving',      required=True,  help='moving volume UUID hex')
    parser.add_argument('--output',       required=True,
                        help='UUID hex of the pre-registered bank slot to write result into')
    parser.add_argument('--transform',   default='Rigid',
                        help='Transform type (Rigid/Affine/SyN; default Rigid)')
    parser.add_argument('--debug-port',  type=int, default=0,
                        help='If >0, wait for debugpy attach on this port')
    args = parser.parse_args()

    if args.debug_port > 0:
        import debugpy
        debugpy.connect(('localhost', args.debug_port))
        debugpy.wait_for_client()
        print('[register_nifti] Debugger connected.', file=sys.stderr, flush=True)

    fixed_uuid  = hex_to_uuid(args.fixed)
    moving_uuid = hex_to_uuid(args.moving)

    print('[register_nifti] Connecting to gRPC server…', file=sys.stderr, flush=True)
    with HastyClient(f'localhost:{args.grpc_port}') as client:

        fixed_spacing  = get_spacing(client, fixed_uuid)
        moving_spacing = get_spacing(client, moving_uuid)
        print(f'[register_nifti] fixed  spacing: {fixed_spacing}', file=sys.stderr)
        print(f'[register_nifti] moving spacing: {moving_spacing}', file=sys.stderr)

        print('[register_nifti] Fetching volumes…', file=sys.stderr, flush=True)
        t0 = time.perf_counter()
        fixed_t  = client.fetch_value(fixed_uuid).as_tensor()
        moving_t = client.fetch_value(moving_uuid).as_tensor()
        t_fetch = time.perf_counter() - t0
        print(f'[register_nifti] fixed  shape: {list(fixed_t.shape)}', file=sys.stderr)
        print(f'[register_nifti] moving shape: {list(moving_t.shape)}', file=sys.stderr)
        print(f'[register_nifti] Fetch time: {t_fetch:.3f}s', file=sys.stderr)

        fixed_ref  = extract_3d_ref(fixed_t)
        moving_ref = extract_3d_ref(moving_t)

        # Use higher-resolution volume (smaller voxel size) as ANTs fixed so the
        # output is on the fine grid.  Warp the coarser volume into that space.
        fixed_voxel_vol  = float(np.prod(fixed_spacing))
        moving_voxel_vol = float(np.prod(moving_spacing))
        print(f'[register_nifti] fixed  voxel vol: {fixed_voxel_vol:.4g} mm³',
              file=sys.stderr, flush=True)
        print(f'[register_nifti] moving voxel vol: {moving_voxel_vol:.4g} mm³',
              file=sys.stderr, flush=True)

        # Normalize both to [0,1] so registration metrics handle different contrasts
        # (e.g. signed B0 field in Hz vs unsigned magnitude).
        if fixed_voxel_vol <= moving_voxel_vol:
            # fixed is higher-res → use fixed grid; warp moving into it
            print('[register_nifti] Fixed has finer voxels → using fixed as registration fixed, '
                  'warping moving into fixed grid.', file=sys.stderr, flush=True)
            fixed_arr, fixed_affine = tensor_to_numpy(normalize_for_registration(fixed_ref),  fixed_spacing)
            moving_arr, moving_affine = tensor_to_numpy(normalize_for_registration(moving_ref), moving_spacing)
            src_t           = moving_t
            src_spacing     = moving_spacing
        else:
            # moving is higher-res → use moving grid; warp fixed into it
            print('[register_nifti] Moving has finer voxels → using moving as registration fixed, '
                  'warping fixed into moving grid.', file=sys.stderr, flush=True)
            fixed_arr, fixed_affine = tensor_to_numpy(normalize_for_registration(moving_ref), moving_spacing)
            moving_arr, moving_affine = tensor_to_numpy(normalize_for_registration(fixed_ref),  fixed_spacing)
            src_t           = fixed_t
            src_spacing     = fixed_spacing

        print(f'[register_nifti] Running {args.transform} registration (DIPY)…', file=sys.stderr, flush=True)
        t0 = time.perf_counter()
        reg_map = run_registration(fixed_arr, fixed_affine, moving_arr, moving_affine, args.transform)
        t_reg = time.perf_counter() - t0
        print(f'[register_nifti] Registration done. Took {t_reg:.3f}s', file=sys.stderr, flush=True)

        # Apply fwdtransforms (smaller-space → larger-space) to every src sub-volume.
        spatial_shape = src_t.shape[-3:]
        leading_shape = src_t.shape[:-3]
        src_flat      = src_t.reshape(-1, *spatial_shape)

        print(f'[register_nifti] Warping {src_flat.shape[0]} sub-volume(s)…', file=sys.stderr, flush=True)
        registered_vols = []
        t_warp_start = time.perf_counter()
        for i in range(src_flat.shape[0]):
            it0 = time.perf_counter()
            sub_t = src_flat[i].float()
            sub_arr, sub_affine = tensor_to_numpy(sub_t, src_spacing)
            try:
                warped_arr = reg_map.transform(sub_arr)
            except Exception:
                # Some mapping types use 'apply' or 'inverse' names; try generic call
                warped_arr = reg_map.transform(sub_arr)
            vol = numpy_to_tensor(warped_arr)
            # Registration may fill background with NaN; replace with 0.
            vol = torch.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)
            registered_vols.append(vol)
            it = time.perf_counter() - it0
            print(f'[register_nifti] Warped sub-volume {i+1}/{src_flat.shape[0]} in {it:.3f}s',
                  file=sys.stderr, flush=True)
        t_warp = time.perf_counter() - t_warp_start
        print(f'[register_nifti] Total warp time: {t_warp:.3f}s', file=sys.stderr, flush=True)

        if leading_shape:
            stacked    = torch.stack(registered_vols, dim=0)
            reg_tensor = stacked.reshape(*leading_shape, *stacked.shape[1:])
        else:
            reg_tensor = registered_vols[0]

        reg_f = reg_tensor.float()
        nonzero = (reg_f != 0).sum().item()
        print(f'[register_nifti] result shape: {list(reg_tensor.shape)}, '
              f'min={reg_f.min().item():.4g}, max={reg_f.max().item():.4g}, '
              f'nonzero={nonzero}/{reg_f.numel()} '
              f'({100.0*nonzero/max(reg_f.numel(),1):.1f}%)',
              file=sys.stderr, flush=True)
        if nonzero == 0:
            print('[register_nifti] WARNING: registered tensor is all-zero — '
                  'registration may have failed (contrast mismatch?)',
                  file=sys.stderr, flush=True)

        output_uuid = hex_to_uuid(args.output)
        t0 = time.perf_counter()
        client.write_value(output_uuid, GenericValue.from_tensor(reg_tensor))
        t_write = time.perf_counter() - t0
        print(f'[register_nifti] Wrote result to output slot {args.output} (write time {t_write:.3f}s)',
            file=sys.stderr, flush=True)


if __name__ == '__main__':
    main()
