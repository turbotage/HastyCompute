#!/usr/bin/env python3
"""
Register two NIfTI volumes using DIPY.

Usage:
    register_nifti.py --grpc-port=PORT --fixed=UUID_HEX --moving=UUID_HEX
                      --output=UUID_HEX --log=UUID_HEX [--transform=Rigid] [--debug-port=PORT]

Result is written to the pre-registered gRPC bank slot given by --output.
Progress/info messages are written to the --log bank slot rather than
printed — run_script() drains that slot into its log file as each line
arrives.

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
try:
    import SimpleITK as sitk
except Exception:
    sitk = None

try:
    import fireants
    from fireants.io.image import Image as FAImage, BatchedImages
    from fireants.registration.rigid import RigidRegistration
    from fireants.registration.affine import AffineRegistration as FAAffineRegistration
except Exception:
    fireants = None

from dipy.align.imaffine import AffineMap, AffineRegistration, MutualInformationMetric
from dipy.align.transforms import RigidTransform3D, AffineTransform3D

sys.path.insert(0, '')  # ensure local imports work when called as subprocess

from hastycompute.grpc_client.hasty_client import HastyClient
from hastycompute.generic_value import GenericValue


# ─── logging ──────────────────────────────────────────────────────────────────
# Routed through the bank's --log slot (read by an after_write_callback on the
# C++ side) instead of stdout/stderr, so script output lands in the server's
# log file rather than being discarded by run_script(). Falls back to stderr
# if called before the client/slot are wired up (e.g. argument errors).

_log_client: HastyClient | None = None
_log_uuid: bytes | None = None

def log(msg: str) -> None:
    if _log_client is not None and _log_uuid is not None:
        _log_client.write_value(_log_uuid, GenericValue.from_string(msg + '\n'))
    else:
        print(msg, file=sys.stderr, flush=True)


# ─── helpers ──────────────────────────────────────────────────────────────────

def hex_to_uuid(h: str) -> bytes:
    return bytes.fromhex(h)

def uuid_to_hex(b: bytes) -> str:
    return b.hex()

def get_affine(client: HastyClient, uuid: bytes) -> np.ndarray:
    """Read the voxel→world affine from NIfTI qform/sform metadata written by
    push_nifti_image(). sform takes priority (matches NIfTI convention); falls
    back to qform, then to a diagonal pixdim-only affine (origin at 0) only if
    neither qform nor sform was set in the source file.
    """
    meta = client.read_metadata(uuid)
    if not meta:
        return np.eye(4)
    d = json.loads(meta)
    if d.get('sform_code', 0) > 0:
        return np.array(d['sform'], dtype=np.float64)
    if d.get('qform_code', 0) > 0:
        return np.array(d['qform'], dtype=np.float64)
    p = d.get('pixdim', [1, 1, 1, 1])
    return np.diag([float(p[1]), float(p[2]), float(p[3]), 1.0])

def affine_spacing(affine: np.ndarray) -> tuple[float, float, float]:
    """Voxel spacing implied by an affine's column norms (handles rotation)."""
    return tuple(float(np.linalg.norm(affine[:3, i])) for i in range(3))

def normalize_for_registration(t: torch.Tensor) -> torch.Tensor:
    """Normalize tensor to [0,1] range so registration metrics behave."""
    t = t.float()
    t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    mn, mx = t.min().item(), t.max().item()
    if mx > mn:
        return (t - mn) / (mx - mn)
    return torch.zeros_like(t)

def tensor_to_numpy(t: torch.Tensor, affine: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert C-order [nz, ny, nx] tensor → numpy array [nx, ny, nz].

    Returns (arr, affine) — affine is passed through unchanged; it's the
    caller's job to supply the real voxel→world affine (see get_affine()).
    """
    assert t.ndim == 3, f"tensor_to_numpy expects 3-D tensor, got shape {list(t.shape)}"
    arr = np.ascontiguousarray(t.float().numpy().T)  # [nx, ny, nz]
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
    a DiffeomorphicMap-like object with .transform(). For ResampleOnly, returns
    a DIPY AffineMap with an identity world-space transform — i.e. no
    optimization, just a resample from moving's grid onto fixed's grid using
    their real qform/sform affines (correct when the volumes are already
    aligned in world space and only differ in grid/spacing, e.g. B0 vs PD from
    the same scan session).
    """
    if transform == 'ResampleOnly':
        log(f'[register_nifti] ResampleOnly: resampling onto fixed grid {fixed_arr.shape} '
            'using qform/sform world-space affines (no registration optimization).')
        return AffineMap(np.eye(4),
                          domain_grid_shape=fixed_arr.shape, domain_grid2world=fixed_affine,
                          codomain_grid_shape=moving_arr.shape, codomain_grid2world=moving_affine)

    # Prefer FireAnts (GPU-accelerated) if available and SimpleITK present.
    if fireants is not None and sitk is not None and transform in ('Rigid', 'Affine', 'SyN'):
        try:
            # Treat SyN as Affine to match current behavior (avoid expensive nonlinear runs)
            tmode = 'Affine' if transform == 'SyN' else transform

            # Create SimpleITK images from numpy arrays. The incoming arrays are [nx, ny, nz].
            # SimpleITK expects arrays in (z,y,x) ordering, so transpose back before creating.
            sx, sy, sz = fixed_affine[0, 0], fixed_affine[1, 1], fixed_affine[2, 2]
            mx, my, mz = moving_affine[0, 0], moving_affine[1, 1], moving_affine[2, 2]

            fixed_itk = sitk.GetImageFromArray(fixed_arr.T)
            fixed_itk.SetSpacing((sx, sy, sz))
            moving_itk = sitk.GetImageFromArray(moving_arr.T)
            moving_itk.SetSpacing((mx, my, mz))

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            fa_fixed = FAImage(fixed_itk, device=device)
            fa_moving = FAImage(moving_itk, device=device)
            fixed_batched = BatchedImages(fa_fixed)
            moving_batched = BatchedImages(fa_moving)

            # Multiresolution settings similar to DIPY's reduced iterations
            iterations = [15, 8, 2]
            scales = [4, 2, 1]

            if tmode == 'Rigid':
                reg = RigidRegistration(scales, iterations, fixed_batched, moving_batched,
                                        loss_type='cc', optimizer='Adam', optimizer_lr=0.03)
            else:
                reg = FAAffineRegistration(scales, iterations, fixed_batched, moving_batched,
                                           loss_type='cc', optimizer='Adam', optimizer_lr=0.03)

            reg.optimize()

            # Adapter to emulate DIPY's affine_map.transform(arr) API expected by caller.
            class _FAAdapter:
                def __init__(self, reg, fixed_batched, moving_spacing, device):
                    self.reg = reg
                    self.fixed_batched = fixed_batched
                    self.moving_spacing = moving_spacing
                    self.device = device

                def transform(self, arr: np.ndarray) -> np.ndarray:
                    # arr: [nx, ny, nz] -> create SimpleITK image and wrap
                    itk_img = sitk.GetImageFromArray(arr.T)
                    itk_img.SetSpacing((self.moving_spacing[0], self.moving_spacing[1], self.moving_spacing[2]))
                    fa_img = FAImage(itk_img, device=self.device)
                    batched = BatchedImages(fa_img)
                    warped = self.reg.evaluate(self.fixed_batched, batched)
                    # warped: torch tensor [N, C, *dims]. Move to cpu and convert to numpy
                    warped_np = warped.detach().cpu().numpy()
                    # Expect warped_np shape [1,1,Z,Y,X] or [1,1,X,Y,Z]. Normalize by trying both.
                    w = warped_np[0, 0]
                    if w.ndim == 3:
                        # If shape is (Z,Y,X) convert to (X,Y,Z)
                        if w.shape == (fixed_arr.shape[2], fixed_arr.shape[1], fixed_arr.shape[0]):
                            return np.ascontiguousarray(w.transpose(2, 1, 0))
                        # If already (X,Y,Z) just return
                        if w.shape == fixed_arr.shape:
                            return np.ascontiguousarray(w)
                        # Otherwise, attempt a best-effort transpose to match (X,Y,Z)
                        return np.ascontiguousarray(np.transpose(w, tuple(reversed(range(w.ndim)))))
                    # If unexpected rank, raise
                    raise RuntimeError('Unexpected warped tensor shape from FireAnts')

            log(f'[register_nifti] Using FireAnts ({device}) for {tmode} registration.')
            return _FAAdapter(reg, fixed_batched, (mx, my, mz), device)
        except Exception as e:
            log(f'[register_nifti] fireants path failed ({e}), falling back to DIPY')

    # Fallback: DIPY implementation unchanged
    try:
        # Force Affine (or Rigid) only. Treat 'SyN' as 'Affine' to avoid expensive runs.
        if transform in ('Rigid', 'Affine', 'SyN'):
            tmode = 'Affine' if transform == 'SyN' else transform
            metric = MutualInformationMetric(nbins=32, sampling_proportion=None)
            # Aggressively reduced iteration counts for speed
            level_iters = [15, 8, 2]
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
        # If an unknown transform name is provided, raise an error (no fallback to SyN).
        raise ValueError(f"Unknown transform '{transform}' - expected Rigid/Affine/SyN")
    except Exception as e:
        # Print diagnostic and re-raise so caller sees the failure immediately.
        log(f'[register_nifti] {transform} failed ({e})')
        raise


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='ANTs NIfTI registration via gRPC')
    parser.add_argument('--grpc-port',   type=int, default=50051)
    parser.add_argument('--fixed',       required=True,  help='fixed volume UUID hex')
    parser.add_argument('--moving',      required=True,  help='moving volume UUID hex')
    parser.add_argument('--output',       required=True,
                        help='UUID hex of the pre-registered bank slot to write result into')
    parser.add_argument('--log',         required=True,
                        help='UUID hex of the pre-registered bank slot to write log lines into')
    parser.add_argument('--transform',   default='Rigid',
                        help='Transform type (Rigid/Affine/SyN/ResampleOnly; default Rigid). '
                             'ResampleOnly skips optimization entirely and just resamples '
                             'the moving volume onto the fixed grid.')
    parser.add_argument('--debug-port',  type=int, default=0,
                        help='If >0, wait for debugpy attach on this port')
    args = parser.parse_args()

    fixed_uuid  = hex_to_uuid(args.fixed)
    moving_uuid = hex_to_uuid(args.moving)
    log_uuid    = hex_to_uuid(args.log)

    global _log_client, _log_uuid
    with HastyClient(f'localhost:{args.grpc_port}') as client:
        _log_client = client
        _log_uuid = log_uuid

        if args.debug_port > 0:
            import debugpy
            debugpy.connect(('localhost', args.debug_port))
            debugpy.wait_for_client()
            log('[register_nifti] Debugger connected.')

        fixed_world_affine  = get_affine(client, fixed_uuid)
        moving_world_affine = get_affine(client, moving_uuid)
        fixed_spacing  = affine_spacing(fixed_world_affine)
        moving_spacing = affine_spacing(moving_world_affine)
        log(f'[register_nifti] fixed  spacing: {fixed_spacing}')
        log(f'[register_nifti] moving spacing: {moving_spacing}')

        log('[register_nifti] Fetching volumes…')
        t0 = time.perf_counter()
        fixed_t  = client.fetch_value(fixed_uuid).as_tensor()
        moving_t = client.fetch_value(moving_uuid).as_tensor()
        t_fetch = time.perf_counter() - t0
        log(f'[register_nifti] fixed  shape: {list(fixed_t.shape)}')
        log(f'[register_nifti] moving shape: {list(moving_t.shape)}')
        log(f'[register_nifti] Fetch time: {t_fetch:.3f}s')

        fixed_ref  = extract_3d_ref(fixed_t)
        moving_ref = extract_3d_ref(moving_t)

        # Use higher-resolution volume (smaller voxel size) as ANTs fixed so the
        # output is on the fine grid.  Warp the coarser volume into that space.
        fixed_voxel_vol  = float(np.prod(fixed_spacing))
        moving_voxel_vol = float(np.prod(moving_spacing))
        log(f'[register_nifti] fixed  voxel vol: {fixed_voxel_vol:.4g} mm³')
        log(f'[register_nifti] moving voxel vol: {moving_voxel_vol:.4g} mm³')

        # Normalize both to [0,1] so registration metrics handle different contrasts
        # (e.g. signed B0 field in Hz vs unsigned magnitude).
        if fixed_voxel_vol <= moving_voxel_vol:
            # fixed is higher-res → use fixed grid; warp moving into it
            log('[register_nifti] Fixed has finer voxels → using fixed as registration fixed, '
                  'warping moving into fixed grid.')
            fixed_arr, fixed_affine = tensor_to_numpy(normalize_for_registration(fixed_ref),  fixed_world_affine)
            moving_arr, moving_affine = tensor_to_numpy(normalize_for_registration(moving_ref), moving_world_affine)
            src_t           = moving_t
            src_affine      = moving_world_affine
        else:
            # moving is higher-res → use moving grid; warp fixed into it
            log('[register_nifti] Moving has finer voxels → using moving as registration fixed, '
                  'warping fixed into moving grid.')
            fixed_arr, fixed_affine = tensor_to_numpy(normalize_for_registration(moving_ref), moving_world_affine)
            moving_arr, moving_affine = tensor_to_numpy(normalize_for_registration(fixed_ref),  fixed_world_affine)
            src_t           = fixed_t
            src_affine      = fixed_world_affine

        log(f'[register_nifti] Running {args.transform} registration…')
        t0 = time.perf_counter()
        reg_map = run_registration(fixed_arr, fixed_affine, moving_arr, moving_affine, args.transform)
        t_reg = time.perf_counter() - t0
        log(f'[register_nifti] Registration done. Took {t_reg:.3f}s')

        # Apply fwdtransforms (smaller-space → larger-space) to every src sub-volume.
        spatial_shape = src_t.shape[-3:]
        leading_shape = src_t.shape[:-3]
        src_flat      = src_t.reshape(-1, *spatial_shape)

        log(f'[register_nifti] Warping {src_flat.shape[0]} sub-volume(s)…')
        registered_vols = []
        t_warp_start = time.perf_counter()
        for i in range(src_flat.shape[0]):
            it0 = time.perf_counter()
            sub_t = src_flat[i].float()
            sub_arr, sub_affine = tensor_to_numpy(sub_t, src_affine)
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
            log(f'[register_nifti] Warped sub-volume {i+1}/{src_flat.shape[0]} in {it:.3f}s')
        t_warp = time.perf_counter() - t_warp_start
        log(f'[register_nifti] Total warp time: {t_warp:.3f}s')

        if leading_shape:
            stacked    = torch.stack(registered_vols, dim=0)
            reg_tensor = stacked.reshape(*leading_shape, *stacked.shape[1:])
        else:
            reg_tensor = registered_vols[0]

        reg_f = reg_tensor.float()
        nonzero = (reg_f != 0).sum().item()
        log(f'[register_nifti] result shape: {list(reg_tensor.shape)}, '
              f'min={reg_f.min().item():.4g}, max={reg_f.max().item():.4g}, '
              f'nonzero={nonzero}/{reg_f.numel()} '
              f'({100.0*nonzero/max(reg_f.numel(),1):.1f}%)')
        if nonzero == 0:
            log('[register_nifti] WARNING: registered tensor is all-zero — '
                  'registration may have failed (contrast mismatch?)')

        output_uuid = hex_to_uuid(args.output)
        t0 = time.perf_counter()
        client.write_value(output_uuid, GenericValue.from_tensor(reg_tensor))
        t_write = time.perf_counter() - t0
        log(f'[register_nifti] Wrote result to output slot {args.output} (write time {t_write:.3f}s)')


if __name__ == '__main__':
    main()
