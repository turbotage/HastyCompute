#!/usr/bin/env python3
"""
Register two NIfTI volumes using ANTs SyN.

Usage:
    register_nifti.py --grpc-port=PORT --fixed=UUID_HEX --moving=UUID_HEX
                      [--transform=SyN] [--debug-port=PORT]

Prints to stdout (one line each):
    <registered_uuid_hex>

Registration direction is chosen automatically: the physically larger volume
is always used as the ANTs fixed reference (better convergence). The OUTPUT
is whichever volume is smaller, warped into the larger volume's grid.

The script reads NIfTI header metadata (pixdim/sform) written by
hasty::python::push_nifti_image() so voxel spacing is correct.
Stderr is used for progress/info messages.
"""

import argparse
import json
import sys

import numpy as np
import torch
import ants

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

def normalize_for_ants(t: torch.Tensor) -> torch.Tensor:
    """Normalize tensor to [0,1] range so ANTs handles different contrast types."""
    t = t.float()
    t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    mn, mx = t.min().item(), t.max().item()
    if mx > mn:
        return (t - mn) / (mx - mn)
    return torch.zeros_like(t)

def tensor_to_ants(t: torch.Tensor, spacing: tuple) -> ants.ANTsImage:
    """C-order [nz, ny, nx] → ANTs [nx, ny, nz]. t must be 3-D."""
    assert t.ndim == 3, f"tensor_to_ants expects 3-D tensor, got shape {list(t.shape)}"
    arr = np.ascontiguousarray(t.float().numpy().T)  # [nx, ny, nz]
    return ants.from_numpy(arr, spacing=spacing)

def ants_to_tensor(img: ants.ANTsImage) -> torch.Tensor:
    """ANTs [nx, ny, nz] → C-order [nz, ny, nx] tensor."""
    arr = np.ascontiguousarray(img.numpy().T)  # [nz, ny, nx]
    return torch.from_numpy(arr.copy())

def extract_3d_ref(t: torch.Tensor) -> torch.Tensor:
    """Mean over leading dims until 3-D (last 3 dims are Z,Y,X)."""
    while t.ndim > 3:
        t = t.float().mean(dim=0)
    return t

_FALLBACK = {'SyN': 'Affine', 'Affine': 'Rigid'}

def run_registration(fixed_ants: ants.ANTsImage, moving_ants: ants.ANTsImage,
                     transform: str) -> dict:
    """Try registration; fall back one level if it fails."""
    try:
        return ants.registration(fixed=fixed_ants, moving=moving_ants,
                                 type_of_transform=transform, verbose=False)
    except RuntimeError as e:
        fallback = _FALLBACK.get(transform)
        if fallback is None:
            raise
        print(f'[register_nifti] {transform} failed ({e}); retrying with {fallback}.',
              file=sys.stderr, flush=True)
        return ants.registration(fixed=fixed_ants, moving=moving_ants,
                                 type_of_transform=fallback, verbose=False)


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='ANTs NIfTI registration via gRPC')
    parser.add_argument('--grpc-port',   type=int, default=50051)
    parser.add_argument('--fixed',       required=True,  help='fixed volume UUID hex')
    parser.add_argument('--moving',      required=True,  help='moving volume UUID hex')
    parser.add_argument('--transform',   default='Rigid',
                        help='ANTs transform type (Rigid/Affine/SyN; default Rigid for '
                             'same-session cross-modality registration)')
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
        fixed_t  = client.fetch_value(fixed_uuid).as_tensor()
        moving_t = client.fetch_value(moving_uuid).as_tensor()
        print(f'[register_nifti] fixed  shape: {list(fixed_t.shape)}', file=sys.stderr)
        print(f'[register_nifti] moving shape: {list(moving_t.shape)}', file=sys.stderr)

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

        # Normalize both to [0,1] so ANTs MI metric handles different contrasts
        # (e.g. signed B0 field in Hz vs unsigned magnitude).
        if fixed_voxel_vol <= moving_voxel_vol:
            # fixed is higher-res → use fixed grid; warp moving into it
            print('[register_nifti] Fixed has finer voxels → using fixed as ANTs fixed, '
                  'warping moving into fixed grid.', file=sys.stderr, flush=True)
            ants_ref_fixed  = tensor_to_ants(normalize_for_ants(fixed_ref),  fixed_spacing)
            ants_ref_moving = tensor_to_ants(normalize_for_ants(moving_ref), moving_spacing)
            src_t           = moving_t
            src_spacing     = moving_spacing
        else:
            # moving is higher-res → use moving grid; warp fixed into it
            print('[register_nifti] Moving has finer voxels → using moving as ANTs fixed, '
                  'warping fixed into moving grid.', file=sys.stderr, flush=True)
            ants_ref_fixed  = tensor_to_ants(normalize_for_ants(moving_ref), moving_spacing)
            ants_ref_moving = tensor_to_ants(normalize_for_ants(fixed_ref),  fixed_spacing)
            src_t           = fixed_t
            src_spacing     = fixed_spacing

        # Resample smaller ref into larger grid before registration.
        print('[register_nifti] Resampling smaller reference to larger grid…',
              file=sys.stderr, flush=True)
        ants_ref_moving = ants.resample_image_to_target(ants_ref_moving, ants_ref_fixed)

        print(f'[register_nifti] Running ANTs {args.transform} registration…',
              file=sys.stderr, flush=True)
        reg_result = run_registration(ants_ref_fixed, ants_ref_moving, args.transform)
        print('[register_nifti] Registration done.', file=sys.stderr, flush=True)

        # Apply fwdtransforms (smaller-space → larger-space) to every src sub-volume.
        spatial_shape = src_t.shape[-3:]
        leading_shape = src_t.shape[:-3]
        src_flat      = src_t.reshape(-1, *spatial_shape)

        print(f'[register_nifti] Warping {src_flat.shape[0]} sub-volume(s)…',
              file=sys.stderr, flush=True)
        registered_vols = []
        for i in range(src_flat.shape[0]):
            sub_ants = tensor_to_ants(src_flat[i].float(), src_spacing)
            warped   = ants.apply_transforms(
                fixed=ants_ref_fixed,
                moving=sub_ants,
                transformlist=reg_result['fwdtransforms'],
            )
            vol = ants_to_tensor(warped)
            # ANTs may fill background with NaN; replace with 0.
            vol = torch.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)
            registered_vols.append(vol)

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

        out_uuid = client.push_value(GenericValue.from_tensor(reg_tensor))
        print(f'[register_nifti] Pushed result UUID: {uuid_to_hex(out_uuid)}',
              file=sys.stderr, flush=True)
        print(uuid_to_hex(out_uuid), flush=True)


if __name__ == '__main__':
    main()
