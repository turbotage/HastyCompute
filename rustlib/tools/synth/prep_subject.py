#!/usr/bin/env python3
"""Prepare one synthetic-phantom subject from an AVT CTA case.

Steps
  1. read the AVT CT (.nrrd) and expert aortic-vessel-tree mask, fix the HU offset (some cases store HU + 1024),
  2. run TotalSegmentator (task `total`) on the corrected CT for the anatomical context,
  3. crop to the thoracic region (heart + arch + descending aorta down to the diaphragm),
  4. export on a 1 mm isotropic grid: HU, TotalSegmentator labels, body mask, AVT vessel mask,
     and the vessel signed-distance function (SDF) on a 0.5 mm grid around the aorta,
  5. centerline skeleton of the vessel tree (kimimaro TEASAR) with inscribed radii,
  6. meta.json with grids (affine voxel -> physical, metres, DICOM LPS: +z = superior = B0 axis for a head-first patient).

Usage:  prep_subject.py <avt_case_dir> <out_dir> [--skip-totalseg]
"""
import json
import os
import subprocess
import sys

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from scipy import ndimage

VENV_BIN = os.path.dirname(sys.executable)
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))  # HastyCompute


def hu_offset(a):
    """AVT cases are stored either in HU or in HU + 1024. Air inside the scan FOV sits near -1000 HU."""
    v = a[::3, ::3, ::3].ravel()
    v = v[v > -1500]  # exclude out-of-FOV padding
    lo = np.percentile(v, 2)
    return 1024.0 if lo > -300 else 0.0


def resample(img, spacing, interp, default):
    sz = [int(round(s * sp / spacing)) for s, sp in zip(img.GetSize(), img.GetSpacing())]
    return sitk.Resample(img, sz, sitk.Transform(), interp, img.GetOrigin(), [spacing] * 3, img.GetDirection(), default, img.GetPixelID())


def to_lps_affine(img):
    """voxel (i,j,k) -> physical LPS in metres, for arrays indexed [i,j,k] (x fastest in ITK = first numpy axis after transpose)."""
    d = np.array(img.GetDirection()).reshape(3, 3)
    sp = np.array(img.GetSpacing())
    A = np.eye(4)
    A[:3, :3] = d * sp[None, :] * 1e-3
    A[:3, 3] = np.array(img.GetOrigin()) * 1e-3
    return A


def arr(img):
    """ITK image -> numpy [x, y, z]."""
    return np.transpose(sitk.GetArrayFromImage(img), (2, 1, 0))


def main():
    case, out = sys.argv[1], sys.argv[2]
    skip_ts = "--skip-totalseg" in sys.argv
    os.makedirs(out, exist_ok=True)
    name = os.path.basename(case.rstrip("/")).split(" ")[0]
    ct = sitk.ReadImage(os.path.join(case, f"{name}.nrrd"))
    seg = sitk.ReadImage(os.path.join(case, f"{name}.seg.nrrd"))
    a = sitk.GetArrayFromImage(ct).astype(np.float32)
    off = hu_offset(a)
    a = a - off
    a[a < -1024] = -1024
    hu = sitk.GetImageFromArray(a.astype(np.int16))
    hu.CopyInformation(ct)
    ct_path = os.path.join(out, "ct_hu.nii.gz")
    sitk.WriteImage(hu, ct_path)
    print(f"{name}: HU offset {off}, size {ct.GetSize()}, spacing {ct.GetSpacing()}")

    ts_path = os.path.join(out, "totalseg.nii")
    if not skip_ts or not os.path.exists(ts_path):
        # model weights inside the repository (data/hasty/totalseg), not in the home directory
        env = dict(os.environ, TOTALSEG_HOME_DIR=os.path.join(REPO, "data", "hasty", "totalseg"))
        subprocess.run([os.path.join(VENV_BIN, "TotalSegmentator"), "-i", ct_path, "-o", os.path.join(out, "totalseg"), "--ml"], check=True, env=env)
    ts = sitk.ReadImage(ts_path)
    ts.CopyInformation(ct)
    seg.CopyInformation(ct)

    # ---------------- crop to the thoracic region (index space of the native grid, numpy [x,y,z])
    tsa = arr(ts)
    av = arr(seg) > 0
    heart = tsa == 51
    zs_heart = np.where(heart.any(axis=(0, 1)))[0]
    lungs = np.isin(tsa, [10, 11, 12, 13, 14])
    zs_lung = np.where(lungs.any(axis=(0, 1)))[0]
    sp = np.array(ct.GetSpacing())
    up = ct.GetDirection()[8] > 0  # +k = superior
    # z from 30 mm below the heart to the lung apex (+20 mm); in-plane: whole body
    if up:
        z0 = max(0, zs_heart.min() - int(30 / sp[2]))
        z1 = min(av.shape[2] - 1, zs_lung.max() + int(20 / sp[2]))
    else:
        z0 = max(0, zs_lung.min() - int(20 / sp[2]))
        z1 = min(av.shape[2] - 1, zs_heart.max() + int(30 / sp[2]))
    roi = [0, 0, int(z0)], [int(ct.GetSize()[0]), int(ct.GetSize()[1]), int(z1 - z0 + 1)]
    crop = lambda im: sitk.RegionOfInterest(im, roi[1], roi[0])
    ct_c, ts_c, av_c = crop(hu), crop(ts), crop(seg)

    # ---------------- 1 mm tissue grid
    ct1 = resample(ct_c, 1.0, sitk.sitkLinear, -1024)
    ts1 = resample(ts_c, 1.0, sitk.sitkNearestNeighbor, 0)
    av1 = resample(sitk.Cast(av_c, sitk.sitkFloat32), 1.0, sitk.sitkLinear, 0)
    hu1 = arr(ct1).astype(np.float32)
    lab1 = arr(ts1).astype(np.uint8)
    ves1 = arr(av1) > 0.5
    # body mask: largest connected component above -500 HU, holes filled slice-wise (lungs, airways stay inside)
    body = hu1 > -500
    lab, n = ndimage.label(body)
    if n > 1:
        sizes = ndimage.sum(body, lab, range(1, n + 1))
        body = lab == (1 + int(np.argmax(sizes)))
    for k in range(body.shape[2]):
        body[:, :, k] = ndimage.binary_fill_holes(body[:, :, k])
    np.save(os.path.join(out, "hu_1mm.npy"), hu1.astype(np.float32))
    np.save(os.path.join(out, "labels_1mm.npy"), lab1)
    np.save(os.path.join(out, "body_1mm.npy"), body.astype(np.uint8))
    np.save(os.path.join(out, "vessel_1mm.npy"), ves1.astype(np.uint8))

    # ---------------- 0.5 mm vessel SDF in the aorta bounding box (+15 mm margin)
    av05 = resample(sitk.Cast(av_c, sitk.sitkFloat32), 0.5, sitk.sitkLinear, 0)
    v = arr(av05)
    v = ndimage.gaussian_filter(v, 1.0) > 0.5  # light smoothing of the staircase from 3 mm / 0.6 mm source slices
    idx = np.argwhere(v)
    lo = np.maximum(idx.min(0) - 30, 0)
    hi = np.minimum(idx.max(0) + 31, v.shape)
    vb = v[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    # keep the largest component (the aortic tree), remove specks
    lab, n = ndimage.label(vb)
    if n > 1:
        sizes = ndimage.sum(vb, lab, range(1, n + 1))
        vb = lab == (1 + int(np.argmax(sizes)))
    sdf = (ndimage.distance_transform_edt(~vb) - ndimage.distance_transform_edt(vb)) * 0.5e-3  # metres, < 0 inside
    np.save(os.path.join(out, "vessel_sdf_05mm.npy"), sdf.astype(np.float32))
    A05 = to_lps_affine(av05)
    A05_box = A05.copy()
    A05_box[:3, 3] = A05[:3, :3] @ lo + A05[:3, 3]

    # ---------------- centerline skeleton
    import kimimaro

    skels = kimimaro.skeletonize(
        vb.astype(np.uint32),
        teasar_params={"scale": 1.5, "const": 3, "pdrf_scale": 100000, "pdrf_exponent": 4, "soma_detection_threshold": 1e9, "max_paths": 50},
        anisotropy=(0.5, 0.5, 0.5),
        fix_branching=True,
        progress=False,
        parallel=1,
    )
    sk = skels[1]
    verts_vox = sk.vertices / 0.5  # nm-like units = mm with anisotropy -> voxel index in the box
    verts_phys = (A05_box[:3, :3] @ verts_vox.T).T + A05_box[:3, 3]
    radius = np.maximum(-ndimage.map_coordinates(sdf, verts_vox.T, order=1), 0.0)
    centerline = {"vertices": verts_phys.tolist(), "edges": sk.edges.tolist(), "radius": radius.tolist()}
    with open(os.path.join(out, "centerline.json"), "w") as f:
        json.dump(centerline, f)

    meta = {
        "case": name,
        "source": "AVT (Radl et al. 2022, CC BY 4.0) + TotalSegmentator v2 labels",
        "hu_offset_removed": off,
        "frame": "DICOM LPS, metres; +z superior = B0 axis (head-first supine)",
        "grid_1mm": {"shape": list(hu1.shape), "affine": to_lps_affine(ct1).tolist()},
        "grid_vessel_05mm": {"shape": list(sdf.shape), "affine": A05_box.tolist()},
        "totalseg_classes": "TotalSegmentator v2 'total' class map",
    }
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print("done:", out, "1mm grid", hu1.shape, "vessel box", sdf.shape, "centerline verts", len(radius))


if __name__ == "__main__":
    main()
