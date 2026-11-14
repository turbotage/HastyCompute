"""Reference (Python) preprocessing of the Pham et al. T1 PVS nnU-Net, written from their published description of
T1wPVSNormalization (wpham17/nnUNet-Perivascular-Spaces, CC BY-NC-SA): the Rust implementation in hasty-infer is
checked against this. Uses skimage + dipy only; nothing from that repository is executed.

usage: python pvs_t1_ref_norm.py in.nii.gz outdir [blockwise|classic]   (env PVS_CONTRAST=t2: T2wPVSNormalization, percentiles (2,98))
writes img.npy [z,y,x] (as SimpleITK would give it), bbox.json, norm_<method>.npy (on the nonzero crop)
"""
import json
import os
import sys
import time

import numpy as np
import SimpleITK as sitk
from dipy.denoise.nlmeans import nlmeans
from dipy.denoise.noise_estimate import estimate_sigma
from skimage import exposure


def t1w_pvs_norm(image, method="blockwise", lo_pct=1, hi_pct=99):
    image = image.astype(np.float32, copy=True)
    nzv = image[image != 0]
    image[image < nzv.min()] = nzv.min()
    image[image > nzv.max()] = nzv.max()
    p1, p2 = np.percentile(image, (lo_pct, hi_pct))
    mask = image > 0
    image = exposure.rescale_intensity(image, in_range=(p1, p2))
    sigma = estimate_sigma(image) / 2
    image = nlmeans(image, mask=mask, sigma=sigma, patch_radius=1, block_radius=2, rician=True, method=method)
    p1, p2 = np.percentile(image, (0, 98))
    image = exposure.rescale_intensity(image, in_range=(p1, p2))
    return exposure.equalize_adapthist(image)


if __name__ == "__main__":
    img = sitk.GetArrayFromImage(sitk.ReadImage(sys.argv[1])).astype(np.float32)
    out = sys.argv[2]
    methods = sys.argv[3:] or ["blockwise", "classic"]
    nz = np.argwhere(img != 0)
    lo, hi = nz.min(0), nz.max(0) + 1
    crop = img[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    np.save(f"{out}/img.npy", img)
    json.dump({"lo": lo.tolist(), "hi": hi.tolist()}, open(f"{out}/bbox.json", "w"))
    for m in methods:
        t = time.time()
        n = t1w_pvs_norm(crop, m, *((2, 98) if os.environ.get('PVS_CONTRAST') == 't2' else (1, 99)))
        print(m, n.shape, n.dtype, float(n.min()), float(n.max()), f"{time.time() - t:.1f}s")
        np.save(f"{out}/norm_{m}.npy", n.astype(np.float32))
