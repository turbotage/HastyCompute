"""NIfTI -> img.npy [z,y,x] float32 + geom.json {spacing_mm, affine (4x4 voxel->RAS mm, x fastest)} for tests/dev.
usage: python -I nii_to_npy.py in.nii.gz outdir   (needs SimpleITK; use rustlib/target/pvs/venv)"""
import json
import sys

import numpy as np
import SimpleITK as sitk

im = sitk.ReadImage(sys.argv[1])
a = sitk.GetArrayFromImage(im).astype(np.float32)
sp = np.array(im.GetSpacing())
d = np.array(im.GetDirection()).reshape(3, 3)
lps = np.eye(4)
lps[:3, :3] = d * sp
lps[:3, 3] = im.GetOrigin()
aff = np.diag([-1, -1, 1, 1]) @ lps
np.save(sys.argv[2] + "/img.npy", a)
json.dump({"spacing_mm": sp.tolist(), "affine": aff.tolist()}, open(sys.argv[2] + "/geom.json", "w"))
print(a.shape, sp, "\n", aff)
