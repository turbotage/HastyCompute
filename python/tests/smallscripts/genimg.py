#!/usr/bin/env python3
"""genimg.py – Generate a HastyCompute GV-format HDF5 file of standard test images.

On-disk layout (must match C++ hdf5::read_generic_value / write_generic_value):

    /gv_root                   Group   gv_type = "dict"
      /<image_name>            Dataset gv_type = "tensor"
                                       gv_dtype        = e.g. "f32"
                                       gv_device_type  = int32(0)   # CPU
                                       gv_device_index = int32(-1)  # no index
                                       gv_shape        = int64[...]
                               data: raw bytes as uint8, shape=(numel,)

All float images are normalised to [0, 1] and stored as float32.

Dependencies
------------
    pip install h5py numpy scikit-image nilearn nibabel
    pip install brainweb   # optional

Usage
-----
    python genimg.py                           # -> images.h5
    python genimg.py --output /path/out.h5
    python genimg.py --no-mri                  # skip heavy downloads
    python genimg.py --brainweb                # also include BrainWeb MRI
"""

import argparse
import pathlib
import sys
from typing import Dict

import h5py
import numpy as np


# ── GV tensor format constants (must match C++ tensor_background.cppm) ────────

_NP_TO_GV_DTYPE: Dict[type, str] = {
    np.float32:    "f32",
    np.float64:    "f64",
    np.int32:      "i32",
    np.int64:      "i64",
    np.uint8:      "u8",
    np.int8:       "i8",
    np.int16:      "i16",
    np.bool_:      "b8",
    np.complex64:  "c32",
    np.complex128: "c64",
}


def _write_tensor(grp: h5py.Group, name: str, arr: np.ndarray) -> None:
    """Write a numpy array as a native typed HDF5 dataset readable from C++."""
    arr = np.ascontiguousarray(arr)
    gv_dtype = _NP_TO_GV_DTYPE.get(arr.dtype.type)
    if gv_dtype is None:
        arr = arr.astype(np.float32)
        gv_dtype = "f32"
    # Natural h5py storage: proper N-D shape + native dtype (float32, int32, etc.)
    # exactly as numpy would write it — readable by h5py, HDFView, and our C++ reader.
    ds = grp.create_dataset(name, data=arr)
    ds.attrs["gv_type"]  = "tensor"
    ds.attrs["gv_dtype"] = gv_dtype
    # Note: NO gv_shape attr here (shape lives in the dataset itself),
    # and NO byte-flattening — this is exactly the format the C++ reader expects.


def write_images_hdf5(output: pathlib.Path, images: Dict[str, np.ndarray]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting {len(images)} images → {output}")
    with h5py.File(output, "w") as f:
        root = f.create_group("gv_root")
        root.attrs["gv_type"] = "dict"
        for name, arr in images.items():
            _write_tensor(root, name, arr)
    print("\nContents:")
    for name, arr in sorted(images.items()):
        print(f"  {name:<52}  shape={str(arr.shape):<22}  dtype={arr.dtype}")


def _norm(arr: np.ndarray) -> np.ndarray:
    """Cast to float32 and normalise to [0, 1]."""
    a = arr.astype(np.float32)
    mx = a.max()
    if mx > 0:
        a /= mx
    return a


# ── Standard 2-D / 3-D test images (scikit-image) ─────────────────────────────

def collect_skimage_images() -> Dict[str, np.ndarray]:
    import skimage.data as skd

    out: Dict[str, np.ndarray] = {}

    # Shepp-Logan phantom – canonical CT/MRI reconstruction benchmark
    out["shepp_logan_400x400"] = skd.shepp_logan_phantom().astype(np.float32)

    # Cameraman – standard deblurring / compressed-sensing test image
    out["cameraman_512x512"] = _norm(skd.camera())

    # Coins – nice for segmentation / level-set methods
    out["coins_303x384"] = _norm(skd.coins())

    # Astronaut – full RGB colour image (512, 512, 3)
    astro = skd.astronaut().astype(np.float32) / 255.0   # (512,512,3)
    out["astronaut_rgb_512x512x3"] = astro

    # Astronaut (RGB → luminance) – colour-to-grey inverse-problem benchmark
    lum = (0.2126 * astro[..., 0] +
           0.7152 * astro[..., 1] +
           0.0722 * astro[..., 2]).astype(np.float32)
    out["astronaut_luma_512x512"] = lum

    # Chelsea – cat photo, standard colour benchmark (300, 451, 3)
    out["chelsea_rgb_300x451x3"] = skd.chelsea().astype(np.float32) / 255.0

    # Coffee – vivid colours, useful for colour-space tests (400, 600, 3)
    try:
        out["coffee_rgb_400x600x3"] = skd.coffee().astype(np.float32) / 255.0
    except AttributeError:
        pass

    # Rocket – launch photo, wide dynamic range (427, 640, 3)
    try:
        out["rocket_rgb_427x640x3"] = skd.rocket().astype(np.float32) / 255.0
    except AttributeError:
        pass

    # Checkerboard – sharp edges, useful for PSF estimation tests
    out["checkerboard_200x200"] = _norm(skd.checkerboard())

    # Retina – circular FOV, high dynamic range (RGB + grayscale versions)
    try:
        retina = skd.retina()
        out["retina_rgb_1411x1411x3"] = _norm(retina)
        out["retina_1411x1411"] = _norm(retina[..., 0])   # R channel (grayscale)
    except AttributeError:
        pass

    # 3-D brain volume (uint16, axes: z, y, x)
    try:
        out["brain_3d_skimage"] = _norm(skd.brain())
    except Exception:
        pass

    return out


# ── Shepp-Logan 3-D phantom ────────────────────────────────────────────────────

def collect_phantom_3d() -> Dict[str, np.ndarray]:
    from skimage.data import shepp_logan_phantom

    sl = shepp_logan_phantom().astype(np.float32)   # (400, 400)

    # Stack 64 axial slices with a cosine-envelope slice profile
    nz = 64
    zfac = (np.cos(np.linspace(0, np.pi, nz)) * 0.3 + 0.7).astype(np.float32)
    vol = sl[np.newaxis] * zfac[:, np.newaxis, np.newaxis]   # (64, 400, 400)

    return {"shepp_logan_3d_64x400x400": vol}


# ── MRI images via nilearn (MNI152 templates) ──────────────────────────────────

def collect_nilearn_mri() -> Dict[str, np.ndarray]:
    try:
        from nilearn import datasets as nds
    except ImportError:
        print("  nilearn not installed – skipping nilearn MRI.\n"
              "  Install with: pip install nilearn nibabel")
        return {}

    out: Dict[str, np.ndarray] = {}

    # MNI152 T1w anatomical template – the standard brain atlas
    for res_mm in (2, 1):
        try:
            img = nds.load_mni152_template(resolution=res_mm)
            vol = img.get_fdata().astype(np.float32)
            vol /= vol.max()
            key = f"mni152_t1w_{res_mm}mm"
            out[key] = vol
            print(f"  {key}: {vol.shape}")
            break   # one resolution is enough; start with 2 mm (smaller)
        except Exception as e:
            print(f"  MNI152 at {res_mm} mm failed: {e}")

    # MNI152 brain mask (binary, useful as a k-space sampling mask generator)
    try:
        mask_img = nds.load_mni152_brain_mask()
        out["mni152_brain_mask"] = mask_img.get_fdata().astype(np.float32)
        print(f"  mni152_brain_mask: {out['mni152_brain_mask'].shape}")
    except Exception as e:
        print(f"  MNI152 brain mask failed: {e}")

    # OASIS cross-sectional dataset – first subject, T1w
    try:
        oasis = nds.fetch_oasis_vbm(n_subjects=1)
        img = oasis.gray_matter_maps[0]
        import nibabel as nib
        vol = nib.load(img).get_fdata().astype(np.float32)
        vol = _norm(vol)
        out["oasis_sub01_gm"] = vol
        print(f"  oasis_sub01_gm: {vol.shape}")
    except Exception as e:
        print(f"  OASIS failed (optional): {e}")

    return out


# ── MRI images via BrainWeb ────────────────────────────────────────────────────

def collect_brainweb_mri() -> Dict[str, np.ndarray]:
    try:
        import brainweb
    except ImportError:
        print("  brainweb not installed – skipping BrainWeb.\n"
              "  Install with: pip install brainweb")
        return {}

    out: Dict[str, np.ndarray] = {}
    print("  Downloading BrainWeb subject 04 (cached after first run) …")
    try:
        fname = brainweb.get_file("subject04.bin.gz")
        raw   = brainweb.load_file(fname)

        # The brainweb package generates simulated T1/T2/PD from tissue maps.
        # Each call returns a dict; keys vary by version – try common ones.
        for modality in ("T1", "T2", "PD"):
            try:
                sim = brainweb.toPetMmr(raw, petNoise=0)
                vol = sim.get(modality, sim.get(modality.lower()))
                if vol is None:
                    continue
                vol = _norm(np.asarray(vol))
                key = f"brainweb_sub04_{modality.lower()}"
                out[key] = vol
                print(f"  {key}: {vol.shape}")
            except Exception as e:
                print(f"  BrainWeb {modality} failed: {e}")
    except Exception as e:
        print(f"  BrainWeb overall failed: {e}")

    return out


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate HastyCompute GV HDF5 image file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output",    default="images.h5",
                        help="Output HDF5 file path")
    parser.add_argument("--no-mri",    action="store_true",
                        help="Skip all MRI downloads (fast, offline-safe)")
    parser.add_argument("--brainweb",  action="store_true",
                        help="Also download BrainWeb MRI (requires brainweb pkg)")
    args = parser.parse_args()

    images: Dict[str, np.ndarray] = {}

    print("── Standard test images (scikit-image) ─────────────────────────────")
    try:
        sk = collect_skimage_images()
        images.update(sk)
        print(f"  collected {len(sk)} images")
    except Exception as e:
        print(f"  ERROR: {e}", file=sys.stderr)

    print("\n── 3-D Shepp-Logan phantom ─────────────────────────────────────────")
    ph = collect_phantom_3d()
    images.update(ph)
    print(f"  collected {len(ph)} phantom(s)")

    if not args.no_mri:
        print("\n── MRI via nilearn (MNI152 + OASIS) ────────────────────────────────")
        images.update(collect_nilearn_mri())

        if args.brainweb:
            print("\n── MRI via BrainWeb ─────────────────────────────────────────────────")
            images.update(collect_brainweb_mri())

    write_images_hdf5(pathlib.Path(args.output), images)


if __name__ == "__main__":
    main()
