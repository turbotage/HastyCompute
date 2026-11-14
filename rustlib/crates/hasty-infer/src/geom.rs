//! Voxel grids, affines and resampling. Affine maps voxel index `(i, j, k)` (x, y, z; i fastest in memory) to RAS+ mm.

use rayon::prelude::*;

use crate::{Error, Result};

pub type Affine = [[f64; 4]; 4];

pub const IDENTITY: Affine = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]];

pub fn apply(a: &Affine, p: [f64; 3]) -> [f64; 3] {
    [0, 1, 2].map(|r| a[r][0] * p[0] + a[r][1] * p[1] + a[r][2] * p[2] + a[r][3])
}

/// inverse of an affine (last row `0 0 0 1`)
pub fn invert(a: &Affine) -> Result<Affine> {
    let m = [[a[0][0], a[0][1], a[0][2]], [a[1][0], a[1][1], a[1][2]], [a[2][0], a[2][1], a[2][2]]];
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if det.abs() < 1e-12 {
        return Err(Error::Geometry("singular affine".into()));
    }
    let c = |i: usize, j: usize| {
        let (r0, r1, c0, c1) = ([1, 0, 0][i], [2, 2, 1][i], [1, 0, 0][j], [2, 2, 1][j]);
        let s = if (i + j) % 2 == 0 { 1.0 } else { -1.0 };
        s * (m[r0][c0] * m[r1][c1] - m[r0][c1] * m[r1][c0])
    };
    // inverse = adj / det, adj[j][i] = cofactor(i, j)
    let mut inv = IDENTITY;
    for i in 0..3 {
        for j in 0..3 {
            inv[i][j] = c(j, i) / det;
        }
    }
    let t = [a[0][3], a[1][3], a[2][3]];
    for i in 0..3 {
        inv[i][3] = -(inv[i][0] * t[0] + inv[i][1] * t[1] + inv[i][2] * t[2]);
    }
    Ok(inv)
}

#[derive(Clone, Debug, PartialEq)]
pub struct Geometry {
    pub dims: [usize; 3],
    pub affine: Affine,
}

impl Geometry {
    /// axis-aligned RAS grid with the given voxel size, origin at voxel (0, 0, 0)
    pub fn from_spacing(dims: [usize; 3], spacing: [f64; 3]) -> Self {
        let mut affine = IDENTITY;
        for i in 0..3 {
            affine[i][i] = spacing[i];
        }
        Self { dims, affine }
    }

    pub fn numel(&self) -> usize {
        self.dims.iter().product()
    }

    /// voxel size along each voxel axis (column norms)
    pub fn spacing(&self) -> [f64; 3] {
        [0, 1, 2].map(|j| (self.affine[0][j].powi(2) + self.affine[1][j].powi(2) + self.affine[2][j].powi(2)).sqrt())
    }

    /// world position of the grid centre
    pub fn center(&self) -> [f64; 3] {
        apply(&self.affine, self.dims.map(|n| (n as f64 - 1.0) / 2.0))
    }

    /// 3-letter orientation of the voxel axes (`"RAS"`, `"LIA"`, ...): the nearest world direction each axis increases toward
    pub fn orientation(&self) -> String {
        (0..3)
            .map(|j| {
                let v = [self.affine[0][j], self.affine[1][j], self.affine[2][j]];
                let (k, s) = (0..3).max_by(|&a, &b| v[a].abs().partial_cmp(&v[b].abs()).unwrap()).map(|k| (k, v[k] >= 0.0)).unwrap();
                ["RL", "AP", "SI"][k].chars().nth(if s { 0 } else { 1 }).unwrap()
            })
            .collect()
    }
}

/// Dense scalar volume, x fastest.
#[derive(Clone, Debug)]
pub struct Volume {
    pub geom: Geometry,
    pub data: Vec<f32>,
}

impl Volume {
    pub fn new(geom: Geometry, data: Vec<f32>) -> Result<Self> {
        if data.len() != geom.numel() {
            return Err(Error::Geometry(format!("{} values for dims {:?}", data.len(), geom.dims)));
        }
        Ok(Self { geom, data })
    }

    pub fn zeros(geom: Geometry) -> Self {
        let n = geom.numel();
        Self { geom, data: vec![0.0; n] }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Interp {
    Nearest,
    Linear,
}

/// Resample `src` onto `dst` through world space; voxels outside `src` get `fill`. (No anti-aliasing: when
/// downsampling by more than ~2× filter first.)
pub fn resample(src: &Volume, dst: &Geometry, interp: Interp, fill: f32) -> Result<Volume> {
    let inv = invert(&src.geom.affine)?;
    // dst voxel -> src voxel: M = inv * dst.affine
    let mut m = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            m[i][j] = (0..4).map(|k| inv[i][k] * dst.affine[k][j]).sum();
        }
    }
    let [nx, ny, nz] = src.geom.dims;
    let [dx, dy, _] = dst.dims;
    let mut out = vec![fill; dst.numel()];
    let at = |i: isize, j: isize, k: isize| -> f32 {
        if i < 0 || j < 0 || k < 0 || i >= nx as isize || j >= ny as isize || k >= nz as isize {
            fill
        } else {
            src.data[(k as usize * ny + j as usize) * nx + i as usize]
        }
    };
    out.par_chunks_mut(dx * dy).enumerate().for_each(|(k, plane)| {
        for j in 0..dy {
            for i in 0..dx {
                let p = [i as f64, j as f64, k as f64];
                let s = [0, 1, 2].map(|r| m[r][0] * p[0] + m[r][1] * p[1] + m[r][2] * p[2] + m[r][3]);
                plane[j * dx + i] = match interp {
                    Interp::Nearest => at(s[0].round() as isize, s[1].round() as isize, s[2].round() as isize),
                    Interp::Linear => {
                        let f = s.map(f64::floor);
                        let (i0, j0, k0) = (f[0] as isize, f[1] as isize, f[2] as isize);
                        let t = [s[0] - f[0], s[1] - f[1], s[2] - f[2]];
                        let mut acc = 0.0f64;
                        for (dk, wk) in [(0, 1.0 - t[2]), (1, t[2])] {
                            for (dj, wj) in [(0, 1.0 - t[1]), (1, t[1])] {
                                for (di, wi) in [(0, 1.0 - t[0]), (1, t[0])] {
                                    let w = wi * wj * wk;
                                    if w != 0.0 {
                                        acc += w * at(i0 + di, j0 + dj, k0 + dk) as f64;
                                    }
                                }
                            }
                        }
                        acc as f32
                    }
                };
            }
        }
    });
    Volume::new(dst.clone(), out)
}

/// world direction of an orientation letter (RAS+): the direction the voxel axis increases toward
pub fn letter_dir(c: char) -> Result<[f64; 3]> {
    Ok(match c.to_ascii_uppercase() {
        'R' => [1.0, 0.0, 0.0],
        'L' => [-1.0, 0.0, 0.0],
        'A' => [0.0, 1.0, 0.0],
        'P' => [0.0, -1.0, 0.0],
        'S' => [0.0, 0.0, 1.0],
        'I' => [0.0, 0.0, -1.0],
        _ => return Err(Error::Geometry(format!("bad orientation letter {c:?}"))),
    })
}

/// Grid of `dims` with isotropic `spacing` and voxel axes oriented as `orientation` (e.g. `"LIA"` = FreeSurfer
/// conformed), centred on the centre of `src`.
pub fn conform_geometry(src: &Geometry, spacing: f64, dims: [usize; 3], orientation: &str) -> Result<Geometry> {
    let o: Vec<char> = orientation.chars().collect();
    if o.len() != 3 {
        return Err(Error::Geometry(format!("orientation {orientation:?} must have 3 letters")));
    }
    let mut a = IDENTITY;
    for j in 0..3 {
        let d = letter_dir(o[j])?;
        for i in 0..3 {
            a[i][j] = d[i] * spacing;
        }
    }
    let c = src.center();
    for i in 0..3 {
        let half: f64 = (0..3).map(|j| a[i][j] * (dims[j] as f64 - 1.0) / 2.0).sum();
        a[i][3] = c[i] - half;
    }
    Ok(Geometry { dims, affine: a })
}

/// Smallest grid with voxel size `spacing` (per voxel axis) and axes oriented as `orientation` that contains every
/// voxel centre of `src`.
pub fn covering_geometry(src: &Geometry, spacing: [f64; 3], orientation: &str) -> Result<Geometry> {
    let o: Vec<char> = orientation.chars().collect();
    if o.len() != 3 {
        return Err(Error::Geometry(format!("orientation {orientation:?} must have 3 letters")));
    }
    let mut a = IDENTITY;
    for j in 0..3 {
        let d = letter_dir(o[j])?;
        for i in 0..3 {
            a[i][j] = d[i] * spacing[j];
        }
    }
    let inv = invert(&a)?;
    let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
    for c in 0..8 {
        let v = [0, 1, 2].map(|j| if (c >> j) & 1 == 1 { src.dims[j] as f64 - 1.0 } else { 0.0 });
        let m = apply(&inv, apply(&src.affine, v));
        for j in 0..3 {
            lo[j] = lo[j].min(m[j]);
            hi[j] = hi[j].max(m[j]);
        }
    }
    let dims = [0, 1, 2].map(|j| ((hi[j] - lo[j]) - 1e-6).ceil().max(0.0) as usize + 1);
    // voxel 0 sits at model-space coordinate lo; world = a * (lo + index)
    let t = apply(&a, lo);
    for i in 0..3 {
        a[i][3] = t[i];
    }
    Ok(Geometry { dims, affine: a })
}

/// Geometry of the sub-box `[lo, hi)` (voxel indices) of `src`.
pub fn crop_geometry(src: &Geometry, lo: [usize; 3], hi: [usize; 3]) -> Geometry {
    let mut affine = src.affine;
    let t = apply(&src.affine, [lo[0] as f64, lo[1] as f64, lo[2] as f64]);
    for i in 0..3 {
        affine[i][3] = t[i];
    }
    Geometry { dims: [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]], affine }
}

/// Same axes and extent as `src` with voxel size `spacing` (nnU-Net resampling: `round(n · s_old / s_new)` voxels,
/// outer voxel faces aligned, i.e. the origin shifts by half the change of voxel size).
pub fn rescaled_geometry(src: &Geometry, spacing: [f64; 3]) -> Geometry {
    let old = src.spacing();
    let mut a = src.affine;
    let mut dims = [0usize; 3];
    for j in 0..3 {
        dims[j] = ((src.dims[j] as f64 * old[j] / spacing[j]).round() as usize).max(1);
        let f = spacing[j] / old[j];
        for i in 0..3 {
            let col = src.affine[i][j];
            a[i][j] = col * f;
            a[i][3] += 0.5 * (col * f - col);
        }
    }
    Geometry { dims, affine: a }
}
