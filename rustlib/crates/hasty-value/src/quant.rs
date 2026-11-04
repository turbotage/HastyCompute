//! "comprep": compressed u16 representation for display transfer (port of the C++ `tensor_comprep`, decoded by the
//! orthoslicer's WGSL `inv_shape`). The dynamic range is stored outside the elements (a JSON config), and a piecewise
//! transfer gives the window of interest most of the 16 bits:
//!
//! ```text
//!   x ∈ [clampa, a]  →  y ∈ [0,  t0]   shaped: linear | gamma (u^(1/γ)) | log (log1p(c·u)/log1p(c))
//!   x ∈ [a, b]       →  y ∈ [t0, t1]   linear
//!   x ∈ [b, clampb]  →  y ∈ [t1, 1]    shaped
//!   t0 = (1 − focus)/2, t1 = 1 − t0;  q = round(65535 · y)
//! ```
//!
//! focus = 1 spends everything on [a, b] (values outside saturate); focus < 1 keeps the tails at reduced precision.
//! Complex input is reduced first (`mode`: abs | real | imag | angle).

use serde_json::{json, Value as Json};

use crate::{DType, Error, Result, Tensor};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Shape {
    Linear,
    Gamma(f64),
    Log(f64),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Comprep {
    pub a: f64,
    pub b: f64,
    pub clampa: f64,
    pub clampb: f64,
    pub focus: f64,
    pub left: Shape,
    pub right: Shape,
    /// complex reduction
    pub mode: String,
}

fn shape_of(j: &Json, side: &str) -> Shape {
    match j[format!("{side}_mode")].as_str().unwrap_or("linear") {
        "gamma" => Shape::Gamma(j[format!("{side}_gamma")].as_f64().unwrap_or(1.0)),
        "log" => Shape::Log(j[format!("{side}_logc")].as_f64().unwrap_or(1.0)),
        _ => Shape::Linear,
    }
}

fn shape_json(s: Shape, side: &str, m: &mut serde_json::Map<String, Json>) {
    let (mode, g, c) = match s {
        Shape::Linear => ("linear", 1.0, 1.0),
        Shape::Gamma(g) => ("gamma", g, 1.0),
        Shape::Log(c) => ("log", 1.0, c),
    };
    m.insert(format!("{side}_mode"), json!(mode));
    m.insert(format!("{side}_gamma"), json!(g));
    m.insert(format!("{side}_logc"), json!(c));
}

fn fwd_shape(u: f64, s: Shape) -> f64 {
    match s {
        Shape::Linear => u,
        Shape::Gamma(g) => u.powf(1.0 / g),
        Shape::Log(c) => fwd_log(u, c),
    }
}

/// exactly the renderer's `inv_shape` (including its guards)
fn inv_shape(v: f64, s: Shape) -> f64 {
    match s {
        Shape::Linear => v,
        Shape::Gamma(g) => v.max(0.0).powf(g),
        Shape::Log(c) => {
            let c = c.max(1e-6);
            (v * c.ln_1p()).exp_m1() / c
        }
    }
}

fn fwd_log(u: f64, c: f64) -> f64 {
    let c = c.max(1e-6);
    (u * c).ln_1p() / c.ln_1p()
}

impl Comprep {
    /// clampa ≤ a < b ≤ clampb, 0 < focus ≤ 1
    pub fn normalized(mut self) -> Result<Self> {
        if !(self.a.is_finite() && self.b.is_finite()) || self.b <= self.a {
            return Err(Error::Type(format!("comprep needs a < b (a {}, b {})", self.a, self.b)));
        }
        self.clampa = self.clampa.min(self.a);
        self.clampb = self.clampb.max(self.b);
        if !(self.focus > 0.0 && self.focus <= 1.0) {
            return Err(Error::Type(format!("comprep focus {} not in (0, 1]", self.focus)));
        }
        Ok(self)
    }

    /// config as given (`a`, `b` required)
    pub fn from_json(j: &Json) -> Result<Self> {
        let (Some(a), Some(b)) = (j["a"].as_f64(), j["b"].as_f64()) else { return Err(Error::Type("comprep config needs a and b".into())) };
        Self {
            a,
            b,
            clampa: j["clampa"].as_f64().unwrap_or(a),
            clampb: j["clampb"].as_f64().unwrap_or(b),
            focus: j["focus"].as_f64().unwrap_or(1.0),
            left: shape_of(j, "left"),
            right: shape_of(j, "right"),
            mode: j["mode"].as_str().unwrap_or("abs").to_string(),
        }
        .normalized()
    }

    /// every key the renderer reads
    pub fn to_json(&self) -> Json {
        let mut m = serde_json::Map::new();
        for (k, v) in [("a", self.a), ("b", self.b), ("clampa", self.clampa), ("clampb", self.clampb), ("focus", self.focus)] {
            m.insert(k.into(), json!(v));
        }
        shape_json(self.left, "left", &mut m);
        shape_json(self.right, "right", &mut m);
        m.insert("mode".into(), json!(self.mode));
        Json::Object(m)
    }

    fn t(&self) -> (f64, f64) {
        let t0 = (1.0 - self.focus) * 0.5;
        (t0, 1.0 - t0)
    }

    /// x → y ∈ [0, 1]
    pub fn forward(&self, x: f64) -> f64 {
        let (t0, t1) = self.t();
        let x = if x.is_nan() { 0.0 } else { x.clamp(self.clampa, self.clampb) };
        if x < self.a {
            let d = self.a - self.clampa;
            t0 * if d > 0.0 { fwd_shape((x - self.clampa) / d, self.left) } else { 0.0 }
        } else if x > self.b {
            let d = self.clampb - self.b;
            t1 + (1.0 - t1) * if d > 0.0 { fwd_shape((x - self.b) / d, self.right) } else { 1.0 }
        } else {
            let span = self.b - self.a;
            t0 + (t1 - t0) * if span > 0.0 { (x - self.a) / span } else { 0.0 }
        }
    }

    /// y ∈ [0, 1] → x (the orthoslicer shader's inverse)
    pub fn inverse(&self, y: f64) -> f64 {
        let (t0, t1) = self.t();
        if y < t0 && t0 > 0.0 {
            inv_shape(y / t0, self.left) * (self.a - self.clampa) + self.clampa
        } else if y > t1 && t1 < 1.0 {
            inv_shape((y - t1) / (1.0 - t1), self.right) * (self.clampb - self.b) + self.b
        } else {
            let span = t1 - t0;
            (if span > 1e-12 { (y - t0) / span } else { 0.0 }) * (self.b - self.a) + self.a
        }
    }
}

fn reduce(t: &Tensor, mode: &str, i: usize) -> f64 {
    let (re, im) = t.get_f64(i);
    if !t.dtype().is_complex() {
        return if mode == "abs" { re.abs() } else { re };
    }
    match mode {
        "real" => re,
        "imag" => im,
        "angle" => im.atan2(re),
        _ => re.hypot(im),
    }
}

/// Default config for `t` (C++ `compress_ui16_default`): integers of ≤ 16 bits map directly; otherwise a, b = the
/// `p_lo` / `p_hi` percentiles (default 1 / 99, from ≤ 1 M strided samples), clamp = [min, max], NaN → 0.
/// `overrides` may set focus, left/right modes, mode, p_lo, p_hi, or any bound explicitly.
pub fn auto_config(t: &Tensor, overrides: &Json) -> Result<Comprep> {
    let real_mode = if t.dtype().is_complex() { "abs" } else { "real" };
    let mode = overrides["mode"].as_str().unwrap_or(real_mode).to_string();
    let direct = match t.dtype() {
        DType::Bool => Some((0.0, 1.0)),
        DType::U8 => Some((0.0, 255.0)),
        DType::I8 => Some((-128.0, 127.0)),
        DType::I16 => Some((-32768.0, 32767.0)),
        DType::U16 => Some((0.0, 65535.0)),
        _ => None,
    };
    let mut c = if let Some((lo, hi)) = direct {
        Comprep { a: lo, b: hi, clampa: lo, clampb: hi, focus: 1.0, left: Shape::Linear, right: Shape::Linear, mode }
    } else {
        let n = t.numel();
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for i in 0..n {
            let v = reduce(t, &mode, i);
            let v = if v.is_nan() { 0.0 } else { v };
            lo = lo.min(v);
            hi = hi.max(v);
        }
        if n == 0 {
            (lo, hi) = (0.0, 1.0);
        }
        let step = (n / 1_000_000).max(1);
        let mut s: Vec<f64> = (0..n).step_by(step).map(|i| reduce(t, &mode, i)).map(|v| if v.is_nan() { 0.0 } else { v }).collect();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q = |p: f64| if s.is_empty() { 0.0 } else { s[(((p / 100.0) * (s.len() - 1) as f64).round() as usize).min(s.len() - 1)] };
        let (mut a, mut b) = (q(overrides["p_lo"].as_f64().unwrap_or(1.0)), q(overrides["p_hi"].as_f64().unwrap_or(99.0)));
        if hi <= lo {
            hi = lo + 1.0;
            (a, b) = (lo, hi);
        } else if a >= b {
            (a, b) = (lo, hi);
        }
        Comprep { a, b, clampa: lo, clampb: hi, focus: 1.0, left: Shape::Linear, right: Shape::Linear, mode }
    };
    for (k, f) in [("a", &mut c.a), ("b", &mut c.b), ("clampa", &mut c.clampa), ("clampb", &mut c.clampb), ("focus", &mut c.focus)] {
        if let Some(v) = overrides[k].as_f64() {
            *f = v;
        }
    }
    if overrides.get("left_mode").is_some() {
        c.left = shape_of(overrides, "left");
    }
    if overrides.get("right_mode").is_some() {
        c.right = shape_of(overrides, "right");
    }
    c.normalized()
}

/// Encode to u16 with `c`.
pub fn encode_u16(t: &Tensor, c: &Comprep) -> Result<Tensor> {
    let q: Vec<u16> = (0..t.numel()).map(|i| (c.forward(reduce(t, &c.mode, i)).clamp(0.0, 1.0) * 65535.0).round() as u16).collect();
    Tensor::from_slice(t.shape(), &q)
}

/// Decode u16 back to float32 values with `c`.
pub fn decode_u16(q: &Tensor, c: &Comprep) -> Result<Tensor> {
    let v: Vec<f32> = q.as_slice::<u16>()?.iter().map(|&u| c.inverse(u as f64 / 65535.0) as f32).collect();
    Tensor::from_slice(q.shape(), &v)
}
