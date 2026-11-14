//! Named commands run on bank values (`Execute`). A command gets its JSON options and input values and returns
//! output values (stored back in the bank) plus a message. Commands run on the blocking pool; GPU commands will run
//! through per-device workers. Base set: host-side helpers the viewer and the node editor need.

use std::collections::BTreeMap;
use std::sync::Arc;

use hasty_value::quant::{auto_config, decode_u16, encode_u16, Comprep};
use hasty_value::{slice, DType, Tensor, Value};
use num_complex::{Complex32, Complex64};
use serde_json::Value as Json;

use crate::bank::Bank;
use crate::Error;

pub struct Ctx {
    pub bank: Arc<Bank>,
}

pub type Output = (String, Vec<Value>);
type Fun = dyn Fn(&Ctx, &Json, &[Arc<Value>]) -> Result<Output, Error> + Send + Sync;

pub struct Command {
    pub doc: String,
    f: Box<Fun>,
}

#[derive(Default)]
pub struct Registry {
    cmds: BTreeMap<String, Command>,
}

fn inv<T>(m: impl Into<String>) -> Result<T, Error> {
    Err(Error::Invalid(m.into()))
}

impl Registry {
    pub fn new() -> Self {
        Self::default()
    }

    /// registry with the base commands
    pub fn base() -> Self {
        let mut r = Self::new();
        base_commands(&mut r);
        crate::flow::register(&mut r);
        crate::models::register(&mut r);
        r
    }

    pub fn register(&mut self, name: &str, doc: &str, f: impl Fn(&Ctx, &Json, &[Arc<Value>]) -> Result<Output, Error> + Send + Sync + 'static) {
        self.cmds.insert(name.to_string(), Command { doc: doc.to_string(), f: Box::new(f) });
    }

    pub fn list(&self) -> Vec<(String, String)> {
        self.cmds.iter().map(|(n, c)| (n.clone(), c.doc.clone())).collect()
    }

    pub fn run(&self, name: &str, ctx: &Ctx, options: &str, inputs: &[Arc<Value>]) -> Result<Output, Error> {
        let c = self.cmds.get(name).ok_or_else(|| Error::Invalid(format!("unknown command {name:?}")))?;
        let opts: Json = if options.trim().is_empty() { Json::Object(Default::default()) } else { serde_json::from_str(options).map_err(|e| Error::Invalid(format!("{name}: options are not JSON: {e}")))? };
        (c.f)(ctx, &opts, inputs)
    }
}

fn tensor<'a>(name: &str, v: &'a [Arc<Value>], i: usize) -> Result<&'a Tensor, Error> {
    v.get(i).and_then(|x| x.as_tensor()).ok_or_else(|| Error::Invalid(format!("{name}: input {i} must be a tensor")))
}

/// elementwise data as f64 or complex f64
enum Num {
    R(Vec<f64>),
    C(Vec<Complex64>),
}

fn num(t: &Tensor) -> Num {
    let n = t.numel();
    if t.dtype().is_complex() {
        Num::C((0..n).map(|i| {
            let (a, b) = t.get_f64(i);
            Complex64::new(a, b)
        }).collect())
    } else {
        Num::R((0..n).map(|i| t.get_f64(i).0).collect())
    }
}

/// result dtype: keep float / complex precision, integers become float64
fn out_dtype(a: DType, complex: bool) -> DType {
    match (a, complex) {
        (DType::F32 | DType::C64 | DType::F16 | DType::BF16, false) => DType::F32,
        (DType::F32 | DType::C64 | DType::F16 | DType::BF16, true) => DType::C64,
        (_, false) => DType::F64,
        (_, true) => DType::C128,
    }
}

fn to_tensor(shape: &[usize], n: Num, d: DType) -> Result<Tensor, Error> {
    Ok(match (n, d) {
        (Num::R(v), DType::F32) => Tensor::from_slice(shape, &v.iter().map(|&x| x as f32).collect::<Vec<_>>())?,
        (Num::R(v), _) => Tensor::from_slice(shape, &v)?,
        (Num::C(v), DType::C64) => Tensor::from_slice(shape, &v.iter().map(|x| Complex32::new(x.re as f32, x.im as f32)).collect::<Vec<_>>())?,
        (Num::C(v), _) => Tensor::from_slice(shape, &v)?,
    })
}

fn unary(r: &mut Registry, name: &str, doc: &str, real_out: bool, f: fn(Complex64) -> Complex64) {
    let n = name.to_string();
    r.register(name, doc, move |_, _, inp| {
        let t = tensor(&n, inp, 0)?;
        let c = match num(t) {
            Num::R(v) => v.into_iter().map(|x| f(Complex64::new(x, 0.0))).collect::<Vec<_>>(),
            Num::C(v) => v.into_iter().map(f).collect(),
        };
        let complex = !real_out && (t.dtype().is_complex() || c.iter().any(|z| z.im != 0.0));
        let out = if complex { Num::C(c) } else { Num::R(c.into_iter().map(|z| z.re).collect()) };
        Ok((String::new(), vec![Value::Tensor(to_tensor(t.shape(), out, out_dtype(t.dtype(), complex))?)]))
    });
}

fn binary(r: &mut Registry, name: &str, f: fn(Complex64, Complex64) -> Complex64) {
    let n = name.to_string();
    r.register(name, &format!("{name}(a, b) elementwise; b may be a scalar tensor or options {{\"value\": x}}"), move |_, o, inp| {
        let a = tensor(&n, inp, 0)?;
        let b: Num = match (inp.get(1).and_then(|v| v.as_tensor()), o["value"].as_f64()) {
            (Some(b), _) => num(b),
            (None, Some(x)) => Num::R(vec![x]),
            _ => return inv(format!("{n}: needs a second tensor or options.value")),
        };
        let blen = match &b { Num::R(v) => v.len(), Num::C(v) => v.len() };
        if blen != 1 && blen != a.numel() {
            return inv(format!("{n}: shapes differ ({} vs {blen} elements)", a.numel()));
        }
        let bc = |i: usize| -> Complex64 {
            let k = if blen == 1 { 0 } else { i };
            match &b { Num::R(v) => Complex64::new(v[k], 0.0), Num::C(v) => v[k] }
        };
        let complex = a.dtype().is_complex() || matches!(b, Num::C(_));
        let c: Vec<Complex64> = match num(a) {
            Num::R(v) => v.into_iter().enumerate().map(|(i, x)| f(Complex64::new(x, 0.0), bc(i))).collect(),
            Num::C(v) => v.into_iter().enumerate().map(|(i, x)| f(x, bc(i))).collect(),
        };
        let out = if complex { Num::C(c) } else { Num::R(c.into_iter().map(|z| z.re).collect()) };
        Ok((String::new(), vec![Value::Tensor(to_tensor(a.shape(), out, out_dtype(a.dtype(), complex))?)]))
    });
}

fn shape_opt(o: &Json) -> Result<Vec<usize>, Error> {
    o["shape"].as_array().map(|a| a.iter().map(|x| x.as_u64().unwrap_or(0) as usize).collect()).ok_or_else(|| Error::Invalid("options.shape missing".into()))
}

/// in-place N-D FFT over `axes` of a complex128 buffer with C-order `shape`
fn fft_nd(data: &mut [Complex64], shape: &[usize], axes: &[usize], inverse: bool) {
    let mut planner = rustfft::FftPlanner::<f64>::new();
    for &ax in axes {
        let n = shape[ax];
        if n <= 1 {
            continue;
        }
        let fft = if inverse { planner.plan_fft_inverse(n) } else { planner.plan_fft_forward(n) };
        let stride: usize = shape[ax + 1..].iter().product();
        let outer: usize = shape[..ax].iter().product();
        let mut line = vec![Complex64::new(0.0, 0.0); n];
        for o in 0..outer {
            for s in 0..stride {
                let base = o * n * stride + s;
                for (k, x) in line.iter_mut().enumerate() {
                    *x = data[base + k * stride];
                }
                fft.process(&mut line);
                let sc = if inverse { 1.0 / n as f64 } else { 1.0 };
                for (k, x) in line.iter().enumerate() {
                    data[base + k * stride] = x * sc;
                }
            }
        }
    }
}

fn base_commands(r: &mut Registry) {
    r.register("summary", "JSON summary of the input value (message)", |_, _, inp| {
        let v = inp.first().ok_or_else(|| Error::Invalid("summary: one input".into()))?;
        Ok((v.summary().to_string(), vec![]))
    });
    r.register("slice", "materialise a slice: options {\"path\": \"[\\\"key\\\"][0, :, 3]\"}", |_, o, inp| {
        let v = inp.first().ok_or_else(|| Error::Invalid("slice: one input".into()))?;
        Ok((String::new(), vec![slice::get(v, o["path"].as_str().unwrap_or(""))?]))
    });
    r.register(
        "comprep_u16",
        "compressed u16 display representation (C++ comprep; decoded by the orthoslicer): options = a full config {a, b, clampa, clampb, focus, left/right_mode linear|gamma|log, left/right_gamma, left/right_logc, mode abs|real|imag|angle} or a partial one (missing a / b: p_lo / p_hi percentiles, default 1 / 99; clamp: min / max). Output [u16 tensor], message = the full config",
        |_, o, inp| {
            let t = tensor("comprep_u16", inp, 0)?;
            let c = if o.get("a").is_some() && o.get("b").is_some() { Comprep::from_json(o)? } else { auto_config(t, o)? };
            Ok((c.to_json().to_string(), vec![Value::Tensor(encode_u16(t, &c)?)]))
        },
    );
    r.register("decomprep_u16", "u16 comprep tensor → float32 values: options = its config", |_, o, inp| {
        let t = tensor("decomprep_u16", inp, 0)?;
        Ok((String::new(), vec![Value::Tensor(decode_u16(t, &Comprep::from_json(o)?)?)]))
    });
    r.register("stats", "statistics (message JSON: min, max, mean, std, p0_5, p1, p99, p99_5) of options.mode (abs | real | imag | angle; real input: the values)", |_, o, inp| {
        let t = tensor("stats", inp, 0)?;
        let mode = o["mode"].as_str().unwrap_or(if t.dtype().is_complex() { "abs" } else { "real" });
        let cplx = t.dtype().is_complex();
        let f = |i: usize| {
            let (a, b) = t.get_f64(i);
            if !cplx {
                return if mode == "abs" { a.abs() } else { a };
            }
            match mode {
                "real" => a,
                "imag" => b,
                "angle" => b.atan2(a),
                _ => a.hypot(b),
            }
        };
        let n = t.numel();
        let (mut s, mut s2, mut lo, mut hi, mut k) = (0.0, 0.0, f64::INFINITY, f64::NEG_INFINITY, 0usize);
        for i in 0..n {
            let v = f(i);
            if v.is_finite() {
                s += v;
                s2 += v * v;
                lo = lo.min(v);
                hi = hi.max(v);
                k += 1;
            }
        }
        let step = (n / 1_000_000).max(1);
        let mut smp: Vec<f64> = (0..n).step_by(step).map(f).filter(|v| v.is_finite()).collect();
        smp.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q = |p: f64| if smp.is_empty() { f64::NAN } else { smp[(((p / 100.0) * (smp.len() - 1) as f64).round() as usize).min(smp.len() - 1)] };
        let mean = s / k.max(1) as f64;
        let js = serde_json::json!({ "mode": mode, "n": k, "min": lo, "max": hi, "mean": mean, "std": (s2 / k.max(1) as f64 - mean * mean).max(0.0).sqrt(),
                                     "p0_5": q(0.5), "p1": q(1.0), "p99": q(99.0), "p99_5": q(99.5) });
        Ok((js.to_string(), vec![]))
    });
    unary(r, "abs", "|x|", true, |z| Complex64::new(z.norm(), 0.0));
    unary(r, "real", "Re x", true, |z| Complex64::new(z.re, 0.0));
    unary(r, "imag", "Im x", true, |z| Complex64::new(z.im, 0.0));
    unary(r, "angle", "arg x", true, |z| Complex64::new(z.arg(), 0.0));
    unary(r, "neg", "−x", false, |z| -z);
    unary(r, "conj", "conj x", false, |z| z.conj());
    binary(r, "add", |a, b| a + b);
    binary(r, "sub", |a, b| a - b);
    binary(r, "mul", |a, b| a * b);
    binary(r, "div", |a, b| a / b);
    for (name, fill) in [("zeros", 0.0f64), ("ones", 1.0)] {
        r.register(name, "options {shape: [...], dtype: float32|float64|complex64|...}", move |_, o, _| {
            let shape = shape_opt(o)?;
            let d = DType::from_name(o["dtype"].as_str().unwrap_or("float32"))?;
            let n: usize = shape.iter().product();
            let t = match d {
                DType::C64 | DType::C128 => to_tensor(&shape, Num::C(vec![Complex64::new(fill, 0.0); n]), d)?,
                DType::F32 | DType::F64 => to_tensor(&shape, Num::R(vec![fill; n]), d)?,
                _ => return inv("zeros/ones: float or complex dtype"),
            };
            Ok((String::new(), vec![Value::Tensor(t)]))
        });
    }
    r.register("rand", "uniform [0, 1) (complex: both parts): options {shape, dtype, seed}", |_, o, _| {
        let shape = shape_opt(o)?;
        let d = DType::from_name(o["dtype"].as_str().unwrap_or("float32"))?;
        let mut s = o["seed"].as_u64().unwrap_or(0x9E37_79B9_7F4A_7C15) | 1;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        let n: usize = shape.iter().product();
        let t = if d.is_complex() { to_tensor(&shape, Num::C((0..n).map(|_| Complex64::new(next(), next())).collect()), d)? } else { to_tensor(&shape, Num::R((0..n).map(|_| next()).collect()), d)? };
        Ok((String::new(), vec![Value::Tensor(t)]))
    });
    for (name, inverse) in [("fft", false), ("ifft", true)] {
        r.register(name, "N-D DFT over options.axes (default: all; negative axes allowed), unnormalised forward / 1/N inverse, CPU (rustfft)", move |_, o, inp| {
            let t = tensor(name, inp, 0)?;
            let nd = t.shape().len() as i64;
            let axes: Vec<usize> = match o["axes"].as_array() {
                Some(a) => a.iter().map(|x| x.as_i64().unwrap_or(0)).map(|a| (if a < 0 { a + nd } else { a }) as usize).collect(),
                None => (0..nd as usize).collect(),
            };
            if axes.iter().any(|&a| a >= nd as usize) {
                return inv(format!("{name}: axis out of range"));
            }
            let mut c = match num(t) {
                Num::R(v) => v.into_iter().map(|x| Complex64::new(x, 0.0)).collect::<Vec<_>>(),
                Num::C(v) => v,
            };
            fft_nd(&mut c, t.shape(), &axes, inverse);
            Ok((String::new(), vec![Value::Tensor(to_tensor(t.shape(), Num::C(c), out_dtype(t.dtype(), true))?)]))
        });
    }
}
