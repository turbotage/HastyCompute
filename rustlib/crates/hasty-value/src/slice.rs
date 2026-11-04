//! Slice paths into value trees: `["key"]` selects a dict entry, `[i]` a list / tuple element, and on a tensor
//! `[a:b:c, i, ..., :]` selects like numpy (negative indices, steps, an ellipsis, integer indices drop the
//! dimension). The path ends at a tensor slice. Examples: `["images"][3]`, `["recon"][..., 10, :, :]`, `[0:2, ::4]`.

use crate::{Error, Result, Tensor, Value};

#[derive(Clone, Debug, PartialEq)]
pub enum Sel {
    Index(i64),
    Range(Option<i64>, Option<i64>, Option<i64>),
    Ellipsis,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Segment {
    Key(String),
    Items(Vec<Sel>),
}

fn err<T>(m: impl Into<String>) -> Result<T> {
    Err(Error::Slice(m.into()))
}

pub fn parse(path: &str) -> Result<Vec<Segment>> {
    let s = path.as_bytes();
    let mut i = 0;
    let mut out = Vec::new();
    let ws = |i: &mut usize| {
        while *i < s.len() && s[*i] == b' ' {
            *i += 1;
        }
    };
    loop {
        ws(&mut i);
        if i >= s.len() {
            return Ok(out);
        }
        if s[i] != b'[' {
            return err(format!("expected '[' at {i} in {path:?}"));
        }
        i += 1;
        ws(&mut i);
        if i < s.len() && (s[i] == b'"' || s[i] == b'\'') {
            let q = s[i];
            let st = i + 1;
            let Some(e) = s[st..].iter().position(|&c| c == q) else { return err(format!("unterminated key in {path:?}")) };
            out.push(Segment::Key(path[st..st + e].to_string()));
            i = st + e + 1;
            ws(&mut i);
            if i >= s.len() || s[i] != b']' {
                return err(format!("expected ']' after key in {path:?}"));
            }
            i += 1;
            continue;
        }
        let Some(e) = s[i..].iter().position(|&c| c == b']') else { return err(format!("unterminated '[' in {path:?}")) };
        let body = &path[i..i + e];
        i += e + 1;
        let mut items = Vec::new();
        for part in body.split(',') {
            let p = part.trim();
            if p == "..." {
                items.push(Sel::Ellipsis);
            } else if p.contains(':') {
                let f: Vec<&str> = p.split(':').map(str::trim).collect();
                if f.len() > 3 {
                    return err(format!("bad range {p:?}"));
                }
                let num = |x: Option<&&str>| -> Result<Option<i64>> {
                    match x {
                        None => Ok(None),
                        Some(t) if t.is_empty() => Ok(None),
                        Some(t) => t.parse::<i64>().map(Some).map_err(|_| Error::Slice(format!("bad number {t:?}"))),
                    }
                };
                items.push(Sel::Range(num(f.first())?, num(f.get(1))?, num(f.get(2))?));
            } else {
                items.push(Sel::Index(p.parse::<i64>().map_err(|_| Error::Slice(format!("bad index {p:?}")))?));
            }
        }
        out.push(Segment::Items(items));
    }
}

/// per-dimension source indices and whether the dimension is kept
fn plan(shape: &[usize], items: &[Sel]) -> Result<Vec<(Vec<usize>, bool)>> {
    let nell = items.iter().filter(|s| matches!(s, Sel::Ellipsis)).count();
    if nell > 1 {
        return err("more than one ellipsis");
    }
    let explicit = items.len() - nell;
    if explicit > shape.len() {
        return err(format!("{explicit} indices for a {}-d tensor", shape.len()));
    }
    let mut full: Vec<Sel> = Vec::with_capacity(shape.len());
    for s in items {
        if matches!(s, Sel::Ellipsis) {
            for _ in 0..shape.len() - explicit {
                full.push(Sel::Range(None, None, None));
            }
        } else {
            full.push(s.clone());
        }
    }
    while full.len() < shape.len() {
        full.push(Sel::Range(None, None, None));
    }
    full.iter()
        .zip(shape)
        .map(|(s, &n)| {
            let n = n as i64;
            match *s {
                Sel::Index(i) => {
                    let j = if i < 0 { i + n } else { i };
                    if j < 0 || j >= n {
                        return err(format!("index {i} out of range for size {n}"));
                    }
                    Ok((vec![j as usize], false))
                }
                Sel::Range(a, b, c) => {
                    let st = c.unwrap_or(1);
                    if st == 0 {
                        return err("slice step 0");
                    }
                    // Python slice.indices semantics
                    let norm = |x: i64, lo: i64, hi: i64| -> i64 { if x < 0 { (x + n).max(lo) } else { x.min(hi) } };
                    let (lo, hi) = if st > 0 { (0, n) } else { (-1, n - 1) };
                    let a = a.map(|x| norm(x, lo, hi)).unwrap_or(if st > 0 { 0 } else { n - 1 });
                    let b = b.map(|x| norm(x, lo, hi)).unwrap_or(if st > 0 { n } else { -1 });
                    let mut v = Vec::new();
                    let mut k = a;
                    while (st > 0 && k < b) || (st < 0 && k > b) {
                        v.push(k as usize);
                        k += st;
                    }
                    Ok((v, true))
                }
                Sel::Ellipsis => unreachable!(),
            }
        })
        .collect()
}

/// visit (destination element index, source element index) pairs of a tensor selection, row by row
fn visit(shape: &[usize], p: &[(Vec<usize>, bool)], mut f: impl FnMut(usize, usize, usize)) {
    let nd = shape.len();
    let mut stride = vec![1usize; nd];
    for d in (0..nd.saturating_sub(1)).rev() {
        stride[d] = stride[d + 1] * shape[d + 1];
    }
    let counts: Vec<usize> = p.iter().map(|(v, _)| v.len()).collect();
    if counts.iter().any(|&c| c == 0) {
        return;
    }
    if nd == 0 {
        f(0, 0, 1);
        return;
    }
    // inner dimension: contiguous run if its indices are consecutive
    let inner = &p[nd - 1].0;
    let run = inner.windows(2).all(|w| w[1] == w[0] + 1);
    let mut idx = vec![0usize; nd - 1];
    let mut dst = 0usize;
    loop {
        let base: usize = (0..nd - 1).map(|d| p[d].0[idx[d]] * stride[d]).sum();
        if run {
            f(dst, base + inner[0], inner.len());
            dst += inner.len();
        } else {
            for &j in inner {
                f(dst, base + j, 1);
                dst += 1;
            }
        }
        let mut d = nd - 1;
        loop {
            if d == 0 {
                return;
            }
            d -= 1;
            idx[d] += 1;
            if idx[d] < counts[d] {
                break;
            }
            idx[d] = 0;
        }
    }
}

pub fn tensor_slice(t: &Tensor, items: &[Sel]) -> Result<Tensor> {
    let p = plan(t.shape(), items)?;
    let shape: Vec<usize> = p.iter().filter(|(_, keep)| *keep).map(|(v, _)| v.len()).collect();
    let es = t.dtype().size();
    let mut out = Tensor::zeros(t.dtype(), shape);
    let src = t.bytes();
    let dst = out.bytes_mut();
    visit(t.shape(), &p, |d, s, n| dst[d * es..(d + n) * es].copy_from_slice(&src[s * es..(s + n) * es]));
    Ok(out)
}

pub fn tensor_assign(t: &mut Tensor, items: &[Sel], src: &Tensor) -> Result<()> {
    let p = plan(t.shape(), items)?;
    let shape: Vec<usize> = p.iter().filter(|(_, keep)| *keep).map(|(v, _)| v.len()).collect();
    if src.dtype() != t.dtype() {
        return err(format!("assigning {} into {}", src.dtype().name(), t.dtype().name()));
    }
    if src.shape() != shape.as_slice() && src.numel() != 1 {
        return err(format!("assigning shape {:?} into selection {shape:?}", src.shape()));
    }
    let es = t.dtype().size();
    let broadcast = src.numel() == 1 && shape.iter().product::<usize>() != 1;
    let full_shape = t.shape().to_vec();
    let s = src.bytes();
    let dst = t.bytes_mut();
    visit(&full_shape, &p, |d, o, n| {
        if broadcast {
            for k in 0..n {
                dst[(o + k) * es..(o + k + 1) * es].copy_from_slice(&s[..es]);
            }
        } else {
            dst[o * es..(o + n) * es].copy_from_slice(&s[d * es..(d + n) * es]);
        }
    });
    Ok(())
}

fn walk<'a>(mut v: &'a Value, segs: &'a [Segment]) -> Result<(&'a Value, Option<&'a [Sel]>)> {
    for (k, s) in segs.iter().enumerate() {
        match (s, v) {
            (Segment::Key(key), Value::Dict(_)) => v = v.get(key).ok_or_else(|| Error::Slice(format!("no key {key:?}")))?,
            (Segment::Items(it), Value::List(l) | Value::Tuple(l)) => {
                let [Sel::Index(i)] = it.as_slice() else { return err("lists take a single integer index") };
                let j = if *i < 0 { *i + l.len() as i64 } else { *i };
                v = l.get(j as usize).filter(|_| j >= 0).ok_or_else(|| Error::Slice(format!("index {i} out of range")))?;
            }
            (Segment::Items(it), Value::Tensor(_)) => {
                if k + 1 != segs.len() {
                    return err("a tensor slice must be the last segment");
                }
                return Ok((v, Some(it)));
            }
            (s, v) => return err(format!("cannot apply {s:?} to a {}", v.kind())),
        }
    }
    Ok((v, None))
}

/// The selected sub-value (tensor slices are materialised as new contiguous tensors).
pub fn get(v: &Value, path: &str) -> Result<Value> {
    let segs = parse(path)?;
    match walk(v, &segs)? {
        (Value::Tensor(t), Some(items)) => Ok(Value::Tensor(tensor_slice(t, items)?)),
        (sub, _) => Ok(sub.clone()),
    }
}

/// Replace the selected sub-value (a tensor selection is assigned element-wise; dtype must match).
pub fn set(v: &mut Value, path: &str, new: Value) -> Result<()> {
    let segs = parse(path)?;
    let mut cur = v;
    for (k, s) in segs.iter().enumerate() {
        let last = k + 1 == segs.len();
        cur = match (s, cur) {
            (Segment::Key(key), d @ Value::Dict(_)) => {
                if last {
                    if let Value::Dict(items) = d {
                        match items.iter_mut().find(|(kk, _)| kk == key) {
                            Some((_, x)) => *x = new,
                            None => items.push((key.clone(), new)),
                        }
                    }
                    return Ok(());
                }
                d.get_mut(key).ok_or_else(|| Error::Slice(format!("no key {key:?}")))?
            }
            (Segment::Items(it), Value::List(l) | Value::Tuple(l)) => {
                let [Sel::Index(i)] = it.as_slice() else { return err("lists take a single integer index") };
                let j = if *i < 0 { *i + l.len() as i64 } else { *i };
                if j < 0 || j as usize >= l.len() {
                    return err(format!("index {i} out of range"));
                }
                if last {
                    l[j as usize] = new;
                    return Ok(());
                }
                &mut l[j as usize]
            }
            (Segment::Items(it), Value::Tensor(t)) => {
                if !last {
                    return err("a tensor slice must be the last segment");
                }
                let Value::Tensor(src) = &new else { return err("only a tensor can be assigned into a tensor selection") };
                return tensor_assign(t, it, src);
            }
            (s, c) => return err(format!("cannot apply {s:?} to a {}", c.kind())),
        };
    }
    *cur = new;
    Ok(())
}
