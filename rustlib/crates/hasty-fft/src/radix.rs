//! Emission of straight-line radix-R butterflies as CUDA C++ (`__device__ void bflyR(cx* v)`).
//! `sign` = -1 for the forward transform (e^{-i...}), +1 for the inverse (unnormalised).

use std::fmt::Write;

fn lit(x: f64) -> String {
    format!("RL({:.17e})", x)
}

pub fn emit_butterfly(r: usize, sign: i32) -> String {
    let mut s = String::new();
    writeln!(
        s,
        "__device__ __forceinline__ void bfly{r}{}(cx* v) {{",
        if sign < 0 { "m" } else { "p" }
    )
    .unwrap();
    match r {
        2 => {
            writeln!(
                s,
                "  cx a = v[0], b = v[1]; v[0] = cadd(a, b); v[1] = csub(a, b);"
            )
            .unwrap();
        }
        4 => emit_dft4(&mut s, sign, "v", &[0, 1, 2, 3], "v", &[0, 1, 2, 3], ""),
        8 => {
            writeln!(s, "  cx e[4], o[4];").unwrap();
            // even / odd 4-point DFTs
            emit_dft4(&mut s, sign, "v", &[0, 2, 4, 6], "e", &[0, 1, 2, 3], "e_");
            emit_dft4(&mut s, sign, "v", &[1, 3, 5, 7], "o", &[0, 1, 2, 3], "o_");
            let c = std::f64::consts::FRAC_1_SQRT_2;
            let sg = sign as f64;
            // W8^u = e^{sign 2 pi i u / 8}
            let w = [(1.0, 0.0), (c, sg * c), (0.0, sg), (-c, sg * c)];
            for u in 0..4 {
                let (wr, wi) = w[u];
                if u == 0 {
                    writeln!(
                        s,
                        "  {{ cx t = o[0]; v[0] = cadd(e[0], t); v[4] = csub(e[0], t); }}"
                    )
                    .unwrap();
                } else if u == 2 {
                    // multiply by sign*i
                    writeln!(
                        s,
                        "  {{ cx t = make_cx({}o[2].y, {}o[2].x); v[2] = cadd(e[2], t); v[6] = csub(e[2], t); }}",
                        if sign < 0 { "" } else { "-" },
                        if sign < 0 { "-" } else { "" }
                    )
                    .unwrap();
                } else {
                    writeln!(
                        s,
                        "  {{ cx t = cmul(o[{u}], make_cx({}, {})); v[{u}] = cadd(e[{u}], t); v[{}] = csub(e[{u}], t); }}",
                        lit(wr),
                        lit(wi),
                        u + 4
                    )
                    .unwrap();
                }
            }
        }
        3 | 5 | 7 | 11 | 13 => emit_odd_prime(&mut s, r, sign),
        _ => panic!("no butterfly for radix {r}"),
    }
    writeln!(s, "}}").unwrap();
    s
}

/// 4-point DFT of src[idx[..]] written to dst[odx[..]] (may alias).
fn emit_dft4(
    s: &mut String,
    sign: i32,
    src: &str,
    idx: &[usize],
    dst: &str,
    odx: &[usize],
    tag: &str,
) {
    let (a, b, c, d) = (idx[0], idx[1], idx[2], idx[3]);
    writeln!(s, "  {{ cx {tag}t0 = cadd({src}[{a}], {src}[{c}]), {tag}t1 = csub({src}[{a}], {src}[{c}]), {tag}t2 = cadd({src}[{b}], {src}[{d}]), {tag}t3 = csub({src}[{b}], {src}[{d}]);").unwrap();
    // forward: y1 = t1 - i t3, y3 = t1 + i t3 ; inverse swapped. -i(x,y) = (y,-x); +i(x,y) = (-y,x)
    let (y1, y3) = if sign < 0 {
        (
            format!("make_cx({tag}t1.x + {tag}t3.y, {tag}t1.y - {tag}t3.x)"),
            format!("make_cx({tag}t1.x - {tag}t3.y, {tag}t1.y + {tag}t3.x)"),
        )
    } else {
        (
            format!("make_cx({tag}t1.x - {tag}t3.y, {tag}t1.y + {tag}t3.x)"),
            format!("make_cx({tag}t1.x + {tag}t3.y, {tag}t1.y - {tag}t3.x)"),
        )
    };
    writeln!(s, "    {dst}[{}] = cadd({tag}t0, {tag}t2); {dst}[{}] = {y1}; {dst}[{}] = csub({tag}t0, {tag}t2); {dst}[{}] = {y3}; }}", odx[0], odx[1], odx[2], odx[3]).unwrap();
}

fn emit_odd_prime(s: &mut String, r: usize, sign: i32) {
    let h = (r - 1) / 2;
    for t in 1..=h {
        writeln!(
            s,
            "  cx a{t} = cadd(v[{t}], v[{}]), b{t} = csub(v[{t}], v[{}]);",
            r - t,
            r - t
        )
        .unwrap();
    }
    let mut e = String::from("v[0]");
    for t in 1..=h {
        e = format!("cadd({e}, a{t})");
    }
    writeln!(s, "  cx y0 = {e};").unwrap();
    // y_u = P + s*i*Q = (P.x - s Q.y, P.y + s Q.x),  y_{R-u} = P - s*i*Q,  s = sign
    let (ux, uy) = if sign < 0 { ("+", "-") } else { ("-", "+") }; // y_u = (P.x ux Q.y, P.y uy Q.x)
    for u in 1..=h {
        let mut p = String::from("v[0]");
        let mut q = String::new();
        for t in 1..=h {
            let th = 2.0 * std::f64::consts::PI * ((t * u) % r) as f64 / r as f64;
            p = format!("cadd({p}, cscale(a{t}, {}))", lit(th.cos()));
            let term = format!("cscale(b{t}, {})", lit(th.sin()));
            q = if q.is_empty() {
                term
            } else {
                format!("cadd({q}, {term})")
            };
        }
        writeln!(s, "  cx P{u} = {p}, Q{u} = {q};").unwrap();
        writeln!(
            s,
            "  cx y{u} = make_cx(P{u}.x {ux} Q{u}.y, P{u}.y {uy} Q{u}.x);"
        )
        .unwrap();
        writeln!(
            s,
            "  cx y{} = make_cx(P{u}.x {uy} Q{u}.y, P{u}.y {ux} Q{u}.x);",
            r - u
        )
        .unwrap();
    }
    for k in 0..r {
        writeln!(s, "  v[{k}] = y{k};").unwrap();
    }
}
