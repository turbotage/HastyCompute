//! Straight-line programs: the nodes a set of outputs needs, in topological order, one slot each. Run by the CPU
//! interpreter ([`Program::run`]) or emitted as CUDA C ([`Program::cuda`]) — the same code on both, so the CPU path is
//! an exact reference for the GPU kernels.

use crate::expr::{Fun, Graph, Id, Node};

#[derive(Clone, Copy, Debug)]
pub enum Op {
    Const(f64),
    Param(u16),
    Var(u16),
    Fixed(u16),
    Add(u32, u32),
    Sub(u32, u32),
    Mul(u32, u32),
    Div(u32, u32),
    Neg(u32),
    Powi(u32, i32),
    Pow(u32, u32),
    Un(Fun, u32),
}

#[derive(Clone, Debug)]
pub struct Program {
    pub ops: Vec<Op>,
    /// slot of each output
    pub outputs: Vec<u32>,
}

impl Program {
    pub fn new(g: &Graph, outputs: &[Id]) -> Program {
        let n = g.len();
        let mut need = vec![false; n];
        let mut stack: Vec<Id> = outputs.to_vec();
        while let Some(id) = stack.pop() {
            if need[id as usize] {
                continue;
            }
            need[id as usize] = true;
            match g.node(id) {
                Node::Add(a, b) | Node::Sub(a, b) | Node::Mul(a, b) | Node::Div(a, b) | Node::Pow(a, b) => {
                    stack.push(a);
                    stack.push(b);
                }
                Node::Neg(a) | Node::Powi(a, _) | Node::Un(_, a) => stack.push(a),
                _ => {}
            }
        }
        let mut slot = vec![u32::MAX; n];
        let mut ops = vec![];
        for id in 0..n {
            if !need[id] {
                continue;
            }
            let s = |x: Id| slot[x as usize];
            let op = match g.node(id as Id) {
                Node::Const(b) => Op::Const(f64::from_bits(b)),
                Node::Param(k) => Op::Param(k),
                Node::Var(k) => Op::Var(k),
                Node::Fixed(k) => Op::Fixed(k),
                Node::Add(a, b) => Op::Add(s(a), s(b)),
                Node::Sub(a, b) => Op::Sub(s(a), s(b)),
                Node::Mul(a, b) => Op::Mul(s(a), s(b)),
                Node::Div(a, b) => Op::Div(s(a), s(b)),
                Node::Neg(a) => Op::Neg(s(a)),
                Node::Powi(a, k) => Op::Powi(s(a), k),
                Node::Pow(a, b) => Op::Pow(s(a), s(b)),
                Node::Un(f, a) => Op::Un(f, s(a)),
            };
            slot[id] = ops.len() as u32;
            ops.push(op);
        }
        Program { ops, outputs: outputs.iter().map(|&o| slot[o as usize]).collect() }
    }

    /// evaluate into `slots` (len ≥ ops.len()); outputs are `slots[outputs[k]]`
    #[inline]
    pub fn run(&self, p: &[f64], v: &[f64], fx: &[f64], slots: &mut [f64]) {
        for (i, op) in self.ops.iter().enumerate() {
            slots[i] = match *op {
                Op::Const(c) => c,
                Op::Param(k) => p[k as usize],
                Op::Var(k) => v[k as usize],
                Op::Fixed(k) => fx[k as usize],
                Op::Add(a, b) => slots[a as usize] + slots[b as usize],
                Op::Sub(a, b) => slots[a as usize] - slots[b as usize],
                Op::Mul(a, b) => slots[a as usize] * slots[b as usize],
                Op::Div(a, b) => slots[a as usize] / slots[b as usize],
                Op::Neg(a) => -slots[a as usize],
                Op::Powi(a, n) => slots[a as usize].powi(n),
                Op::Pow(a, b) => slots[a as usize].powf(slots[b as usize]),
                Op::Un(f, a) => f.apply(slots[a as usize]),
            };
        }
    }

    /// CUDA C body: `const T tN = …;` for every op, then `out[k] = …;`. Leaves are read from the arrays `p`, `v`
    /// and `fx` (local arrays in the kernel); `ty` is `float` or `double`
    pub fn cuda(&self, ty: &str, out: &str) -> String {
        let f32 = ty == "float";
        let lit = |c: f64| -> String {
            if c.is_nan() {
                return format!("({ty})NAN");
            }
            if c.is_infinite() {
                return format!("({ty}){}INFINITY", if c < 0.0 { "-" } else { "" });
            }
            if f32 {
                format!("{:e}f", c)
            } else {
                format!("{:e}", c)
            }
        };
        let fname = |f: Fun| -> String {
            let base = match f {
                Fun::Abs => "fabs",
                f => f.name(),
            };
            if f32 {
                format!("{base}f")
            } else {
                base.to_string()
            }
        };
        let mut s = String::new();
        for (i, op) in self.ops.iter().enumerate() {
            let e = match *op {
                Op::Const(c) => lit(c),
                Op::Param(k) => format!("p[{k}]"),
                Op::Var(k) => format!("v[{k}]"),
                Op::Fixed(k) => format!("fx[{k}]"),
                Op::Add(a, b) => format!("t{a} + t{b}"),
                Op::Sub(a, b) => format!("t{a} - t{b}"),
                Op::Mul(a, b) => format!("t{a} * t{b}"),
                Op::Div(a, b) => format!("t{a} / t{b}"),
                Op::Neg(a) => format!("-t{a}"),
                Op::Powi(a, n) => match n {
                    2 => format!("t{a} * t{a}"),
                    3 => format!("t{a} * t{a} * t{a}"),
                    4 => format!("(t{a} * t{a}) * (t{a} * t{a})"),
                    -1 => format!("{} / t{a}", lit(1.0)),
                    -2 => format!("{} / (t{a} * t{a})", lit(1.0)),
                    n => format!("{}(t{a}, {})", if f32 { "powf" } else { "pow" }, lit(n as f64)),
                },
                Op::Pow(a, b) => format!("{}(t{a}, t{b})", if f32 { "powf" } else { "pow" }),
                Op::Un(Fun::Sign, a) => format!("({ty})((t{a} > {z}) - (t{a} < {z}))", z = lit(0.0)),
                Op::Un(f, a) => format!("{}(t{a})", fname(f)),
            };
            s += &format!("    const {ty} t{i} = {e};\n");
        }
        for (k, o) in self.outputs.iter().enumerate() {
            s += &format!("    {out}[{k}] = t{o};\n");
        }
        s
    }
}
