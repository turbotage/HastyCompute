//! Symbolic expressions: a hash-consed DAG with simplifying constructors, a string parser and symbolic derivatives.
//!
//! Every node is interned once (`Graph::intern`), so equal subexpressions share one id and common-subexpression
//! elimination is automatic. Children are always created before their parents, so node ids are a topological order;
//! [`crate::program::Program`] turns any set of output nodes into straight-line code for the CPU interpreter and CUDA.
//!
//! Leaves: numeric constants, fitted parameters (`Param`), per-data-point variables shared by all problems (`Var`,
//! e.g. echo time, b-value) and per-problem fixed inputs (`Fixed`, e.g. a B1 or T1 map).

use std::collections::HashMap;

pub type Id = u32;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Fun {
    Exp,
    Log,
    Sqrt,
    Sin,
    Cos,
    Tan,
    Tanh,
    Sinh,
    Cosh,
    Atan,
    Abs,
    /// −1, 0, 1 (derivative 0)
    Sign,
}

impl Fun {
    pub fn name(&self) -> &'static str {
        match self {
            Fun::Exp => "exp",
            Fun::Log => "log",
            Fun::Sqrt => "sqrt",
            Fun::Sin => "sin",
            Fun::Cos => "cos",
            Fun::Tan => "tan",
            Fun::Tanh => "tanh",
            Fun::Sinh => "sinh",
            Fun::Cosh => "cosh",
            Fun::Atan => "atan",
            Fun::Abs => "abs",
            Fun::Sign => "sign",
        }
    }
    pub fn apply(&self, x: f64) -> f64 {
        match self {
            Fun::Exp => x.exp(),
            Fun::Log => x.ln(),
            Fun::Sqrt => x.sqrt(),
            Fun::Sin => x.sin(),
            Fun::Cos => x.cos(),
            Fun::Tan => x.tan(),
            Fun::Tanh => x.tanh(),
            Fun::Sinh => x.sinh(),
            Fun::Cosh => x.cosh(),
            Fun::Atan => x.atan(),
            Fun::Abs => x.abs(),
            Fun::Sign => {
                if x > 0.0 {
                    1.0
                } else if x < 0.0 {
                    -1.0
                } else {
                    0.0
                }
            }
        }
    }
    fn from_name(s: &str) -> Option<Fun> {
        Some(match s {
            "exp" => Fun::Exp,
            "log" | "ln" => Fun::Log,
            "sqrt" => Fun::Sqrt,
            "sin" => Fun::Sin,
            "cos" => Fun::Cos,
            "tan" => Fun::Tan,
            "tanh" => Fun::Tanh,
            "sinh" => Fun::Sinh,
            "cosh" => Fun::Cosh,
            "atan" => Fun::Atan,
            "abs" => Fun::Abs,
            "sign" => Fun::Sign,
            _ => return None,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Node {
    /// f64 bits
    Const(u64),
    Param(u16),
    Var(u16),
    Fixed(u16),
    Add(Id, Id),
    Sub(Id, Id),
    Mul(Id, Id),
    Div(Id, Id),
    Neg(Id),
    Powi(Id, i32),
    Pow(Id, Id),
    Un(Fun, Id),
}

/// expression graph with named leaves
#[derive(Clone, Debug)]
pub struct Graph {
    nodes: Vec<Node>,
    index: HashMap<Node, Id>,
    pub params: Vec<String>,
    pub vars: Vec<String>,
    pub fixed: Vec<String>,
}

impl Graph {
    pub fn new(params: &[&str], vars: &[&str], fixed: &[&str]) -> Graph {
        let mut g = Graph { nodes: vec![], index: HashMap::new(), params: params.iter().map(|s| s.to_string()).collect(), vars: vars.iter().map(|s| s.to_string()).collect(), fixed: fixed.iter().map(|s| s.to_string()).collect() };
        // the constants 0 and 1 first
        g.c(0.0);
        g.c(1.0);
        g
    }
    pub fn len(&self) -> usize {
        self.nodes.len()
    }
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
    pub fn node(&self, id: Id) -> Node {
        self.nodes[id as usize]
    }
    fn intern(&mut self, n: Node) -> Id {
        if let Some(&id) = self.index.get(&n) {
            return id;
        }
        let id = self.nodes.len() as Id;
        self.nodes.push(n);
        self.index.insert(n, id);
        id
    }
    pub fn constant(&self, id: Id) -> Option<f64> {
        match self.nodes[id as usize] {
            Node::Const(b) => Some(f64::from_bits(b)),
            _ => None,
        }
    }
    fn is(&self, id: Id, v: f64) -> bool {
        self.constant(id) == Some(v)
    }

    // ── leaves ──
    pub fn c(&mut self, v: f64) -> Id {
        // one representation of zero
        let v = if v == 0.0 { 0.0 } else { v };
        self.intern(Node::Const(v.to_bits()))
    }
    pub fn param(&mut self, i: usize) -> Id {
        assert!(i < self.params.len());
        self.intern(Node::Param(i as u16))
    }
    pub fn var(&mut self, i: usize) -> Id {
        assert!(i < self.vars.len());
        self.intern(Node::Var(i as u16))
    }
    pub fn fixed(&mut self, i: usize) -> Id {
        assert!(i < self.fixed.len());
        self.intern(Node::Fixed(i as u16))
    }

    // ── simplifying constructors ──
    pub fn add(&mut self, a: Id, b: Id) -> Id {
        match (self.constant(a), self.constant(b)) {
            (Some(x), Some(y)) => return self.c(x + y),
            (Some(x), _) if x == 0.0 => return b,
            (_, Some(y)) if y == 0.0 => return a,
            _ => {}
        }
        if a == b {
            let two = self.c(2.0);
            return self.mul(two, a);
        }
        if let Node::Neg(nb) = self.node(b) {
            return self.sub(a, nb);
        }
        if let Node::Neg(na) = self.node(a) {
            return self.sub(b, na);
        }
        let (a, b) = if a <= b { (a, b) } else { (b, a) };
        self.intern(Node::Add(a, b))
    }
    pub fn sub(&mut self, a: Id, b: Id) -> Id {
        match (self.constant(a), self.constant(b)) {
            (Some(x), Some(y)) => return self.c(x - y),
            (_, Some(y)) if y == 0.0 => return a,
            (Some(x), _) if x == 0.0 => return self.neg(b),
            _ => {}
        }
        if a == b {
            return self.c(0.0);
        }
        if let Node::Neg(nb) = self.node(b) {
            return self.add(a, nb);
        }
        self.intern(Node::Sub(a, b))
    }
    pub fn mul(&mut self, a: Id, b: Id) -> Id {
        match (self.constant(a), self.constant(b)) {
            (Some(x), Some(y)) => return self.c(x * y),
            (Some(x), _) if x == 0.0 => return self.c(0.0),
            (_, Some(y)) if y == 0.0 => return self.c(0.0),
            (Some(x), _) if x == 1.0 => return b,
            (_, Some(y)) if y == 1.0 => return a,
            (Some(x), _) if x == -1.0 => return self.neg(b),
            (_, Some(y)) if y == -1.0 => return self.neg(a),
            _ => {}
        }
        if a == b {
            return self.powi(a, 2);
        }
        // pull negations out: (−a)·b = −(a·b)
        if let Node::Neg(na) = self.node(a) {
            let m = self.mul(na, b);
            return self.neg(m);
        }
        if let Node::Neg(nb) = self.node(b) {
            let m = self.mul(a, nb);
            return self.neg(m);
        }
        // constant first, and fold c1·(c2·x)
        let (a, b) = if self.constant(b).is_some() && self.constant(a).is_none() { (b, a) } else if self.constant(a).is_none() && a > b { (b, a) } else { (a, b) };
        if let (Some(x), Node::Mul(c2, y)) = (self.constant(a), self.node(b)) {
            if let Some(z) = self.constant(c2) {
                let k = self.c(x * z);
                return self.mul(k, y);
            }
        }
        self.intern(Node::Mul(a, b))
    }
    pub fn div(&mut self, a: Id, b: Id) -> Id {
        match (self.constant(a), self.constant(b)) {
            (Some(x), Some(y)) => return self.c(x / y),
            (Some(x), _) if x == 0.0 => return self.c(0.0),
            (_, Some(y)) if y == 1.0 => return a,
            (_, Some(y)) if y == -1.0 => return self.neg(a),
            (_, Some(y)) => {
                let r = self.c(1.0 / y);
                return self.mul(r, a);
            }
            _ => {}
        }
        if a == b {
            return self.c(1.0);
        }
        self.intern(Node::Div(a, b))
    }
    pub fn neg(&mut self, a: Id) -> Id {
        if let Some(x) = self.constant(a) {
            return self.c(-x);
        }
        if let Node::Neg(x) = self.node(a) {
            return x;
        }
        if let Node::Sub(x, y) = self.node(a) {
            return self.sub(y, x);
        }
        self.intern(Node::Neg(a))
    }
    pub fn powi(&mut self, a: Id, n: i32) -> Id {
        if n == 0 {
            return self.c(1.0);
        }
        if n == 1 {
            return a;
        }
        if let Some(x) = self.constant(a) {
            return self.c(x.powi(n));
        }
        if let Node::Powi(x, m) = self.node(a) {
            return self.powi(x, m * n);
        }
        self.intern(Node::Powi(a, n))
    }
    pub fn pow(&mut self, a: Id, b: Id) -> Id {
        if let Some(y) = self.constant(b) {
            if y.fract() == 0.0 && y.abs() <= 64.0 {
                return self.powi(a, y as i32);
            }
            if y == 0.5 {
                return self.fun(Fun::Sqrt, a);
            }
            if let Some(x) = self.constant(a) {
                return self.c(x.powf(y));
            }
        }
        self.intern(Node::Pow(a, b))
    }
    pub fn fun(&mut self, f: Fun, a: Id) -> Id {
        if let Some(x) = self.constant(a) {
            return self.c(f.apply(x));
        }
        match (f, self.node(a)) {
            (Fun::Exp, Node::Un(Fun::Log, x)) => return x,
            (Fun::Log, Node::Un(Fun::Exp, x)) => return x,
            (Fun::Sign, Node::Un(Fun::Exp, _)) => return self.c(1.0),
            _ => {}
        }
        self.intern(Node::Un(f, a))
    }

    /// value of node `id` for given leaves (recursive; for tests and small checks)
    pub fn eval(&self, id: Id, p: &[f64], v: &[f64], fx: &[f64]) -> f64 {
        let mut memo = vec![f64::NAN; self.nodes.len()];
        // ids are topological: evaluate everything up to id
        for i in 0..=id as usize {
            memo[i] = match self.nodes[i] {
                Node::Const(b) => f64::from_bits(b),
                Node::Param(k) => p[k as usize],
                Node::Var(k) => v[k as usize],
                Node::Fixed(k) => fx[k as usize],
                Node::Add(a, b) => memo[a as usize] + memo[b as usize],
                Node::Sub(a, b) => memo[a as usize] - memo[b as usize],
                Node::Mul(a, b) => memo[a as usize] * memo[b as usize],
                Node::Div(a, b) => memo[a as usize] / memo[b as usize],
                Node::Neg(a) => -memo[a as usize],
                Node::Powi(a, n) => memo[a as usize].powi(n),
                Node::Pow(a, b) => memo[a as usize].powf(memo[b as usize]),
                Node::Un(f, a) => f.apply(memo[a as usize]),
            };
        }
        memo[id as usize]
    }

    /// ∂e/∂param(p); `memo` caches derivatives of nodes for this parameter (share it across outputs)
    pub fn diff(&mut self, e: Id, p: usize, memo: &mut HashMap<Id, Id>) -> Id {
        if let Some(&d) = memo.get(&e) {
            return d;
        }
        let d = match self.node(e) {
            Node::Const(_) | Node::Var(_) | Node::Fixed(_) => self.c(0.0),
            Node::Param(k) => self.c(if k as usize == p { 1.0 } else { 0.0 }),
            Node::Add(a, b) => {
                let (da, db) = (self.diff(a, p, memo), self.diff(b, p, memo));
                self.add(da, db)
            }
            Node::Sub(a, b) => {
                let (da, db) = (self.diff(a, p, memo), self.diff(b, p, memo));
                self.sub(da, db)
            }
            Node::Mul(a, b) => {
                let (da, db) = (self.diff(a, p, memo), self.diff(b, p, memo));
                let t1 = self.mul(da, b);
                let t2 = self.mul(a, db);
                self.add(t1, t2)
            }
            Node::Div(a, b) => {
                // d(a/b) = (da − (a/b)·db) / b, reusing the quotient node
                let (da, db) = (self.diff(a, p, memo), self.diff(b, p, memo));
                if self.is(db, 0.0) {
                    self.div(da, b)
                } else {
                    let t = self.mul(e, db);
                    let n = self.sub(da, t);
                    self.div(n, b)
                }
            }
            Node::Neg(a) => {
                let da = self.diff(a, p, memo);
                self.neg(da)
            }
            Node::Powi(a, n) => {
                let da = self.diff(a, p, memo);
                if self.is(da, 0.0) {
                    self.c(0.0)
                } else {
                    let pw = self.powi(a, n - 1);
                    let k = self.c(n as f64);
                    let t = self.mul(k, pw);
                    self.mul(t, da)
                }
            }
            Node::Pow(a, b) => {
                // d(a^b) = a^b (db log a + b da / a)
                let (da, db) = (self.diff(a, p, memo), self.diff(b, p, memo));
                let la = self.fun(Fun::Log, a);
                let t1 = self.mul(db, la);
                let q = self.div(da, a);
                let t2 = self.mul(b, q);
                let s = self.add(t1, t2);
                self.mul(e, s)
            }
            Node::Un(f, a) => {
                let da = self.diff(a, p, memo);
                if self.is(da, 0.0) {
                    self.c(0.0)
                } else {
                    let inner = match f {
                        Fun::Exp => e,
                        Fun::Log => {
                            let one = self.c(1.0);
                            self.div(one, a)
                        }
                        Fun::Sqrt => {
                            let h = self.c(0.5);
                            self.div(h, e)
                        }
                        Fun::Sin => self.fun(Fun::Cos, a),
                        Fun::Cos => {
                            let s = self.fun(Fun::Sin, a);
                            self.neg(s)
                        }
                        Fun::Tan => {
                            let one = self.c(1.0);
                            let t2 = self.powi(e, 2);
                            self.add(one, t2)
                        }
                        Fun::Tanh => {
                            let one = self.c(1.0);
                            let t2 = self.powi(e, 2);
                            self.sub(one, t2)
                        }
                        Fun::Sinh => self.fun(Fun::Cosh, a),
                        Fun::Cosh => self.fun(Fun::Sinh, a),
                        Fun::Atan => {
                            let one = self.c(1.0);
                            let a2 = self.powi(a, 2);
                            let d = self.add(one, a2);
                            self.div(one, d)
                        }
                        Fun::Abs => self.fun(Fun::Sign, a),
                        Fun::Sign => self.c(0.0),
                    };
                    self.mul(inner, da)
                }
            }
        };
        memo.insert(e, d);
        d
    }

    /// parse an expression over this graph's names (`pi` is a constant; `^` or `**` is power)
    pub fn parse(&mut self, s: &str) -> Result<Id, String> {
        let toks = lex(s)?;
        let mut p = Parser { t: toks, i: 0, src: s };
        let e = p.expr(self)?;
        if p.i != p.t.len() {
            return Err(format!("unexpected `{}` in `{s}`", p.t[p.i].text()));
        }
        Ok(e)
    }

    /// readable infix form (for diagnostics)
    pub fn show(&self, id: Id) -> String {
        match self.node(id) {
            Node::Const(b) => format!("{}", f64::from_bits(b)),
            Node::Param(k) => self.params[k as usize].clone(),
            Node::Var(k) => self.vars[k as usize].clone(),
            Node::Fixed(k) => self.fixed[k as usize].clone(),
            Node::Add(a, b) => format!("({} + {})", self.show(a), self.show(b)),
            Node::Sub(a, b) => format!("({} - {})", self.show(a), self.show(b)),
            Node::Mul(a, b) => format!("({} * {})", self.show(a), self.show(b)),
            Node::Div(a, b) => format!("({} / {})", self.show(a), self.show(b)),
            Node::Neg(a) => format!("(-{})", self.show(a)),
            Node::Powi(a, n) => format!("{}^{n}", self.show(a)),
            Node::Pow(a, b) => format!("{}^{}", self.show(a), self.show(b)),
            Node::Un(f, a) => format!("{}({})", f.name(), self.show(a)),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Num(f64),
    Ident(String),
    Op(char),
}

impl Tok {
    fn text(&self) -> String {
        match self {
            Tok::Num(x) => x.to_string(),
            Tok::Ident(s) => s.clone(),
            Tok::Op(c) => c.to_string(),
        }
    }
}

fn lex(s: &str) -> Result<Vec<Tok>, String> {
    let b: Vec<char> = s.chars().collect();
    let mut out = vec![];
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        if c.is_whitespace() {
            i += 1;
        } else if c.is_ascii_digit() || (c == '.' && i + 1 < b.len() && b[i + 1].is_ascii_digit()) {
            let st = i;
            while i < b.len() && (b[i].is_ascii_digit() || b[i] == '.') {
                i += 1;
            }
            if i < b.len() && (b[i] == 'e' || b[i] == 'E') {
                let mut j = i + 1;
                if j < b.len() && (b[j] == '+' || b[j] == '-') {
                    j += 1;
                }
                if j < b.len() && b[j].is_ascii_digit() {
                    i = j;
                    while i < b.len() && b[i].is_ascii_digit() {
                        i += 1;
                    }
                }
            }
            let t: String = b[st..i].iter().collect();
            out.push(Tok::Num(t.parse().map_err(|_| format!("bad number `{t}`"))?));
        } else if c.is_alphabetic() || c == '_' {
            let st = i;
            while i < b.len() && (b[i].is_alphanumeric() || b[i] == '_') {
                i += 1;
            }
            out.push(Tok::Ident(b[st..i].iter().collect()));
        } else if c == '*' && i + 1 < b.len() && b[i + 1] == '*' {
            out.push(Tok::Op('^'));
            i += 2;
        } else if "+-*/^(),".contains(c) {
            out.push(Tok::Op(c));
            i += 1;
        } else {
            return Err(format!("unexpected character `{c}` in `{s}`"));
        }
    }
    Ok(out)
}

struct Parser<'a> {
    t: Vec<Tok>,
    i: usize,
    src: &'a str,
}

impl Parser<'_> {
    fn peek(&self) -> Option<&Tok> {
        self.t.get(self.i)
    }
    fn eat(&mut self, c: char) -> bool {
        if self.peek() == Some(&Tok::Op(c)) {
            self.i += 1;
            true
        } else {
            false
        }
    }
    fn expr(&mut self, g: &mut Graph) -> Result<Id, String> {
        let mut a = self.term(g)?;
        loop {
            if self.eat('+') {
                let b = self.term(g)?;
                a = g.add(a, b);
            } else if self.eat('-') {
                let b = self.term(g)?;
                a = g.sub(a, b);
            } else {
                return Ok(a);
            }
        }
    }
    fn term(&mut self, g: &mut Graph) -> Result<Id, String> {
        let mut a = self.unary(g)?;
        loop {
            if self.eat('*') {
                let b = self.unary(g)?;
                a = g.mul(a, b);
            } else if self.eat('/') {
                let b = self.unary(g)?;
                a = g.div(a, b);
            } else {
                return Ok(a);
            }
        }
    }
    fn unary(&mut self, g: &mut Graph) -> Result<Id, String> {
        if self.eat('-') {
            let a = self.unary(g)?;
            return Ok(g.neg(a));
        }
        if self.eat('+') {
            return self.unary(g);
        }
        self.power(g)
    }
    fn power(&mut self, g: &mut Graph) -> Result<Id, String> {
        let a = self.atom(g)?;
        if self.eat('^') {
            let b = self.unary(g)?;
            return Ok(g.pow(a, b));
        }
        Ok(a)
    }
    fn atom(&mut self, g: &mut Graph) -> Result<Id, String> {
        match self.peek().cloned() {
            Some(Tok::Num(x)) => {
                self.i += 1;
                Ok(g.c(x))
            }
            Some(Tok::Op('(')) => {
                self.i += 1;
                let e = self.expr(g)?;
                if !self.eat(')') {
                    return Err(format!("missing `)` in `{}`", self.src));
                }
                Ok(e)
            }
            Some(Tok::Ident(name)) => {
                self.i += 1;
                if self.eat('(') {
                    let mut args = vec![self.expr(g)?];
                    while self.eat(',') {
                        args.push(self.expr(g)?);
                    }
                    if !self.eat(')') {
                        return Err(format!("missing `)` after arguments of `{name}`"));
                    }
                    if name == "pow" {
                        if args.len() != 2 {
                            return Err("pow takes 2 arguments".into());
                        }
                        return Ok(g.pow(args[0], args[1]));
                    }
                    let f = Fun::from_name(&name).ok_or_else(|| format!("unknown function `{name}`"))?;
                    if args.len() != 1 {
                        return Err(format!("`{name}` takes 1 argument"));
                    }
                    return Ok(g.fun(f, args[0]));
                }
                if let Some(k) = g.params.iter().position(|s| *s == name) {
                    return Ok(g.param(k));
                }
                if let Some(k) = g.vars.iter().position(|s| *s == name) {
                    return Ok(g.var(k));
                }
                if let Some(k) = g.fixed.iter().position(|s| *s == name) {
                    return Ok(g.fixed(k));
                }
                if name == "pi" {
                    return Ok(g.c(std::f64::consts::PI));
                }
                Err(format!("unknown name `{name}` in `{}`", self.src))
            }
            Some(t) => Err(format!("unexpected `{}` in `{}`", t.text(), self.src)),
            None => Err(format!("unexpected end of `{}`", self.src)),
        }
    }
}
