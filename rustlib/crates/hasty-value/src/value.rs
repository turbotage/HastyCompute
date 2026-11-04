use crate::Tensor;

/// A value tree. Dicts keep insertion order.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Tensor(Tensor),
    List(Vec<Value>),
    Tuple(Vec<Value>),
    Dict(Vec<(String, Value)>),
}

impl Value {
    pub fn kind(&self) -> &'static str {
        match self {
            Value::None => "none",
            Value::Bool(_) => "bool",
            Value::Int(_) => "int",
            Value::Float(_) => "float",
            Value::Str(_) => "str",
            Value::Tensor(_) => "tensor",
            Value::List(_) => "list",
            Value::Tuple(_) => "tuple",
            Value::Dict(_) => "dict",
        }
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Value::Dict(d) => d.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn get_mut(&mut self, key: &str) -> Option<&mut Value> {
        match self {
            Value::Dict(d) => d.iter_mut().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn as_tensor(&self) -> Option<&Tensor> {
        match self {
            Value::Tensor(t) => Some(t),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }

    /// total payload bytes (tensor data + strings)
    pub fn nbytes(&self) -> usize {
        match self {
            Value::Tensor(t) => t.nbytes(),
            Value::Str(s) => s.len(),
            Value::List(v) | Value::Tuple(v) => v.iter().map(Value::nbytes).sum(),
            Value::Dict(d) => d.iter().map(|(k, v)| k.len() + v.nbytes()).sum(),
            _ => 8,
        }
    }

    /// JSON description of the tree: types, tensor shapes / dtypes, scalars; no tensor data.
    pub fn summary(&self) -> serde_json::Value {
        use serde_json::json;
        match self {
            Value::None => json!(null),
            Value::Bool(b) => json!(b),
            Value::Int(i) => json!(i),
            Value::Float(f) => json!(f),
            Value::Str(s) if s.len() <= 256 => json!(s),
            Value::Str(s) => json!({ "type": "str", "len": s.len() }),
            Value::Tensor(t) => json!({ "type": "tensor", "dtype": t.dtype().name(), "shape": t.shape(), "nbytes": t.nbytes() }),
            Value::List(v) => json!({ "type": "list", "items": v.iter().map(Value::summary).collect::<Vec<_>>() }),
            Value::Tuple(v) => json!({ "type": "tuple", "items": v.iter().map(Value::summary).collect::<Vec<_>>() }),
            Value::Dict(d) => {
                let m: serde_json::Map<String, serde_json::Value> = d.iter().map(|(k, v)| (k.clone(), v.summary())).collect();
                json!({ "type": "dict", "items": m })
            }
        }
    }
}

impl From<Tensor> for Value {
    fn from(t: Tensor) -> Self {
        Value::Tensor(t)
    }
}
impl From<String> for Value {
    fn from(s: String) -> Self {
        Value::Str(s)
    }
}
impl From<&str> for Value {
    fn from(s: &str) -> Self {
        Value::Str(s.to_string())
    }
}
impl From<i64> for Value {
    fn from(i: i64) -> Self {
        Value::Int(i)
    }
}
impl From<f64> for Value {
    fn from(f: f64) -> Self {
        Value::Float(f)
    }
}
impl From<bool> for Value {
    fn from(b: bool) -> Self {
        Value::Bool(b)
    }
}
impl From<Vec<Value>> for Value {
    fn from(v: Vec<Value>) -> Self {
        Value::List(v)
    }
}
