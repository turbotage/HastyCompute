//! Symbolic models: one or more real outputs m_k(θ; v, fx) per data point (a complex signal = two outputs, real and
//! imaginary part), with their symbolic Jacobian ∂m_k/∂θ_j and Hessian ∂²m_k/∂θ_i∂θ_j (upper triangle, packed
//! `hidx(i, j) = j (j + 1) / 2 + i`, i ≤ j). The least-squares Hessian is then
//!   H = Σ_points Σ_k w (J_kᵀ J_k + r_k ∇² m_k),   r_k = m_k − y_k.

use std::collections::HashMap;

use crate::expr::{Graph, Id};
use crate::program::Program;

/// packed index of (i, j), i ≤ j
#[inline]
pub fn hidx(i: usize, j: usize) -> usize {
    let (i, j) = if i <= j { (i, j) } else { (j, i) };
    j * (j + 1) / 2 + i
}

#[derive(Clone, Debug)]
pub struct Model {
    pub graph: Graph,
    pub outputs: Vec<Id>,
    /// [output][param]
    pub jac: Vec<Vec<Id>>,
    /// [output][packed (i, j)]
    pub hes: Vec<Vec<Id>>,
    /// m only
    pub prog_val: Program,
    /// m, then J (output-major)
    pub prog_gn: Program,
    /// m, J, then packed second derivatives (output-major)
    pub prog_full: Program,
}

impl Model {
    /// `outputs`: expressions over the names in `params` (fitted), `vars` (per data point, shared by all problems)
    /// and `fixed` (per problem)
    pub fn new(params: &[&str], vars: &[&str], fixed: &[&str], outputs: &[&str]) -> Result<Model, String> {
        let mut g = Graph::new(params, vars, fixed);
        let outs: Vec<Id> = outputs.iter().map(|s| g.parse(s)).collect::<Result<_, _>>()?;
        Ok(Model::from_graph(g, outs))
    }

    pub fn from_graph(mut g: Graph, outs: Vec<Id>) -> Model {
        let np = g.params.len();
        let mut jac = vec![vec![0; np]; outs.len()];
        let mut hes = vec![vec![0; np * (np + 1) / 2]; outs.len()];
        // one memo per parameter, shared by all outputs and by the second derivatives (maximum sharing)
        let mut memo: Vec<HashMap<Id, Id>> = vec![HashMap::new(); np];
        for (k, &o) in outs.iter().enumerate() {
            for j in 0..np {
                jac[k][j] = g.diff(o, j, &mut memo[j]);
            }
            for j in 0..np {
                for i in 0..=j {
                    let dj = jac[k][j];
                    hes[k][hidx(i, j)] = g.diff(dj, i, &mut memo[i]);
                }
            }
        }
        let gn: Vec<Id> = outs.iter().copied().chain(jac.iter().flatten().copied()).collect();
        let full: Vec<Id> = gn.iter().copied().chain(hes.iter().flatten().copied()).collect();
        Model {
            prog_val: Program::new(&g, &outs),
            prog_gn: Program::new(&g, &gn),
            prog_full: Program::new(&g, &full),
            graph: g,
            outputs: outs,
            jac,
            hes,
        }
    }
    pub fn n_param(&self) -> usize {
        self.graph.params.len()
    }
    pub fn n_var(&self) -> usize {
        self.graph.vars.len()
    }
    pub fn n_fixed(&self) -> usize {
        self.graph.fixed.len()
    }
    pub fn n_out(&self) -> usize {
        self.outputs.len()
    }
    pub fn n_hes(&self) -> usize {
        let n = self.n_param();
        n * (n + 1) / 2
    }
}
