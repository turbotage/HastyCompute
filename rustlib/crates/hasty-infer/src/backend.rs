//! Inference runtime abstraction. A backend maps one float32 tensor to another (C-order); recipes own everything
//! around that (tiling, normalisation, aggregation). Runtimes (ONNX Runtime, libtorch) implement [`Backend`].

use crate::Result;

#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Self {
        assert_eq!(shape.iter().product::<usize>(), data.len());
        Self { shape, data }
    }
}

pub trait Backend: Send {
    /// short runtime name for messages ("onnx", "torch", "mock")
    fn name(&self) -> &str;
    /// run the network of `role` (`"main"` for single-network models) on `input`
    fn forward(&mut self, role: &str, input: &Tensor) -> Result<Tensor>;
}

/// Test backend: applies a closure per role.
pub struct Mock<F: FnMut(&str, &Tensor) -> Tensor + Send>(pub F);

impl<F: FnMut(&str, &Tensor) -> Tensor + Send> Backend for Mock<F> {
    fn name(&self) -> &str {
        "mock"
    }
    fn forward(&mut self, role: &str, input: &Tensor) -> Result<Tensor> {
        Ok((self.0)(role, input))
    }
}
