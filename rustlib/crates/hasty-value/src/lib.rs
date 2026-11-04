//! Values exchanged between hasty binaries, the hasty server, Python and the browser: a tree of tensors, lists,
//! tuples, dicts, strings and scalars ([`Value`]), with a streaming, block-compressed binary format ([`format`],
//! specification in `FORMAT.md`), a slice-path language (`["key"][0][1:5, :, ..., 3]`, [`slice`]) and u16
//! quantisation for compact display transfer ([`quant`]).
//!
//! Tensors are host-side, contiguous, C-order (last index fastest), little-endian. hasty's x-fastest volumes map to
//! C-order shape `[Z, Y, X]` ([`Tensor::from_fortran`]).

pub mod codec;
mod dtype;
pub mod format;
pub mod lazy;
pub mod npy;
pub mod quant;
pub mod slice;
mod tensor;
mod value;

pub use codec::{Codec, Options};
pub use dtype::{DType, Element};
pub use format::{decode, encode, encode_to_vec, encode_with, read, write, Decoder};
pub use lazy::{Lazy, LazyFile, TensorRef};
pub use tensor::{Buffer, Tensor};
pub use value::Value;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("format: {0}")]
    Format(String),
    #[error("type: {0}")]
    Type(String),
    #[error("slice: {0}")]
    Slice(String),
    #[error("codec: {0}")]
    Codec(String),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
