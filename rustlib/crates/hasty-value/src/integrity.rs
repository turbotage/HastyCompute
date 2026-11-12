//! Optional end-to-end integrity digest of a stream / file. Off (the default) it costs nothing: no hasher exists, no
//! byte is touched, nothing is appended. On, the encoder feeds every byte it emits (1 MiB at a time) to an
//! accumulating hasher and appends the digest after the value; the stream header's flags byte holds the algorithm, so
//! decoders know to expect it. The digest covers the header and the value (not itself) — it detects truncation,
//! corruption, mis-assembled chunks and bit rot between the writer's memory and the reader's, independent of the
//! per-block CRC (which stays available for random-access reads).
//!
//! Not authentication: anyone able to alter the stream can recompute an unkeyed digest.

use crate::{Error, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Integrity {
    #[default]
    None,
    /// XXH3-128: ~10 GB/s per core, non-cryptographic (accidental corruption)
    Xxh3,
    /// BLAKE3-256: cryptographic, ~1 GB/s per core (multi-threaded for large inputs)
    Blake3,
}

impl Integrity {
    pub(crate) fn code(self) -> u8 {
        self as u8
    }

    pub(crate) fn from_code(c: u8) -> Result<Self> {
        match c {
            0 => Ok(Self::None),
            1 => Ok(Self::Xxh3),
            2 => Ok(Self::Blake3),
            c => Err(Error::Format(format!("unsupported stream flags {c} (integrity algorithm)"))),
        }
    }

    pub fn digest_len(self) -> usize {
        match self {
            Self::None => 0,
            Self::Xxh3 => 16,
            Self::Blake3 => 32,
        }
    }

    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "" | "none" => Ok(Self::None),
            "xxh3" => Ok(Self::Xxh3),
            "blake3" => Ok(Self::Blake3),
            _ => Err(Error::Codec(format!("unknown integrity algorithm {name}"))),
        }
    }
}

pub(crate) enum Hasher {
    Xxh3(Box<xxhash_rust::xxh3::Xxh3>),
    Blake3(Box<blake3::Hasher>),
}

impl Hasher {
    pub fn new(a: Integrity) -> Option<Self> {
        match a {
            Integrity::None => None,
            Integrity::Xxh3 => Some(Self::Xxh3(Box::new(xxhash_rust::xxh3::Xxh3::new()))),
            Integrity::Blake3 => Some(Self::Blake3(Box::new(blake3::Hasher::new()))),
        }
    }

    pub fn update(&mut self, b: &[u8]) {
        match self {
            Self::Xxh3(h) => h.update(b),
            Self::Blake3(h) => {
                if b.len() >= 1 << 17 {
                    h.update_rayon(b);
                } else {
                    h.update(b);
                }
            }
        }
    }

    pub fn finalize(self) -> Vec<u8> {
        match self {
            Self::Xxh3(h) => h.digest128().to_le_bytes().to_vec(),
            Self::Blake3(h) => h.finalize().as_bytes().to_vec(),
        }
    }
}
