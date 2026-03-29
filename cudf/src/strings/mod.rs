//! String operations on GPU string columns.
//!
//! This module provides GPU-accelerated string manipulation functions
//! including case conversion, search, pattern matching, replacement,
//! splitting, stripping, type conversion, concatenation, slicing,
//! and regex extraction.
//!
//! All operations work on string-type [`Column`](crate::Column)s and
//! return new columns (or tables) without modifying the input.

pub mod case;
pub mod find;
pub mod contains;
pub mod replace;
pub mod split;
pub mod strip;
pub mod convert;
pub mod combine;
pub mod slice;
pub mod extract;
