//! String operations on GPU string columns.
//!
//! This module provides GPU-accelerated string manipulation functions
//! including case conversion, search, pattern matching, replacement,
//! splitting, stripping, type conversion, concatenation, slicing,
//! and regex extraction.
//!
//! All operations work on string-type [`Column`](crate::Column)s and
//! return new columns (or tables) without modifying the input.

pub mod attributes;
pub mod case;
pub mod char_types;
pub mod combine;
pub mod contains;
pub mod convert;
pub mod extract;
pub mod find;
pub mod findall;
pub mod like;
pub mod padding;
pub mod repeat;
pub mod replace;
pub mod reverse;
pub mod slice;
pub mod split;
pub mod strip;
pub mod translate;
pub mod wrap;
