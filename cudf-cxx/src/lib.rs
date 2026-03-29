//! # cudf-cxx
//!
//! Low-level cxx bridge between Rust and libcudf's C++ API.
//!
//! This crate provides the raw FFI layer. Users should prefer the safe
//! [`cudf`](../cudf/index.html) crate instead.
//!
//! ## Architecture
//!
//! Each libcudf module has a corresponding Rust bridge file and C++ shim:
//!
//! | Rust bridge | C++ shim | libcudf module |
//! |-------------|----------|----------------|
//! | `types.rs` | `types_shim.h/cpp` | `cudf/types.hpp` |
//! | `column.rs` | `column_shim.h/cpp` | `cudf/column/` |
//! | `table.rs` | `table_shim.h/cpp` | `cudf/table/` |

pub mod types;
pub mod column;
pub mod table;
pub mod sorting;
pub mod copying;
pub mod scalar;
pub mod unary;
pub mod binaryop;
pub mod stream_compaction;
pub mod filling;
pub mod concatenate;
pub mod null_mask;
pub mod aggregation;
pub mod groupby;
pub mod reduction;
pub mod quantiles;
pub mod rolling;
pub mod io;
pub mod join;
pub mod strings;
pub mod interop;
pub mod hashing;
pub mod datetime;
pub mod round;
pub mod transform;
pub mod reshape;
pub mod transpose;
pub mod partitioning;
pub mod merge;
pub mod search;
