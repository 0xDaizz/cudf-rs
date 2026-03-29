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

pub mod aggregation;
pub mod binaryop;
pub mod column;
pub mod concatenate;
pub mod copying;
pub mod datetime;
pub mod filling;
pub mod groupby;
pub mod hashing;
pub mod interop;
pub mod io;
pub mod join;
pub mod merge;
pub mod null_mask;
pub mod partitioning;
pub mod quantiles;
pub mod reduction;
pub mod reshape;
pub mod rolling;
pub mod round;
pub mod scalar;
pub mod search;
pub mod sorting;
pub mod stream_compaction;
pub mod strings;
pub mod table;
pub mod transform;
pub mod transpose;
pub mod types;
pub mod unary;
