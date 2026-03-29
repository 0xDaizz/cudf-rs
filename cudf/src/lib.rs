//! # cudf -- GPU-Accelerated DataFrames for Rust
//!
//! `cudf` provides safe Rust bindings for NVIDIA's [libcudf](https://github.com/rapidsai/cudf),
//! enabling GPU-accelerated DataFrame operations with zero `unsafe` in the public API.
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use cudf::{Column, Table, Result};
//!
//! fn main() -> Result<()> {
//!     // Create a GPU column from host data
//!     let prices = Column::from_slice(&[100.0f64, 200.5, 300.75, 150.25])?;
//!     let quantities = Column::from_slice(&[10i32, 20, 30, 15])?;
//!
//!     // Inspect
//!     assert_eq!(prices.len(), 4);
//!     assert!(!prices.has_nulls());
//!
//!     // Read back to host
//!     let host_prices: Vec<f64> = prices.to_vec()?;
//!     assert_eq!(host_prices, vec![100.0, 200.5, 300.75, 150.25]);
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Architecture
//!
//! This crate is the **safe public API** layer in a 3-crate stack:
//!
//! ```text
//! +---------------------------------------------+
//! |  cudf (this crate) -- 100% safe Rust API    |
//! +---------------------------------------------+
//! |  cudf-cxx -- cxx bridge + C++ shim layer    |
//! +---------------------------------------------+
//! |  cudf-sys -- links libcudf.so               |
//! +---------------------------------------------+
//! ```
//!
//! ## Feature Flags
//!
//! | Feature | Default | Description |
//! |---------|---------|-------------|
//! | `arrow-interop` | Yes | Zero-copy conversion to/from `arrow` arrays |

pub mod error;
pub mod types;
pub mod column;
pub mod table;
pub mod scalar;
pub mod sorting;
pub mod copying;
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

// Re-exports for convenience
pub use error::{CudfError, Result};
pub use types::{DataType, TypeId};
pub use column::{Column, CudfType};
pub use table::Table;
pub use scalar::Scalar;
pub use sorting::{SortOrder, NullOrder};
pub use copying::OutOfBoundsPolicy;
pub use unary::UnaryOp;
pub use binaryop::BinaryOp;
pub use stream_compaction::DuplicateKeepOption;
pub use aggregation::AggregationKind;
pub use groupby::GroupBy;
pub use reduction::{ReduceOp, ScanOp};
pub use quantiles::Interpolation;
pub use rolling::RollingAgg;
pub use join::JoinResult;
