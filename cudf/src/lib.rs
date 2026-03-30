//! # cudf -- GPU-Accelerated DataFrames for Rust
//!
//! `cudf` provides safe Rust bindings for NVIDIA's [libcudf](https://github.com/rapidsai/cudf),
//! enabling GPU-accelerated DataFrame operations. The public API is fully safe, with the
//! sole exception of `DLPackTensor::from_raw_ptr` which requires an unsafe block.
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
//! |  cudf (this crate) -- safe Rust API          |
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

pub mod aggregation;
pub mod binaryop;
pub mod column;
pub mod concatenate;
pub mod copying;
pub mod datetime;
pub mod dictionary;
pub mod error;
pub mod filling;
pub mod groupby;
pub mod hashing;
pub mod interop;
pub mod io;
pub mod join;
pub mod json;
pub mod label_bins;
pub mod lists;
pub mod merge;
pub mod null_mask;
pub mod partitioning;
pub mod quantiles;
pub mod reduction;
pub mod replace;
pub mod reshape;
pub mod rolling;
pub mod round;
pub mod scalar;
pub mod search;
pub mod sorting;
pub mod stream_compaction;
pub mod strings;
pub mod structs;
pub mod table;
pub mod timezone;
pub mod transform;
pub mod transpose;
pub mod types;
pub mod unary;

// Re-exports for convenience
pub use aggregation::AggregationKind;
pub use binaryop::BinaryOp;
pub use column::{Column, CudfType};
pub use copying::OutOfBoundsPolicy;
pub use error::{CudfError, Result};
pub use groupby::{GroupBy, GroupByGroups, GroupByReplacePolicy, GroupByScan, GroupByScanOp};
pub use join::{HashJoin, JoinResult, SemiJoinResult};
pub use json::JsonObjectOptions;
pub use partitioning::PartitionResult;
pub use quantiles::Interpolation;
pub use reduction::{MinMaxResult, ReduceOp, ScanOp};
pub use replace::NullReplacePolicy;
pub use rolling::RollingAgg;
pub use scalar::Scalar;
pub use sorting::{NullOrder, SortOrder};
pub use stream_compaction::DuplicateKeepOption;
pub use table::{Table, TableWithMetadata};
pub use types::{DataType, NullHandling, TypeId};
pub use unary::UnaryOp;

// Interop re-exports
pub use interop::{DLPackTensor, PackedTable, SplitResult};
