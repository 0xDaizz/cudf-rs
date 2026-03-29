//! # cudf-sys
//!
//! Native build crate for linking against NVIDIA's libcudf.
//!
//! This crate contains no Rust code — it exists solely to trigger the build
//! script that locates and links libcudf. Downstream crates (`cudf-cxx`)
//! depend on this to inherit the correct linker flags.
//!
//! ## Installation
//!
//! libcudf must be installed on your system. The recommended method is via conda:
//!
//! ```sh
//! conda install -c rapidsai -c conda-forge libcudf cuda-version=12.2
//! ```
//!
//! Alternatively, set `CUDF_ROOT` to point to a manual installation prefix.

// This crate intentionally contains no code.
// The build.rs script handles all linking configuration.
