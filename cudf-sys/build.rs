//! Build script for cudf-sys.
//!
//! Locates and links against the pre-built libcudf shared library.
//! Supports two discovery methods:
//!
//! 1. `CUDF_ROOT` environment variable (highest priority)
//! 2. `CONDA_PREFIX` environment variable (conda environment)
//! 3. pkg-config fallback
//!
//! # Required Libraries
//! - libcudf.so (main library)
//! - libcudart.so (CUDA runtime)

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=CUDF_ROOT");
    println!("cargo:rerun-if-env-changed=CONDA_PREFIX");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");

    if let Ok(root) = env::var("CUDF_ROOT") {
        link_from_prefix(&root);
        return;
    }

    if let Ok(prefix) = env::var("CONDA_PREFIX") {
        link_from_prefix(&prefix);
        return;
    }

    // Fallback: try pkg-config
    if pkg_config::Config::new()
        .atleast_version("24.0")
        .probe("cudf")
        .is_ok()
    {
        return;
    }

    eprintln!(
        "error: Could not find libcudf. Please set one of:\n\
         - CUDF_ROOT: path to libcudf installation prefix\n\
         - CONDA_PREFIX: path to conda environment with libcudf installed\n\
         - Ensure pkg-config can find cudf"
    );
    std::process::exit(1);
}

fn link_from_prefix(prefix: &str) {
    let prefix = PathBuf::from(prefix);
    let lib_dir = prefix.join("lib");
    let lib64_dir = prefix.join("lib64");
    let include_dir = prefix.join("include");

    // Library search paths
    if lib_dir.exists() {
        println!("cargo:rustc-link-search=native={}", lib_dir.display());
    }
    if lib64_dir.exists() {
        println!("cargo:rustc-link-search=native={}", lib64_dir.display());
    }

    // Link libraries
    println!("cargo:rustc-link-lib=dylib=cudf");

    // CUDA runtime
    if let Ok(cuda_path) = env::var("CUDA_PATH") {
        let cuda_lib = PathBuf::from(&cuda_path).join("lib64");
        if cuda_lib.exists() {
            println!("cargo:rustc-link-search=native={}", cuda_lib.display());
        }
    }
    println!("cargo:rustc-link-lib=dylib=cudart");

    // Export include path for downstream crates
    if include_dir.exists() {
        println!("cargo:include={}", include_dir.display());
    }

    // Export root for downstream
    println!("cargo:root={}", prefix.display());
}
