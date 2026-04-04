//! Build script for cudf-cxx.
//!
//! Compiles C++ shim files and generates cxx bridge code.
//! Include paths are inherited from cudf-sys via DEP_CUDF_INCLUDE.

use std::env;
use std::path::PathBuf;

fn main() {
    // docs.rs doesn't have libcudf/CUDA installed; skip C++ compilation
    if std::env::var("DOCS_RS").is_ok() {
        return;
    }

    // Get include path from cudf-sys
    let cudf_include = env::var("DEP_CUDF_INCLUDE").unwrap_or_else(|_| {
        // Fallback: try CONDA_PREFIX
        env::var("CONDA_PREFIX")
            .map(|p| format!("{}/include", p))
            .unwrap_or_else(|_| {
                // Check if /usr/local/include/cudf exists as a last resort
                let fallback = "/usr/local/include";
                let cudf_header = std::path::Path::new(fallback)
                    .join("cudf")
                    .join("types.hpp");
                if !cudf_header.exists() {
                    panic!(
                        "Cannot find cudf headers. Set CUDF_ROOT, CONDA_PREFIX, \
                     or ensure cudf headers are in /usr/local/include"
                    );
                }
                eprintln!("cargo:warning=Using fallback cudf headers from /usr/local/include");
                fallback.to_string()
            })
    });

    let cudf_include_path = PathBuf::from(&cudf_include);

    // Auto-discover sibling include paths (librmm, rapids_logger, pyarrow, etc.)
    // When CUDF_ROOT points to e.g. .../site-packages/libcudf, the parent
    // directory may contain librmm/include, rapids_logger/include, etc.
    let mut extra_includes: Vec<PathBuf> = Vec::new();
    if let Some(parent) = cudf_include_path.parent().and_then(|p| p.parent()) {
        // parent is e.g. .../site-packages/libcudf -> parent.parent() = .../site-packages
        let site_packages = parent;
        // libcudf/include/rapids/ contains CCCL headers (cuda/, cub/, thrust/)
        let rapids_inc = site_packages.join("libcudf").join("include").join("rapids");
        if rapids_inc.exists() {
            extra_includes.push(rapids_inc);
        }
        // nvidia/cuda_cccl also has CCCL headers as fallback
        let cccl_inc = site_packages
            .join("nvidia")
            .join("cuda_cccl")
            .join("include");
        if cccl_inc.exists() {
            extra_includes.push(cccl_inc);
        }
        for sibling in &["librmm", "rapids_logger", "pyarrow"] {
            let inc = site_packages.join(sibling).join("include");
            if inc.exists() {
                extra_includes.push(inc);
            }
        }

        // Link against Arrow C++ (for IPC serialization in interop shim).
        // Arrow shared libraries live inside pyarrow's package directory.
        let pyarrow_lib = site_packages.join("pyarrow");
        if pyarrow_lib.exists() {
            println!("cargo:rustc-link-search=native={}", pyarrow_lib.display());
            println!("cargo:rustc-link-lib=dylib=arrow");
        }
    }

    // Fallback: PYARROW_DIR env var for Arrow include/lib discovery
    // when pyarrow is installed in a different site-packages than libcudf.
    if let Ok(pyarrow_dir) = env::var("PYARROW_DIR") {
        let pyarrow_path = PathBuf::from(&pyarrow_dir);
        let inc = pyarrow_path.join("include");
        if inc.exists() {
            extra_includes.push(inc);
        }
        if pyarrow_path.exists() {
            println!("cargo:rustc-link-search=native={}", pyarrow_path.display());
            println!("cargo:rustc-link-lib=dylib=arrow");
        }
    }
    println!("cargo:rerun-if-env-changed=PYARROW_DIR");

    // For conda environments, CCCL headers are at $CONDA_PREFIX/include/rapids/
    let rapids_conda = cudf_include_path.join("rapids");
    if rapids_conda.exists() {
        extra_includes.push(rapids_conda);
    }

    // Also check CUDA include path
    if let Ok(cuda_path) = env::var("CUDA_PATH") {
        let cuda_inc = PathBuf::from(&cuda_path).join("include");
        if cuda_inc.exists() {
            extra_includes.push(cuda_inc);
        }
    }

    let cpp_dir = PathBuf::from("cpp");
    let cpp_include = cpp_dir.join("include");
    let cpp_src = cpp_dir.join("src");

    // Build cxx bridges with C++ shim sources
    let mut build = cxx_build::bridges([
        "src/types.rs",
        "src/column.rs",
        "src/table.rs",
        "src/sorting.rs",
        "src/copying.rs",
        "src/scalar.rs",
        "src/unary.rs",
        "src/binaryop.rs",
        "src/stream_compaction.rs",
        "src/filling.rs",
        "src/concatenate.rs",
        "src/null_mask.rs",
        "src/aggregation.rs",
        "src/groupby.rs",
        "src/reduction.rs",
        "src/quantiles.rs",
        "src/rolling.rs",
        "src/io/parquet.rs",
        "src/io/csv.rs",
        "src/io/json.rs",
        "src/io/orc.rs",
        "src/io/avro.rs",
        "src/join.rs",
        "src/json.rs",
        "src/label_bins.rs",
        "src/timezone.rs",
        "src/strings/case.rs",
        "src/strings/find.rs",
        "src/strings/contains.rs",
        "src/strings/replace.rs",
        "src/strings/split.rs",
        "src/strings/split_re.rs",
        "src/strings/partition.rs",
        "src/strings/strip.rs",
        "src/strings/convert.rs",
        "src/strings/combine.rs",
        "src/strings/slice.rs",
        "src/strings/extract.rs",
        "src/strings/padding.rs",
        "src/strings/repeat.rs",
        "src/strings/findall.rs",
        "src/strings/attributes.rs",
        "src/strings/translate.rs",
        "src/strings/reverse.rs",
        "src/strings/wrap.rs",
        "src/strings/char_types.rs",
        "src/strings/like.rs",
        "src/interop.rs",
        "src/hashing.rs",
        "src/datetime.rs",
        "src/round.rs",
        "src/transform.rs",
        "src/reshape.rs",
        "src/transpose.rs",
        "src/partitioning.rs",
        "src/merge.rs",
        "src/search.rs",
        "src/lists/ops.rs",
        "src/structs.rs",
        "src/dictionary.rs",
        "src/replace.rs",
    ]);
    build
        .file(cpp_src.join("types_shim.cpp"))
        .file(cpp_src.join("column_shim.cpp"))
        .file(cpp_src.join("table_shim.cpp"))
        .file(cpp_src.join("sorting_shim.cpp"))
        .file(cpp_src.join("copying_shim.cpp"))
        .file(cpp_src.join("scalar_shim.cpp"))
        .file(cpp_src.join("unary_shim.cpp"))
        .file(cpp_src.join("binaryop_shim.cpp"))
        .file(cpp_src.join("stream_compaction_shim.cpp"))
        .file(cpp_src.join("filling_shim.cpp"))
        .file(cpp_src.join("concatenate_shim.cpp"))
        .file(cpp_src.join("null_mask_shim.cpp"))
        .file(cpp_src.join("aggregation_shim.cpp"))
        .file(cpp_src.join("groupby_shim.cpp"))
        .file(cpp_src.join("reduction_shim.cpp"))
        .file(cpp_src.join("quantiles_shim.cpp"))
        .file(cpp_src.join("rolling_shim.cpp"))
        .file(cpp_src.join("io/parquet_shim.cpp"))
        .file(cpp_src.join("io/csv_shim.cpp"))
        .file(cpp_src.join("io/json_shim.cpp"))
        .file(cpp_src.join("io/orc_shim.cpp"))
        .file(cpp_src.join("io/avro_shim.cpp"))
        .file(cpp_src.join("join_shim.cpp"))
        .file(cpp_src.join("json_shim.cpp"))
        .file(cpp_src.join("label_bins_shim.cpp"))
        .file(cpp_src.join("timezone_shim.cpp"))
        .file(cpp_src.join("strings/case_shim.cpp"))
        .file(cpp_src.join("strings/find_shim.cpp"))
        .file(cpp_src.join("strings/contains_shim.cpp"))
        .file(cpp_src.join("strings/replace_shim.cpp"))
        .file(cpp_src.join("strings/split_shim.cpp"))
        .file(cpp_src.join("strings/split_re_shim.cpp"))
        .file(cpp_src.join("strings/partition_shim.cpp"))
        .file(cpp_src.join("strings/strip_shim.cpp"))
        .file(cpp_src.join("strings/convert_shim.cpp"))
        .file(cpp_src.join("strings/combine_shim.cpp"))
        .file(cpp_src.join("strings/slice_shim.cpp"))
        .file(cpp_src.join("strings/extract_shim.cpp"))
        .file(cpp_src.join("strings/padding_shim.cpp"))
        .file(cpp_src.join("strings/repeat_shim.cpp"))
        .file(cpp_src.join("strings/findall_shim.cpp"))
        .file(cpp_src.join("strings/attributes_shim.cpp"))
        .file(cpp_src.join("strings/translate_shim.cpp"))
        .file(cpp_src.join("strings/reverse_shim.cpp"))
        .file(cpp_src.join("strings/wrap_shim.cpp"))
        .file(cpp_src.join("strings/char_types_shim.cpp"))
        .file(cpp_src.join("strings/like_shim.cpp"))
        .file(cpp_src.join("interop_shim.cpp"))
        .file(cpp_src.join("hashing_shim.cpp"))
        .file(cpp_src.join("datetime_shim.cpp"))
        .file(cpp_src.join("round_shim.cpp"))
        .file(cpp_src.join("transform_shim.cpp"))
        .file(cpp_src.join("reshape_shim.cpp"))
        .file(cpp_src.join("transpose_shim.cpp"))
        .file(cpp_src.join("partitioning_shim.cpp"))
        .file(cpp_src.join("merge_shim.cpp"))
        .file(cpp_src.join("search_shim.cpp"))
        .file(cpp_src.join("lists/lists_shim.cpp"))
        .file(cpp_src.join("structs_shim.cpp"))
        .file(cpp_src.join("dictionary_shim.cpp"))
        .file(cpp_src.join("replace_shim.cpp"))
        .include(&cpp_include)
        .include(&cudf_include);

    for inc in &extra_includes {
        build.include(inc);
    }

    build
        .std("c++20")
        .define("LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE", None)
        // Force-include default_stream.hpp so all shims have access to cudf::get_default_stream()
        .flag("-include")
        .flag("cudf/utilities/default_stream.hpp")
        .flag_if_supported("-Wno-unused-parameter")
        .flag_if_supported("-Wno-missing-field-initializers")
        .compile("cudf_cxx");

    // Explicitly link native libraries.
    // cudf-sys declares these via `links = "cudf"`, but cargo may not propagate
    // -l flags if the sys crate has no Rust symbols. We re-emit them here.
    println!("cargo:rustc-link-lib=dylib=cudf");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=rmm");

    // For conda environments, link Arrow from CONDA_PREFIX/lib
    if let Ok(conda_prefix) = env::var("CONDA_PREFIX") {
        let conda_lib = PathBuf::from(&conda_prefix).join("lib");
        if conda_lib.join("libarrow.so").exists() || conda_lib.join("libarrow.dylib").exists() {
            println!("cargo:rustc-link-search=native={}", conda_lib.display());
            println!("cargo:rustc-link-lib=dylib=arrow");
        }
    }

    // Emit rerun-if-changed for each source file individually,
    // because cargo only checks directory mtime, not contents.
    for dir in &["cpp/include", "cpp/src"] {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_file() {
                    println!("cargo:rerun-if-changed={}", path.display());
                } else if path.is_dir() {
                    // Handle subdirectories (e.g., cpp/include/io/, cpp/src/strings/)
                    if let Ok(sub) = std::fs::read_dir(&path) {
                        for sub_entry in sub.flatten() {
                            if sub_entry.path().is_file() {
                                println!("cargo:rerun-if-changed={}", sub_entry.path().display());
                            }
                        }
                    }
                }
            }
        }
    }
    // Also watch Rust bridge sources
    if let Ok(entries) = std::fs::read_dir("src") {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() {
                println!("cargo:rerun-if-changed={}", path.display());
            } else if path.is_dir() {
                if let Ok(sub) = std::fs::read_dir(&path) {
                    for sub_entry in sub.flatten() {
                        if sub_entry.path().is_file() {
                            println!("cargo:rerun-if-changed={}", sub_entry.path().display());
                        }
                    }
                }
            }
        }
    }
}
