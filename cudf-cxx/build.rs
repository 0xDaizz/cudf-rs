//! Build script for cudf-cxx.
//!
//! Compiles C++ shim files and generates cxx bridge code.
//! Include paths are inherited from cudf-sys via DEP_CUDF_INCLUDE.

use std::env;
use std::path::PathBuf;

fn main() {
    // Get include path from cudf-sys
    let cudf_include = env::var("DEP_CUDF_INCLUDE")
        .unwrap_or_else(|_| {
            // Fallback: try CONDA_PREFIX
            env::var("CONDA_PREFIX")
                .map(|p| format!("{}/include", p))
                .unwrap_or_else(|_| "/usr/local/include".to_string())
        });

    let cpp_dir = PathBuf::from("cpp");
    let cpp_include = cpp_dir.join("include");
    let cpp_src = cpp_dir.join("src");

    // Build cxx bridges with C++ shim sources
    cxx_build::bridges(&[
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
        "src/strings/case.rs",
        "src/strings/find.rs",
        "src/strings/contains.rs",
        "src/strings/replace.rs",
        "src/strings/split.rs",
        "src/strings/strip.rs",
        "src/strings/convert.rs",
        "src/strings/combine.rs",
        "src/strings/slice.rs",
        "src/strings/extract.rs",
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
    ])
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
    .file(cpp_src.join("strings/case_shim.cpp"))
    .file(cpp_src.join("strings/find_shim.cpp"))
    .file(cpp_src.join("strings/contains_shim.cpp"))
    .file(cpp_src.join("strings/replace_shim.cpp"))
    .file(cpp_src.join("strings/split_shim.cpp"))
    .file(cpp_src.join("strings/strip_shim.cpp"))
    .file(cpp_src.join("strings/convert_shim.cpp"))
    .file(cpp_src.join("strings/combine_shim.cpp"))
    .file(cpp_src.join("strings/slice_shim.cpp"))
    .file(cpp_src.join("strings/extract_shim.cpp"))
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
    .include(&cpp_include)
    .include(&cudf_include)
    .std("c++17")
    .flag_if_supported("-Wno-unused-parameter")
    .flag_if_supported("-Wno-missing-field-initializers")
    .compile("cudf_cxx");

    println!("cargo:rerun-if-changed=cpp/");
    println!("cargo:rerun-if-changed=src/");
}
