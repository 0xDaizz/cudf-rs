# cudf-rs: Unofficial Rust FFI Bindings for NVIDIA libcudf

**GPU-accelerated DataFrame operations for Rust -- near-zero `unsafe` in public API (only `DLPackTensor::from_raw_ptr`), zero-cost FFI via cxx, 61 bridge modules covering the full libcudf surface.**

[![License: MIT/Apache-2.0](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue)](LICENSE-MIT)
[![Rust 1.85+](https://img.shields.io/badge/rust-1.85+-orange?logo=rust)](https://www.rust-lang.org)
[![Platform: Linux](https://img.shields.io/badge/platform-Linux-lightgrey?logo=linux)](https://github.com)
[![CUDA 12.2+](https://img.shields.io/badge/CUDA-12.2+-76B900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![libcudf](https://img.shields.io/badge/libcudf-RAPIDS-7400B8)](https://github.com/rapidsai/cudf)

---

## Features

**Near-Zero Unsafe Public API**
All `unsafe` is confined to the internal FFI layer, with the sole exception of `DLPackTensor::from_raw_ptr`. Your application code never touches raw pointers.

**Zero-Cost FFI**
[cxx](https://cxx.rs) bridge with no serialization overhead -- C++ calls are as cheap as a function pointer indirection.

**Full libcudf Coverage**
61 bridge modules spanning compute, I/O, strings, nested types, and interop.

**Arrow Interop**
Conversion to/from `arrow-rs` via Arrow IPC. Seamless data exchange between GPU and the Rust Arrow ecosystem.

**Builder-Pattern I/O**
`ParquetReader`, `CsvReader`, `JsonReader`, `OrcReader`, `AvroReader` -- fluent APIs with compression, column selection, and header control.

**GPU String Processing**
Case conversion, find, replace, split, regex, extract, and more -- all running on the GPU.

**RAII Memory Management**
`Column` and `Table` drops free GPU memory automatically. No manual resource tracking.

## Installation

### 1. Install libcudf

**Recommended: conda**

```sh
conda create -n cudf-dev -c rapidsai -c conda-forge libcudf cuda-version=12.2
conda activate cudf-dev
```

**Alternative: from source**

Follow the [RAPIDS build guide](https://docs.rapids.ai/install). After building, set:

```sh
export CUDF_ROOT=/path/to/libcudf/prefix
# expects: $CUDF_ROOT/lib/libcudf.so and $CUDF_ROOT/include/cudf/
```

### 2. Set CUDA path (if non-standard)

```sh
export CUDA_PATH=/usr/local/cuda
```

### 3. Add cudf-rs to your project

```toml
[dependencies]
cudf = { path = "cudf" }
# or once published:
# cudf = "0.1"
```

### 4. Build

```sh
cargo build --release
```

## Quick Start

```rust,no_run
use cudf::{Column, Table, Result};
use cudf::sorting::{SortOrder, NullOrder};
use cudf::groupby::GroupBy;
use cudf::aggregation::AggregationKind;
use cudf::io::parquet;

fn main() -> Result<()> {
    // --- Create GPU columns from host data ---
    let ids    = Column::from_slice(&[1i32, 1, 2, 2, 3])?;
    let values = Column::from_slice(&[10.0f64, 20.0, 30.0, 40.0, 50.0])?;
    let table  = Table::new(vec![ids, values])?;

    // --- Sort ---
    let sorted = table.sort(
        &[SortOrder::Ascending, SortOrder::Descending],
        &[NullOrder::After,     NullOrder::After],
    )?;

    // --- GroupBy ---
    let keys   = Table::new(vec![Column::from_slice(&[1i32, 1, 2, 2, 3])?])?;
    let vals   = Table::new(vec![Column::from_slice(&[10.0f64, 20.0, 30.0, 40.0, 50.0])?])?;
    let result = GroupBy::new(&keys)
        .agg(0, AggregationKind::Sum)
        .execute(&vals)?;

    // --- Parquet I/O ---
    parquet::write_parquet(&table, "/tmp/test.parquet")?;
    let loaded = parquet::read_parquet("/tmp/test.parquet")?;
    assert_eq!(loaded.num_rows(), 5);

    // --- Read data back to host ---
    let col   = loaded.column(0)?;
    let data: Vec<i32> = col.to_vec()?;
    println!("{:?}", data);

    Ok(())
}
```

## Architecture

```
                   +----------------------------------------------+
                   |  cudf       -- safe, idiomatic Rust API      |
                   |             -- Column, Table, GroupBy, I/O   |
                   +----------------------------------------------+
                                         |
                   +----------------------------------------------+
                   |  cudf-cxx   -- cxx bridge + C++ shim layer   |
                   |             -- one bridge per libcudf module  |
                   +----------------------------------------------+
                                         |
                   +----------------------------------------------+
                   |  cudf-sys   -- links libcudf.so (build only) |
                   |             -- CUDF_ROOT / conda / pkg-config|
                   +----------------------------------------------+
                                         |
                              NVIDIA libcudf (C++)
```

Each libcudf C++ module maps to three files:

| Layer | Files | Role |
|-------|-------|------|
| C++ shim | `cudf-cxx/cpp/{include,src}/{module}_shim.{h,cpp}` | Wraps libcudf types for cxx compatibility |
| cxx bridge | `cudf-cxx/src/{module}.rs` | `#[cxx::bridge]` FFI declarations |
| Safe API | `cudf/src/{module}.rs` | Idiomatic Rust wrappers with full safety |

## Modules

### Core

| Module | Description |
|--------|-------------|
| `column` | GPU-resident column type with typed data and optional null bitmask |
| `table` | Ordered collection of columns (DataFrame equivalent) |
| `scalar` | GPU-resident single typed value with validity flag |
| `types` | `TypeId` and `DataType` mirroring libcudf's type system |
| `error` | `CudfError` and `Result<T>` -- unified error handling |

### Compute

| Module | Description |
|--------|-------------|
| `sorting` | Sort, rank, and is-sorted checks for tables and columns |
| `groupby` | Builder-pattern groupby with multi-column aggregation |
| `aggregation` | `AggregationKind` enum (Sum, Mean, Min, Max, Count, ...) |
| `reduction` | Reduce a column to a scalar (sum, product, min, max, ...) |
| `quantiles` | Quantile and percentile computation |
| `rolling` | Fixed-size rolling window aggregation |
| `binaryop` | Element-wise binary ops (arithmetic, comparison, logic) |
| `unary` | Element-wise unary ops (math functions, null/NaN checks, casts) |
| `round` | Numeric rounding (floor, ceil, half-even) |
| `transform` | NaN-to-null conversion and boolean mask generation |
| `search` | Binary search and containment checks on sorted data |
| `label_bins` | Label elements based on membership in specified bins |

### Data Manipulation

| Module | Description |
|--------|-------------|
| `copying` | Gather, scatter, slice, split, conditional copy |
| `filling` | Fill, repeat, and arithmetic sequence generation |
| `concatenate` | Vertical stacking of columns and tables |
| `merge` | Merge two pre-sorted tables into one sorted table |
| `join` | Inner, left, full outer, and cross joins |
| `stream_compaction` | Drop nulls, boolean masking, unique, distinct, duplicate removal |
| `null_mask` | Inspect and manipulate column validity bitmasks |
| `reshape` | Interleave and tile table columns |
| `transpose` | Swap rows and columns |
| `partitioning` | Hash and round-robin partitioning |
| `hashing` | Row-wise hashing (Murmur3, MD5, SHA-256, ...) |
| `datetime` | Extract year, month, day, hour, ... from timestamp columns |
| `replace` | Replace nulls, NaNs, and clamp values |
| `lists` | Explode, sort, contains, and extract on list (nested) columns |
| `structs` | Extract child columns from struct columns |
| `dictionary` | Dictionary encoding and decoding |
| `json` | JSONPath queries on JSON string columns |
| `timezone` | Timezone transition table support for ORC timestamp conversion |

### I/O

| Module | Read | Write | Notes |
|--------|------|-------|-------|
| `io::parquet` | `ParquetReader` | `ParquetWriter` | Snappy/Gzip/Zstd/LZ4 compression |
| `io::csv` | `CsvReader` | `CsvWriter` | Custom delimiter, header control |
| `io::json` | `JsonReader` | `JsonWriter` | Standard and JSON Lines |
| `io::orc` | `OrcReader` | `OrcWriter` | Column selection, compression |
| `io::avro` | `AvroReader` | -- | Read-only |

### String Operations (`strings::`)

| Module | Description |
|--------|-------------|
| `strings::case` | Upper/lower/swap case conversion (`str_to_upper`, `str_to_lower`, `str_swapcase`) |
| `strings::find` | Find substring position |
| `strings::contains` | Literal and regex containment checks |
| `strings::replace` | Literal and regex replacement |
| `strings::split` | Split by delimiter into columns or records |
| `strings::strip` | Strip (trim) leading/trailing characters |
| `strings::slice` | Substring extraction by position |
| `strings::combine` | Concatenate/join strings |
| `strings::convert` | String-to-numeric and numeric-to-string conversion |
| `strings::extract` | Regex capture group extraction |
| `strings::findall` | Regex match extraction (all occurrences) |
| `strings::like` | SQL LIKE pattern matching |
| `strings::padding` | String padding (left, right, center) |
| `strings::partition` | Split string at first/last occurrence of delimiter |
| `strings::repeat` | Repeat each string N times |
| `strings::reverse` | Reverse each string character-by-character |
| `strings::split_re` | Regex-based string splitting |
| `strings::attributes` | String character/byte count and code point extraction |
| `strings::char_types` | Character type checking (alpha, digit, etc.) |
| `strings::translate` | Character-by-character translation |
| `strings::wrap` | Word-wrap long strings by inserting newlines |

### Interop

| Module | Description |
|--------|-------------|
| `interop` | Arrow C Data Interface and IPC; DLPack tensor exchange; pack/unpack/contiguous_split |

## Prerequisites

- **GPU**: NVIDIA Volta or newer (compute capability 7.0+)
- **CUDA**: 12.2+
- **libcudf**: 24.0+ required, tested with 26.2.1; installed via conda or from source (see above). Enum discriminants are matched to specific libcudf versions -- using an untested version may produce incorrect results.
- **OS**: Linux only (libcudf does not support macOS or Windows)
- **Rust**: 1.85+ (edition 2024)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the contribution guide, including how to add new bindings.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT License ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.
