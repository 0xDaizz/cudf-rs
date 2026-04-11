# cudf-rs

[![License](https://img.shields.io/badge/license-Apache--2.0%20OR%20MIT-blue)](LICENSE-APACHE)
[![Crates.io](https://img.shields.io/crates/v/cudf.svg)](https://crates.io/crates/cudf)
[![docs.rs](https://img.shields.io/docsrs/cudf)](https://docs.rs/cudf)
[![Rust 1.85+](https://img.shields.io/badge/rust-1.85+-orange?logo=rust)](https://www.rust-lang.org)
[![CUDA 12.2+](https://img.shields.io/badge/CUDA-12.2+-76B900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/0xDaizz/cudf-rs)

Unofficial Rust bindings for NVIDIA's [libcudf](https://github.com/rapidsai/cudf) -- GPU-accelerated DataFrame operations.

> **This project is unofficial and not affiliated with NVIDIA or RAPIDS.**

## What's New in v0.3.1

**libcudf v26.04.00 compatibility update.**

### Breaking Fix
- **`distinct_count` / `unique_count` header relocation**: Added `<cudf/reduction.hpp>` include to support libcudf v26.04.00 where these functions moved out of `<cudf/stream_compaction.hpp>`.

### New APIs
- **`ScanOp::CountValid` / `ScanOp::CountAll`**: Cumulative count operations for prefix scan, with null-handling control.
- **`Table::mark_semi_join` / `Table::mark_anti_join`**: Returns a boolean column instead of gather maps -- useful for filtering without a gather step. Based on libcudf v26.04.00 `mark_join` API.
- **`Column::index_of_first_valid`**: Find the index of the first non-null element in a column.

### Not Implemented (by design)
| API | Reason |
|-----|--------|
| `column_nans_to_nulls` | Existing `nans_to_nulls` shim already provides equivalent functionality |
| Named capture groups (regex) | Transparent through existing `regex_program::create` -- no code change needed |
| `packed_metadata_view` | Rust side copies bytes via `Vec<u8>` regardless, so a view provides no benefit |
| `from_arrow_device` / ArrowStringView | Significant standalone effort; deferred to a future release |

## What's New in v0.3.0

- **Polars integration via `_collect_post_opt`**: `collect_gpu()` now uses polars' native post-optimization callback. Zero changes to polars fork required for standalone use.
- **`Engine::Gpu` dispatch**: polars-fork integration enables `lf.collect_with_engine(Engine::Gpu)` -- cudf-polars as a native GPU backend.
- **New expressions**: `Not` (Bool logical / Int bitwise), `IsIn` (with null propagation), GroupBy `Quantile`.
- **Window functions**: `AExpr::Over` with `GroupsToRows` mapping -- groupby then scatter broadcast (O(n)).
- **Temporal types**: Date, Datetime (naive), Duration mapped to cudf timestamp/duration types.
- **Performance**: HStack/HConcat zero-copy via `into_parts()`, GPU-native scalar broadcast (`sequence_*`), HashMap name lookup O(1).
- **Safety**: `try_data_type()` fallible API, Arrow FFI debug assertions, First/Last empty guard.

## Features

- **Near-zero unsafe public API** -- all `unsafe` is confined to the internal FFI layer (sole exception: `DLPackTensor::from_raw_ptr`)
- **Zero-cost FFI** -- [cxx](https://cxx.rs) bridge with no serialization overhead
- **61 bridge modules** covering the full libcudf surface: compute, I/O, strings, nested types, interop
- **Arrow interop** -- conversion to/from `arrow-rs` via Arrow C Data Interface and IPC
- **Builder-pattern I/O** -- `ParquetReader`, `CsvReader`, `JsonReader`, `OrcReader`, `AvroReader`
- **GPU string processing** -- case, find, replace, split, regex, extract, and more
- **RAII memory management** -- `Column` and `Table` drops free GPU memory automatically

## Prerequisites

| Requirement | Version |
|-------------|---------|
| OS | Linux (libcudf does not support macOS or Windows) |
| GPU | NVIDIA Volta or newer (compute capability 7.0+) |
| CUDA | 12.2+ |
| libcudf | 24.0+ (tested with 26.04.00) |
| Rust | 1.85+ (edition 2024) |

## Installation

### 1. Install libcudf

**Conda (recommended):**

```sh
conda create -n cudf-dev -c rapidsai -c conda-forge libcudf cuda-version=12.2
conda activate cudf-dev
```

**From source:**

Follow the [RAPIDS build guide](https://docs.rapids.ai/install), then set:

```sh
export CUDF_ROOT=/path/to/libcudf/prefix
# expects: $CUDF_ROOT/lib/libcudf.so and $CUDF_ROOT/include/cudf/
```

### 2. Set CUDA path (if non-standard)

```sh
export CUDA_PATH=/usr/local/cuda
```

### 3. Add to your project

```toml
[dependencies]
cudf = "0.3"
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
    // Create GPU columns from host data
    let ids    = Column::from_slice(&[1i32, 1, 2, 2, 3])?;
    let values = Column::from_slice(&[10.0f64, 20.0, 30.0, 40.0, 50.0])?;
    let table  = Table::new(vec![ids, values])?;

    // Sort
    let sorted = table.sort(
        &[SortOrder::Ascending, SortOrder::Descending],
        &[NullOrder::After,     NullOrder::After],
    )?;

    // GroupBy
    let keys   = Table::new(vec![Column::from_slice(&[1i32, 1, 2, 2, 3])?])?;
    let vals   = Table::new(vec![Column::from_slice(&[10.0f64, 20.0, 30.0, 40.0, 50.0])?])?;
    let result = GroupBy::new(&keys)
        .agg(0, AggregationKind::Sum)
        .execute(&vals)?;

    // Parquet I/O
    parquet::write_parquet(&table, "/tmp/test.parquet")?;
    let loaded = parquet::read_parquet("/tmp/test.parquet")?;
    assert_eq!(loaded.num_rows(), 5);

    // Read data back to host
    let col  = loaded.column(0)?;
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

## Supported Operations

### Core

| Module | Description |
|--------|-------------|
| `column` | GPU-resident column with typed data and optional null bitmask |
| `table` | Ordered collection of columns (DataFrame equivalent) |
| `scalar` | GPU-resident single typed value with validity flag |
| `types` | `TypeId` and `DataType` mirroring libcudf's type system |

### Compute

| Module | Description |
|--------|-------------|
| `sorting` | Sort, rank, is-sorted checks |
| `groupby` | Builder-pattern groupby with multi-column aggregation |
| `reduction` | Reduce/scan a column (sum, min, max, count, ...) |
| `quantiles` | Quantile and percentile computation |
| `rolling` | Fixed-size rolling window aggregation |
| `binaryop` | Element-wise binary ops (arithmetic, comparison, logic) |
| `unary` | Element-wise unary ops (math, casts, null/NaN checks) |
| `round` | Numeric rounding |
| `transform` | NaN-to-null conversion, boolean mask generation |
| `search` | Binary search and containment checks |
| `hashing` | Row-wise hashing (Murmur3, MD5, SHA-256, ...) |
| `datetime` | Extract year, month, day, hour, ... from timestamps |

### Data Manipulation

| Module | Description |
|--------|-------------|
| `copying` | Gather, scatter, slice, split, conditional copy |
| `filling` | Fill, repeat, sequence generation |
| `concatenate` | Vertical stacking of columns and tables |
| `merge` | Merge pre-sorted tables |
| `join` | Inner, left, full outer, semi, anti, cross, mark joins |
| `stream_compaction` | Drop nulls, boolean mask, unique, distinct |
| `replace` | Replace nulls, NaNs, clamp values |
| `reshape` | Interleave and tile |
| `transpose` | Swap rows and columns |
| `partitioning` | Hash and round-robin partitioning |
| `lists` | Explode, sort, contains on list columns |
| `structs` | Extract child columns from struct columns |
| `dictionary` | Dictionary encoding/decoding |
| `json` | JSONPath queries on string columns |

### I/O

| Format | Read | Write | Compression |
|--------|------|-------|-------------|
| Parquet | `ParquetReader` | `ParquetWriter` | Snappy, Gzip, Zstd, LZ4 |
| CSV | `CsvReader` | `CsvWriter` | -- |
| JSON | `JsonReader` | `JsonWriter` | -- |
| ORC | `OrcReader` | `OrcWriter` | Snappy, Zlib, Zstd |
| Avro | `AvroReader` | -- | -- |

### String Operations (`strings::`)

Case, find, contains, replace, split, strip, slice, combine, convert, extract, findall, like, padding, partition, repeat, reverse, split_re, attributes, char_types, translate, wrap -- 21 submodules total.

### Interop

Arrow C Data Interface, Arrow IPC, DLPack tensor exchange, pack/unpack/contiguous_split.

## Limitations / Notes

- **GroupBy `maintain_order`**: Approximated by key-column sort, not true input-order preservation.
- **Std/Var ddof**: Default standalone reduction uses ddof=1. Full ddof support via `reduce_var_with_ddof` / `reduce_std_with_ddof`.
- **Polars version**: cudf-polars is compatible with Polars 0.53.0.
- **Unsupported types**: Categorical, List, Struct are not yet mapped (returns explicit error).
- **Unsupported expressions**: expression-level Sort/Filter/Slice.
- **Unsupported IR nodes**: `Cache`, `MapFunction`, `ExtContext`, `Sink`.
- **Multi-file Parquet**: Only reads the first file in multi-file scans.

## Testing

A GPU with CUDA is required to run tests.

```sh
# Unit tests (cudf crate — 55 tests)
cargo test -p cudf --features gpu-tests

# End-to-end tests (cudf-polars — 56 tests + 1 doctest)
cargo test -p cudf-polars --features gpu-tests

# Python polars-gpu integration tests (81 tests)
python tests/polars_gpu_integration.py
```

Total: **193 tests** covering type roundtrips, filter, select, sort, groupby, join, slice, binary ops, distinct, with_columns, concat, nulls, edge cases, and error paths.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on adding new bindings, code conventions, and testing.

## License

Licensed under either of

- [Apache License, Version 2.0](LICENSE-APACHE)
- [MIT License](LICENSE-MIT)

at your option.
