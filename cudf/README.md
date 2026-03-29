# cudf

Safe Rust bindings for NVIDIA's [libcudf](https://github.com/rapidsai/cudf) -- GPU-accelerated DataFrame operations.

## Features

- **100% safe public API** -- all `unsafe` is confined to the internal FFI layer
- **Zero-cost ownership** -- `Column` and `Table` map directly to libcudf's RAII types
- **Compile-time safety** -- Rust's borrow checker prevents use-after-free on GPU memory
- **Arrow interop** -- zero-copy conversion to/from `arrow-rs` arrays
- **Full libcudf coverage** -- all operations including groupby, join, sort, I/O, strings

## Quick Start

```rust,no_run
use cudf::{Column, Table, Result};

fn main() -> Result<()> {
    // Create GPU columns from host data
    let ids = Column::from_slice(&[1i32, 2, 3, 4, 5])?;
    let values = Column::from_slice(&[10.0f64, 20.0, 30.0, 40.0, 50.0])?;

    // Build a table
    let table = Table::new(vec![ids, values])?;
    assert_eq!(table.num_columns(), 2);
    assert_eq!(table.num_rows(), 5);

    // Read back
    let col = table.column(0)?;
    let data: Vec<i32> = col.to_vec()?;
    assert_eq!(data, vec![1, 2, 3, 4, 5]);

    Ok(())
}
```

## Prerequisites

- NVIDIA GPU (Volta or newer, compute capability 7.0+)
- CUDA 12.2+
- libcudf installed (see [cudf-sys](../cudf-sys/) for installation instructions)
- Linux (libcudf does not support macOS or Windows)

## Crate Structure

```
cudf-rs/
├── cudf-sys   -- links libcudf.so (build script only)
├── cudf-cxx   -- cxx-based FFI bridge + C++ shim layer
└── cudf       -- this crate: safe, idiomatic Rust API
```

## Public API Surface

### Core Types

| Type | Description |
|------|-------------|
| `Column` | GPU-resident column with typed data and optional null bitmask |
| `Table` | Ordered collection of `Column`s (DataFrame equivalent) |
| `Scalar` | GPU-resident single typed value with validity flag |
| `DataType` / `TypeId` | Type system mirroring libcudf |
| `CudfError` / `Result<T>` | Unified error handling with C++ exception conversion |

### Re-exports

```rust
pub use Column, Table, Scalar;
pub use DataType, TypeId;
pub use CudfError, Result;
pub use SortOrder, NullOrder;
pub use OutOfBoundsPolicy;
pub use UnaryOp, BinaryOp;
pub use DuplicateKeepOption;
pub use AggregationKind, GroupBy;
pub use ReduceOp, ScanOp;
pub use Interpolation;
pub use RollingAgg;
pub use JoinResult;
```

### Compute Operations

| Function / Method | Module | Description |
|-------------------|--------|-------------|
| `table.sort()` | `sorting` | Sort table by one or more columns |
| `GroupBy::new(&keys).agg(...).execute(&values)` | `groupby` | Groupby aggregation |
| `col.reduce(op, dtype)` | `reduction` | Reduce column to scalar |
| `col.scan(op, inclusive)` | `reduction` | Prefix-sum / scan |
| `col.quantile(q, interp)` | `quantiles` | Compute quantiles |
| `col.rolling_agg(agg, window, min_periods)` | `rolling` | Rolling window aggregation |
| `col.binary_op(op, &rhs, out_type)` | `binaryop` | Element-wise binary ops |
| `col.unary_op(op)` | `unary` | Element-wise unary ops |
| `col.round(decimals)` | `round` | Numeric rounding |
| `table.hash(algo)` | `hashing` | Row-wise hashing |
| `col.extract_year()`, `.extract_month()`, ... | `datetime` | Datetime component extraction |
| `table.lower_bound(...)`, `table.upper_bound(...)` | `search` | Binary search on sorted tables |
| `col.nans_to_nulls()` | `transform` | NaN-to-null conversion |

### Data Manipulation

| Function / Method | Module | Description |
|-------------------|--------|-------------|
| `table.gather(&map)` | `copying` | Gather rows by index |
| `table.scatter(...)` | `copying` | Scatter values to indices |
| `table.slice(offset, size)` | `copying` | Slice a contiguous range |
| `table.split(indices)` | `copying` | Split at given indices |
| `col.fill(value, begin, end)` | `filling` | Fill range with value |
| `Table::concatenate(&[tables])` | `concatenate` | Vertical stacking |
| `Table::merge(...)` | `merge` | Merge pre-sorted tables |
| `table.inner_join(...)` | `join` | Inner / left / full / cross join |
| `table.drop_nulls(...)` | `stream_compaction` | Drop null rows |
| `table.apply_boolean_mask(...)` | `stream_compaction` | Filter by boolean mask |
| `table.unique(...)` | `stream_compaction` | Remove duplicates |
| `table.interleave()` | `reshape` | Interleave columns |
| `table.transpose()` | `transpose` | Swap rows and columns |
| `table.hash_partition(...)` | `partitioning` | Hash / round-robin partition |

### I/O

| Function | Module | Description |
|----------|--------|-------------|
| `parquet::read_parquet(path)` | `io::parquet` | Read Parquet file to GPU |
| `parquet::write_parquet(&table, path)` | `io::parquet` | Write table to Parquet |
| `csv::read_csv(path)` | `io::csv` | Read CSV file to GPU |
| `csv::write_csv(&table, path)` | `io::csv` | Write table to CSV |
| `json::read_json(path)` | `io::json` | Read JSON file to GPU |
| `json::write_json(&table, path)` | `io::json` | Write table to JSON |
| `orc::read_orc(path)` | `io::orc` | Read ORC file to GPU |
| `orc::write_orc(&table, path)` | `io::orc` | Write table to ORC |
| `avro::read_avro(path)` | `io::avro` | Read Avro file to GPU |

### String Operations

All string operations are methods on `Column` (for string-typed columns):

| Method | Module | Description |
|--------|--------|-------------|
| `col.str_to_upper()` / `str_to_lower()` | `strings::case` | Case conversion |
| `col.str_find(target)` | `strings::find` | Find substring position |
| `col.str_contains(target)` / `str_contains_re(pattern)` | `strings::contains` | Containment checks |
| `col.str_replace(target, repl)` / `str_replace_re(...)` | `strings::replace` | Replacement |
| `col.str_split(delimiter)` | `strings::split` | Split into columns |
| `col.str_strip(chars)` | `strings::strip` | Trim leading/trailing chars |
| `col.str_slice(start, stop)` | `strings::slice` | Substring extraction |
| `col.str_cat(separator)` | `strings::combine` | Concatenation |
| `col.str_to_integers(dtype)` / `col.integers_to_str()` | `strings::convert` | Type conversion |
| `col.str_extract(pattern)` | `strings::extract` | Regex capture groups |

### Arrow Interop

| Method | Module | Description |
|--------|--------|-------------|
| `col.to_arrow_ipc()` | `interop` | Export column to Arrow IPC bytes |
| `Column::from_arrow_ipc(data)` | `interop` | Import column from Arrow IPC bytes |
| `table.to_arrow_ipc()` | `interop` | Export table to Arrow IPC bytes |
| `Table::from_arrow_ipc(data)` | `interop` | Import table from Arrow IPC bytes |

## Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `arrow-interop` | Yes | Zero-copy conversion to/from `arrow` arrays |
