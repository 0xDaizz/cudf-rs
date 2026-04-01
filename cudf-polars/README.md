# cudf-polars

GPU execution engine for [Polars](https://pola.rs/) DataFrames using NVIDIA libcudf.

`cudf-polars` transparently offloads Polars DataFrame operations to the GPU,
providing significant speedups for filter, sort, groupby, join, and other
data-intensive operations.

## Prerequisites

- NVIDIA GPU with CUDA support
- CUDA Toolkit 12.x
- libcudf (built from `cudf-sys` / `cudf-cxx` in this workspace)
- Rust 1.85+

## Quick Start

```rust
use polars_core::prelude::*;
use cudf_polars::gpu_frame::GpuDataFrame;

fn main() {
    // Create a Polars DataFrame
    let df = df!(
        "id"    => [1i32, 2, 3, 1, 2, 3],
        "value" => [10.0f64, 20.0, 30.0, 40.0, 50.0, 60.0],
    ).unwrap();

    // Upload to GPU
    let gpu_df = GpuDataFrame::from_polars(&df).unwrap();
    println!("GPU: {} rows x {} cols", gpu_df.height(), gpu_df.width());

    // Download back to CPU
    let result = gpu_df.to_polars().unwrap();
    println!("{}", result);
}
```

## Supported Operations

| Category       | Operation              | API                                 | Status |
|----------------|------------------------|-------------------------------------|--------|
| **Transfer**   | CPU -> GPU             | `GpuDataFrame::from_polars()`       | Done   |
|                | GPU -> CPU             | `GpuDataFrame::to_polars()`         | Done   |
| **Selection**  | Column select          | `GpuDataFrame::select_columns()`    | Done   |
|                | Row slice              | `GpuDataFrame::slice()`             | Done   |
| **Filter**     | Boolean mask           | `GpuDataFrame::apply_boolean_mask()`| Done   |
| **Sort**       | Sort by key columns    | `GpuDataFrame::sort_by_key()`       | Done   |
| **GroupBy**     | Aggregation            | `GpuDataFrame::groupby()`           | Done   |
| **Dedup**      | Distinct rows          | `GpuDataFrame::distinct()`          | Done   |
| **Join**       | Inner/Left/Full        | `Table::inner_join()` etc.          | Done   |
|                | Semi/Anti/Cross        | `Table::left_semi_join()` etc.      | Done   |
| **Union**      | Vertical concat        | `concatenate_tables()`              | Done   |
| **HConcat**    | Horizontal concat      | Column collection                   | Done   |
| **Binary Ops** | Column-column          | `Column::binary_op()`               | Done   |
|                | Column-scalar          | `Column::binary_op_scalar()`        | Done   |
| **Ternary**    | when/then/otherwise    | `Column::copy_if_else()`            | Done   |
| **Expression** | Polars expr -> GPU     | `cudf_polars::expr`                 | Done   |
| **Plan Exec**  | Full plan execution    | `cudf_polars::execute_plan()`       | Done   |

### Supported Aggregations (GroupBy)

Sum, Min, Max, Count, Mean, Median, Variance, Std, Nunique, First, Last.

### Supported Data Types

| Polars Type  | cudf Type   |
|-------------|-------------|
| Int8        | INT8        |
| Int16       | INT16       |
| Int32       | INT32       |
| Int64       | INT64       |
| UInt8       | UINT8       |
| UInt16      | UINT16      |
| UInt32      | UINT32      |
| UInt64      | UINT64      |
| Float32     | FLOAT32     |
| Float64     | FLOAT64     |
| Boolean     | BOOL8       |
| String      | STRING      |

## Benchmark

```bash
cargo run --example benchmark -p cudf-polars --features gpu-tests --release
```

## Architecture

```
Polars DataFrame
      |
      v  (Arrow C Data Interface)
cudf-polars::convert   -- zero-copy CPU <-> GPU bridge
      |
      v
cudf-polars::GpuDataFrame -- named GPU columns
      |
      v
cudf (Rust)  ->  cudf-cxx (C++ bridge)  ->  libcudf (NVIDIA)
```

## Limitations

- **Polars version**: Pinned to polars 0.46.0. Upgrading requires manual IR compatibility audit.
- **Types**: Date, Time, Datetime, Duration, Categorical, List, Struct types pass through Arrow bridge but are not tested.
- **String operations**: String-specific expressions (contains, replace, split) are not supported.
- **Window functions**:  expressions are not supported.
- **GroupBy maintain_order**: Approximated by key-column sort (not true input-order preservation).
- **Std/Var ddof**: Default standalone reduce uses ddof=1; full ddof support available via `reduce_var_with_ddof` / `reduce_std_with_ddof`. GroupBy passes ddof correctly.
- **IsFinite/IsInfinite**: Fully supported via `is_inf()` / `is_not_inf()`.
- **Quantile aggregation**: Not supported in GroupBy context.
- **Multi-file Parquet**: Only reads the first file in multi-file scans.
- **MapFunction**: rename, explode, melt, pivot not supported.
- **Cache/ExtContext**: IR nodes not supported.

## License

Apache-2.0 OR MIT
