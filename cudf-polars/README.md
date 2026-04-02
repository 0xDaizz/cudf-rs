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
use polars_lazy::prelude::*;
use cudf_polars::collect_gpu;

fn main() {
    let df = df!(
        "id"    => [1i32, 2, 3, 1, 2, 3],
        "value" => [10.0f64, 20.0, 30.0, 40.0, 50.0, 60.0],
    ).unwrap();

    // Execute on GPU — one-line API
    let result = collect_gpu(
        df.lazy()
          .filter(col("value").gt(lit(25.0)))
          .group_by([col("id")])
          .agg([col("value").sum()])
    ).unwrap();
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

`execute_plan()` takes an `IRPlan` obtained from polars-lazy's `to_alp_optimized()`.

### LazyFrame Integration

```rust
use polars_lazy::frame::LazyFrame;
use cudf_polars::engine::execute_plan;

let lf: LazyFrame = df.lazy().filter(...).group_by(...);
let plan = lf.to_alp_optimized()?;
let result = execute_plan(plan)?;
```

## Testing

```sh
# Run GPU e2e tests (56 tests + 1 doctest)
cargo test -p cudf-polars --features gpu-tests

# Python polars-gpu integration (81 tests)
python tests/polars_gpu_integration.py
```

## Limitations

- **Polars version**: Compatible with polars 0.53.0.
- **Unsupported types**: Date, Datetime, Duration, Categorical, List, Struct return explicit errors.
- **Unsupported expressions**: Window functions (`.over()`), `IsIn`, expression-level Sort/Filter/Slice, `Not`.
- **Unsupported IR nodes**: `Cache`, `MapFunction` (rename, explode, melt), `ExtContext`.
- **Multi-file Parquet**: Only reads the first file in multi-file scans.
- **GroupBy maintain_order**: Approximated by key-column sort (not true input-order preservation).
- **Quantile aggregation**: Not supported in GroupBy context.

## License

Apache-2.0 OR MIT
