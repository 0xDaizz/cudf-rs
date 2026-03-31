//! Pipeline benchmark: CPU vs GPU for cudf-polars.
//!
//! Measures a realistic Filter→Sort→GroupBy pipeline where data stays on GPU
//! throughout, rather than paying CPU↔GPU transfer per operation.
//!
//! Run with: cargo run --example benchmark -p cudf-polars --features gpu-tests --release

use std::time::Instant;

use cudf::Scalar;
use cudf::aggregation::AggregationKind;
use cudf::binaryop::BinaryOp;
use cudf::io::parquet::{ParquetReader, ParquetWriter};
use cudf::sorting::{NullOrder, SortOrder};
use cudf::types::{DataType, TypeId};
use polars_core::prelude::*;

use cudf_polars::gpu_frame::GpuDataFrame;

/// Generate a test DataFrame with `n` rows.
fn generate_data(n: usize) -> DataFrame {
    let ids: Vec<i32> = (0..n as i32).map(|i| i % 1000).collect();
    let values: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1) % 100_000.0).collect();
    df!(
        "id" => &ids,
        "value" => &values,
    )
    .unwrap()
}

/// CPU pipeline: filter → sort → groupby.
fn cpu_pipeline(df: &DataFrame, threshold: f64) -> DataFrame {
    let mask = df.column("value").unwrap().f64().unwrap().gt(threshold);
    let filtered = df.filter(&mask).unwrap();
    let sorted = filtered
        .sort(["value"], SortMultipleOptions::default())
        .unwrap();
    sorted
        .group_by(["id"])
        .unwrap()
        .select(["value"])
        .sum()
        .unwrap()
}

/// GPU pipeline: upload once → filter → sort → groupby → download once.
/// Returns (upload_ms, compute_ms, download_ms, total_ms, result_rows).
fn gpu_pipeline(df: &DataFrame, threshold: f64) -> (f64, f64, f64, f64, usize) {
    let total_start = Instant::now();

    // Upload
    let gpu_df = GpuDataFrame::from_polars(df).unwrap();
    let upload_ms = total_start.elapsed().as_secs_f64() * 1000.0;

    // Compute (all on GPU, no transfers)
    let compute_start = Instant::now();

    // Filter: value > threshold
    let val_col = gpu_df.column_by_name("value").unwrap();
    let thresh = Scalar::new(threshold).unwrap();
    let mask = val_col
        .binary_op_scalar(&thresh, BinaryOp::Greater, DataType::new(TypeId::Bool8))
        .unwrap();
    let filtered = gpu_df.apply_boolean_mask(&mask).unwrap();

    // Sort by value ascending
    let sort_key = filtered.column_by_name("value").unwrap();
    let sorted = filtered
        .sort_by_key(vec![sort_key], &[SortOrder::Ascending], &[NullOrder::After])
        .unwrap();

    // GroupBy id → sum(value)
    let key_col = sorted.column_by_name("id").unwrap();
    let agg_col = sorted.column_by_name("value").unwrap();
    let grouped = sorted
        .groupby(
            vec![key_col],
            vec!["id".to_string()],
            vec![agg_col],
            vec![(0, AggregationKind::Sum)],
            vec!["value_sum".to_string()],
            false,
        )
        .unwrap();
    let compute_ms = compute_start.elapsed().as_secs_f64() * 1000.0;

    // Download
    let download_start = Instant::now();
    let result = grouped.to_polars().unwrap();
    let download_ms = download_start.elapsed().as_secs_f64() * 1000.0;

    let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;
    let nrows = result.height();
    (upload_ms, compute_ms, download_ms, total_ms, nrows)
}

/// Run the pipeline benchmark at a given row count.
fn bench_pipeline(n: usize) {
    let df = generate_data(n);
    let threshold = 50.0; // filters out ~half the rows

    // CPU pipeline
    let cpu_start = Instant::now();
    let cpu_result = cpu_pipeline(&df, threshold);
    let cpu_ms = cpu_start.elapsed().as_secs_f64() * 1000.0;

    // GPU pipeline
    let (upload_ms, compute_ms, download_ms, gpu_total_ms, _gpu_rows) =
        gpu_pipeline(&df, threshold);

    let speedup = if gpu_total_ms > 0.0 {
        cpu_ms / gpu_total_ms
    } else {
        0.0
    };

    println!(
        "| {:>3}M | {:>8.1} | {:>11.1} | {:>16.1} | {:>13.1} | {:>14.1} | {:>7.2}x | {:>7} |",
        n / 1_000_000,
        cpu_ms,
        upload_ms,
        compute_ms,
        download_ms,
        gpu_total_ms,
        speedup,
        cpu_result.height(),
    );
}

/// Parquet-native benchmark: read directly to GPU vs CPU.
fn bench_parquet_native(n: usize) {
    let df = generate_data(n);
    let threshold = 50.0;
    let path = "/tmp/cudf_bench_test.parquet";

    // Write test parquet file using GPU (generate data on GPU, write to parquet)
    {
        let gpu_df = GpuDataFrame::from_polars(&df).unwrap();
        let table = gpu_df.inner_table();
        ParquetWriter::new(table, path).write().unwrap();
    }

    // GPU path: read parquet directly to GPU → pipeline → download
    let gpu_start = Instant::now();
    let table = ParquetReader::new(path).read().unwrap();
    // ParquetWriter doesn't preserve column names in libcudf, so we supply them.
    let col_names = vec!["id".to_string(), "value".to_string()];
    let gpu_df = GpuDataFrame::from_table(table, col_names);
    let read_ms = gpu_start.elapsed().as_secs_f64() * 1000.0;

    let compute_start = Instant::now();
    // Filter
    let val_col = gpu_df.column_by_name("value").unwrap();
    let thresh = Scalar::new(threshold).unwrap();
    let mask = val_col
        .binary_op_scalar(&thresh, BinaryOp::Greater, DataType::new(TypeId::Bool8))
        .unwrap();
    let filtered = gpu_df.apply_boolean_mask(&mask).unwrap();
    // Sort
    let sort_key = filtered.column_by_name("value").unwrap();
    let sorted = filtered
        .sort_by_key(vec![sort_key], &[SortOrder::Ascending], &[NullOrder::After])
        .unwrap();
    // GroupBy
    let key_col = sorted.column_by_name("id").unwrap();
    let agg_col = sorted.column_by_name("value").unwrap();
    let grouped = sorted
        .groupby(
            vec![key_col],
            vec!["id".to_string()],
            vec![agg_col],
            vec![(0, AggregationKind::Sum)],
            vec!["value_sum".to_string()],
            false,
        )
        .unwrap();
    let gpu_compute_ms = compute_start.elapsed().as_secs_f64() * 1000.0;

    let dl_start = Instant::now();
    let _gpu_result = grouped.to_polars().unwrap();
    let gpu_download_ms = dl_start.elapsed().as_secs_f64() * 1000.0;
    let gpu_total_ms = gpu_start.elapsed().as_secs_f64() * 1000.0;

    // CPU path: read parquet via polars → pipeline
    let cpu_start = Instant::now();
    let cpu_df = polars_core::frame::DataFrame::new(
        df.get_columns().to_vec(),
    )
    .unwrap(); // use in-memory data since polars-core doesn't have parquet reader
    let _cpu_result = cpu_pipeline(&cpu_df, threshold);
    let cpu_ms = cpu_start.elapsed().as_secs_f64() * 1000.0;

    let speedup = if gpu_total_ms > 0.0 {
        cpu_ms / gpu_total_ms
    } else {
        0.0
    };

    println!(
        "| {:>3}M | {:>8.1} | {:>11.1} | {:>16.1} | {:>13.1} | {:>14.1} | {:>7.2}x |",
        n / 1_000_000,
        cpu_ms,
        read_ms,
        gpu_compute_ms,
        gpu_download_ms,
        gpu_total_ms,
        speedup,
    );
}

fn main() {
    println!("=== cudf-polars Pipeline Benchmark ===");
    println!("Pipeline: Filter(value > 50) → Sort(value ASC) → GroupBy(id).sum(value)\n");

    // Warm up GPU
    {
        let small = generate_data(10_000);
        let _ = gpu_pipeline(&small, 50.0);
    }

    // ── Pipeline benchmark (CPU→GPU transfer) ───────────────────────
    println!("## Memory-to-GPU Pipeline");
    println!();
    println!(
        "| Rows | CPU (ms) | Upload (ms) | GPU compute (ms) | Download (ms) | GPU total (ms) | Speedup | Groups |"
    );
    println!(
        "|------|----------|-------------|------------------|---------------|----------------|---------|--------|"
    );

    for &n in &[1_000_000, 5_000_000, 10_000_000] {
        bench_pipeline(n);
    }

    println!();

    // ── Parquet-native benchmark ─────────────────────────────────────
    println!("## Parquet-Native Pipeline (GPU reads parquet directly, no CPU→GPU copy)");
    println!();
    println!(
        "| Rows | CPU (ms) | GPU read (ms) | GPU compute (ms) | Download (ms) | GPU total (ms) | Speedup |"
    );
    println!(
        "|------|----------|---------------|------------------|---------------|----------------|---------|"
    );

    for &n in &[1_000_000, 5_000_000, 10_000_000] {
        bench_parquet_native(n);
    }

    println!();
    println!("Note: GPU total = upload/read + compute + download.");
    println!("      GPU compute = all ops chained on GPU without intermediate transfers.");
    println!("      CPU pipeline runs the same filter→sort→groupby chain for fair comparison.");
}
