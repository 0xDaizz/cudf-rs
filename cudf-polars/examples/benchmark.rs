//! CPU vs GPU benchmark for cudf-polars operations.
//!
//! Run with: cargo run --example benchmark -p cudf-polars --features gpu-tests --release

use std::time::Instant;

use cudf::aggregation::AggregationKind;
use cudf::binaryop::BinaryOp;
use cudf::sorting::{NullOrder, SortOrder};
use cudf::types::{DataType, TypeId};
use cudf::Scalar;
use polars_core::prelude::*;

use cudf_polars::convert;
use cudf_polars::gpu_frame::GpuDataFrame;

fn main() {
    let n: usize = 1_000_000;
    println!("=== cudf-polars Benchmark ===");
    println!("Rows: {}\n", n);

    // Generate data
    let ids: Vec<i32> = (0..n as i32).map(|i| i % 1000).collect();
    let values: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();
    let flags: Vec<bool> = (0..n).map(|i| i % 2 == 0).collect();
    let df = df!(
        "id" => &ids,
        "value" => &values,
        "flag" => &flags,
    )
    .unwrap();

    // Warm up GPU with a small roundtrip
    {
        let small = df!("x" => [1i32, 2, 3]).unwrap();
        let gpu = GpuDataFrame::from_polars(&small).unwrap();
        let _ = gpu.to_polars().unwrap();
    }

    println!(
        "| Operation | CPU (ms) | GPU total (ms) | GPU compute (ms) | Speedup |"
    );
    println!(
        "|-----------|----------|----------------|------------------|---------|"
    );

    // -- Filter: value > 50000.0 --
    {
        // CPU
        let start = Instant::now();
        let mask = df.column("value").unwrap().f64().unwrap().gt(50000.0);
        let _cpu = df.filter(&mask).unwrap();
        let cpu_ms = start.elapsed().as_secs_f64() * 1000.0;

        // GPU
        let start = Instant::now();
        let gpu_df = GpuDataFrame::from_polars(&df).unwrap();
        let upload_ms = start.elapsed().as_secs_f64() * 1000.0;

        let start2 = Instant::now();
        let val_col = gpu_df.column_by_name("value").unwrap();
        let threshold = Scalar::new(50000.0f64).unwrap();
        let mask_col = val_col
            .binary_op_scalar(
                &threshold,
                BinaryOp::Greater,
                DataType::new(TypeId::Bool8),
            )
            .unwrap();
        let filtered = gpu_df.apply_boolean_mask(&mask_col).unwrap();
        let compute_ms = start2.elapsed().as_secs_f64() * 1000.0;

        let start3 = Instant::now();
        let _result = filtered.to_polars().unwrap();
        let download_ms = start3.elapsed().as_secs_f64() * 1000.0;
        let gpu_total = upload_ms + compute_ms + download_ms;

        let speedup = if gpu_total > 0.0 {
            cpu_ms / gpu_total
        } else {
            0.0
        };
        println!(
            "| Filter    | {:.2}   | {:.2}         | {:.2}            | {:.2}x  |",
            cpu_ms, gpu_total, compute_ms, speedup
        );
    }

    // -- Sort by value ascending --
    {
        // CPU
        let start = Instant::now();
        let _cpu = df
            .sort(["value"], SortMultipleOptions::default())
            .unwrap();
        let cpu_ms = start.elapsed().as_secs_f64() * 1000.0;

        // GPU
        let start = Instant::now();
        let gpu_df = GpuDataFrame::from_polars(&df).unwrap();
        let upload_ms = start.elapsed().as_secs_f64() * 1000.0;

        let start2 = Instant::now();
        let key_col = gpu_df.column_by_name("value").unwrap();
        let sorted = gpu_df
            .sort_by_key(
                vec![key_col],
                &[SortOrder::Ascending],
                &[NullOrder::After],
            )
            .unwrap();
        let compute_ms = start2.elapsed().as_secs_f64() * 1000.0;

        let start3 = Instant::now();
        let _result = sorted.to_polars().unwrap();
        let download_ms = start3.elapsed().as_secs_f64() * 1000.0;
        let gpu_total = upload_ms + compute_ms + download_ms;

        let speedup = if gpu_total > 0.0 {
            cpu_ms / gpu_total
        } else {
            0.0
        };
        println!(
            "| Sort      | {:.2}   | {:.2}         | {:.2}            | {:.2}x  |",
            cpu_ms, gpu_total, compute_ms, speedup
        );
    }

    // -- GroupBy id -> sum(value) --
    {
        // CPU
        let start = Instant::now();
        let _cpu = df
            .group_by(["id"])
            .unwrap()
            .select(["value"])
            .sum()
            .unwrap();
        let cpu_ms = start.elapsed().as_secs_f64() * 1000.0;

        // GPU
        let start = Instant::now();
        let gpu_df = GpuDataFrame::from_polars(&df).unwrap();
        let upload_ms = start.elapsed().as_secs_f64() * 1000.0;

        let start2 = Instant::now();
        let key_col = gpu_df.column_by_name("id").unwrap();
        let val_col = gpu_df.column_by_name("value").unwrap();
        let grouped = gpu_df
            .groupby(
                vec![key_col],
                vec!["id".to_string()],
                vec![val_col],
                vec![(0, AggregationKind::Sum)],
                vec!["value_sum".to_string()],
                false,
            )
            .unwrap();
        let compute_ms = start2.elapsed().as_secs_f64() * 1000.0;

        let start3 = Instant::now();
        let _result = grouped.to_polars().unwrap();
        let download_ms = start3.elapsed().as_secs_f64() * 1000.0;
        let gpu_total = upload_ms + compute_ms + download_ms;

        let speedup = if gpu_total > 0.0 {
            cpu_ms / gpu_total
        } else {
            0.0
        };
        println!(
            "| GroupBy   | {:.2}   | {:.2}         | {:.2}            | {:.2}x  |",
            cpu_ms, gpu_total, compute_ms, speedup
        );
    }

    println!();
    println!("Note: GPU total = CPU->GPU upload + compute + GPU->CPU download.");
    println!("      GPU compute is the actual kernel execution time.");
}
