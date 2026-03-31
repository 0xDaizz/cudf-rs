//! GPU execution engine for Polars using NVIDIA libcudf.

pub mod convert;
pub mod engine;
pub mod error;
pub mod expr;
pub mod gpu_frame;
pub mod types;

pub use engine::execute_plan;
pub use gpu_frame::GpuDataFrame;

#[cfg(test)]
mod tests {
    use super::*;
    use polars_core::prelude::*;

    #[test]
    #[cfg(feature = "gpu-tests")]
    fn roundtrip_i32() {
        let df = df!("x" => [1i32, 2, 3, 4, 5]).unwrap();
        let (gpu_table, names) = convert::dataframe_to_gpu(&df).unwrap();
        let back = convert::gpu_to_dataframe(gpu_table, &names).unwrap();
        let orig = df.column("x").unwrap().i32().unwrap();
        let result = back.column("x").unwrap().i32().unwrap();
        assert_eq!(orig.len(), result.len());
        for i in 0..orig.len() {
            assert_eq!(orig.get(i), result.get(i), "mismatch at index {}", i);
        }
    }

    #[test]
    #[cfg(feature = "gpu-tests")]
    fn roundtrip_f64() {
        let df = df!("val" => [1.1f64, 2.2, 3.3]).unwrap();
        let (gpu_table, names) = convert::dataframe_to_gpu(&df).unwrap();
        let back = convert::gpu_to_dataframe(gpu_table, &names).unwrap();
        let orig = df.column("val").unwrap().f64().unwrap();
        let result = back.column("val").unwrap().f64().unwrap();
        assert_eq!(orig.len(), result.len());
        for i in 0..orig.len() {
            let o = orig.get(i).unwrap();
            let r = result.get(i).unwrap();
            assert!((o - r).abs() < f64::EPSILON, "mismatch at index {}: {} vs {}", i, o, r);
        }
    }

    #[test]
    #[cfg(feature = "gpu-tests")]
    fn roundtrip_string() {
        let df = df!("s" => ["hello", "world", "gpu"]).unwrap();
        let (gpu_table, names) = convert::dataframe_to_gpu(&df).unwrap();
        let back = convert::gpu_to_dataframe(gpu_table, &names).unwrap();
        let orig = df.column("s").unwrap().str().unwrap();
        let result = back.column("s").unwrap().str().unwrap();
        assert_eq!(orig.len(), result.len());
        for i in 0..orig.len() {
            assert_eq!(orig.get(i), result.get(i), "mismatch at index {}", i);
        }
    }

    #[test]
    #[cfg(feature = "gpu-tests")]
    fn roundtrip_multi_column() {
        let df = df!(
            "id" => [1i64, 2, 3],
            "value" => [10.0f64, 20.0, 30.0],
            "name" => ["a", "b", "c"]
        )
        .unwrap();
        let (gpu_table, names) = convert::dataframe_to_gpu(&df).unwrap();
        let back = convert::gpu_to_dataframe(gpu_table, &names).unwrap();
        assert_eq!(df.height(), back.height());
        assert_eq!(df.width(), back.width());
        let orig_id = df.column("id").unwrap().i64().unwrap();
        let result_id = back.column("id").unwrap().i64().unwrap();
        for i in 0..orig_id.len() {
            assert_eq!(orig_id.get(i), result_id.get(i), "id mismatch at index {}", i);
        }
        let orig_val = df.column("value").unwrap().f64().unwrap();
        let result_val = back.column("value").unwrap().f64().unwrap();
        for i in 0..orig_val.len() {
            let o = orig_val.get(i).unwrap();
            let r = result_val.get(i).unwrap();
            assert!((o - r).abs() < f64::EPSILON, "value mismatch at index {}: {} vs {}", i, o, r);
        }
        let orig_name = df.column("name").unwrap().str().unwrap();
        let result_name = back.column("name").unwrap().str().unwrap();
        for i in 0..orig_name.len() {
            assert_eq!(orig_name.get(i), result_name.get(i), "name mismatch at index {}", i);
        }
    }

    #[test]
    #[cfg(feature = "gpu-tests")]
    fn roundtrip_boolean() {
        let df = df!("flag" => [true, false, true]).unwrap();
        let (gpu_table, names) = convert::dataframe_to_gpu(&df).unwrap();
        let back = convert::gpu_to_dataframe(gpu_table, &names).unwrap();
        let orig = df.column("flag").unwrap().bool().unwrap();
        let result = back.column("flag").unwrap().bool().unwrap();
        assert_eq!(orig.len(), result.len());
        for i in 0..orig.len() {
            assert_eq!(orig.get(i), result.get(i), "mismatch at index {}", i);
        }
    }

    #[test]
    #[cfg(feature = "gpu-tests")]
    fn roundtrip_nullable_i32() {
        let df = df!("x" => &[Some(1i32), None, Some(3), None, Some(5)]).unwrap();
        let (gpu_table, names) = convert::dataframe_to_gpu(&df).unwrap();
        let back = convert::gpu_to_dataframe(gpu_table, &names).unwrap();
        let orig = df.column("x").unwrap().i32().unwrap();
        let result = back.column("x").unwrap().i32().unwrap();
        assert_eq!(orig.len(), result.len());
        for i in 0..orig.len() {
            assert_eq!(orig.get(i), result.get(i), "mismatch at index {}", i);
        }
    }

    #[test]
    #[cfg(feature = "gpu-tests")]
    fn roundtrip_empty() {
        let df = df!("x" => Vec::<i32>::new()).unwrap();
        let (gpu_table, names) = convert::dataframe_to_gpu(&df).unwrap();
        let back = convert::gpu_to_dataframe(gpu_table, &names).unwrap();
        assert_eq!(back.height(), 0);
        assert_eq!(back.width(), 1);
    }
}

#[cfg(test)]
mod engine_tests {
    use crate::expr as gpu_expr;
    use crate::error as gpu_error;
    use crate::gpu_frame::GpuDataFrame;
    use polars_core::prelude::*;

    // ── GpuDataFrame component tests ──

    #[test]
    #[cfg(feature = "gpu-tests")]
    fn gpu_frame_select_columns() {
        let df = df!(
            "a" => [1i32, 2, 3],
            "b" => [4i32, 5, 6],
            "c" => [7i32, 8, 9]
        ).unwrap();
        let gpu_df = GpuDataFrame::from_polars(&df).unwrap();
        let selected = gpu_df.select_columns(&["a", "c"]).unwrap();
        assert_eq!(selected.width(), 2);
        assert_eq!(selected.height(), 3);
        let back = selected.to_polars().unwrap();
        assert_eq!(back.width(), 2);
        let a = back.column("a").unwrap().i32().unwrap();
        assert_eq!(a.get(0), Some(1));
        assert_eq!(a.get(2), Some(3));
        let c = back.column("c").unwrap().i32().unwrap();
        assert_eq!(c.get(0), Some(7));
    }

    #[test]
    #[cfg(feature = "gpu-tests")]
    fn gpu_frame_boolean_mask() {
        let df = df!("x" => [1i32, 2, 3, 4, 5]).unwrap();
        let gpu_df = GpuDataFrame::from_polars(&df).unwrap();

        // Create mask: x > 2 → [false, false, true, true, true]
        let x_col = gpu_df.column_by_name("x").unwrap();
        let threshold = cudf::Scalar::new(2i32).unwrap();
        let mask = gpu_error::gpu_result(
            x_col.binary_op_scalar(
                &threshold,
                cudf::BinaryOp::Greater,
                cudf::types::DataType::new(cudf::types::TypeId::Bool8),
            )
        ).unwrap();

        let filtered = gpu_df.apply_boolean_mask(&mask).unwrap();
        assert_eq!(filtered.height(), 3);
        let back = filtered.to_polars().unwrap();
        let vals: Vec<i32> = back.column("x").unwrap().i32().unwrap()
            .into_no_null_iter().collect();
        assert_eq!(vals, vec![3, 4, 5]);
    }

    #[test]
    #[cfg(feature = "gpu-tests")]
    fn gpu_frame_slice() {
        let df = df!("x" => [10i32, 20, 30, 40, 50]).unwrap();
        let gpu_df = GpuDataFrame::from_polars(&df).unwrap();
        let sliced = gpu_df.slice(1, 3).unwrap();
        assert_eq!(sliced.height(), 3);
        let back = sliced.to_polars().unwrap();
        let vals: Vec<i32> = back.column("x").unwrap().i32().unwrap()
            .into_no_null_iter().collect();
        assert_eq!(vals, vec![20, 30, 40]);
    }

    #[test]
    #[cfg(feature = "gpu-tests")]
    fn gpu_frame_negative_offset_slice() {
        let df = df!("x" => [10i32, 20, 30, 40, 50]).unwrap();
        let gpu_df = GpuDataFrame::from_polars(&df).unwrap();
        // Negative offset: last 2 rows
        let sliced = gpu_df.slice(-2, 2).unwrap();
        assert_eq!(sliced.height(), 2);
        let back = sliced.to_polars().unwrap();
        let vals: Vec<i32> = back.column("x").unwrap().i32().unwrap()
            .into_no_null_iter().collect();
        assert_eq!(vals, vec![40, 50]);
    }

    // ── Expression evaluation tests ──

    #[test]
    #[cfg(feature = "gpu-tests")]
    fn expr_binary_add() {
        use polars_plan::plans::AExpr;
        use polars_plan::dsl::Operator;
        use polars_utils::arena::Arena;

        let df = df!("a" => [1i32, 2, 3], "b" => [10i32, 20, 30]).unwrap();
        let gpu_df = GpuDataFrame::from_polars(&df).unwrap();

        let mut arena = Arena::new();
        let left = arena.add(AExpr::Column("a".into()));
        let right = arena.add(AExpr::Column("b".into()));
        let add = arena.add(AExpr::BinaryExpr {
            left,
            op: Operator::Plus,
            right,
        });

        let result = gpu_expr::eval_expr(add, &arena, &gpu_df).unwrap();
        let vals: Vec<i32> = gpu_error::gpu_result(result.to_vec()).unwrap();
        assert_eq!(vals, vec![11, 22, 33]);
    }

    #[test]
    #[cfg(feature = "gpu-tests")]
    fn expr_comparison() {
        use polars_plan::plans::AExpr;
        use polars_plan::plans::LiteralValue;
        use polars_plan::dsl::Operator;
        use polars_utils::arena::Arena;

        let df = df!("x" => [1i32, 2, 3, 4, 5]).unwrap();
        let gpu_df = GpuDataFrame::from_polars(&df).unwrap();

        let mut arena = Arena::new();
        let col_node = arena.add(AExpr::Column("x".into()));
        let lit_node = arena.add(AExpr::Literal(LiteralValue::Int32(3)));
        let cmp = arena.add(AExpr::BinaryExpr {
            left: col_node,
            op: Operator::Gt,
            right: lit_node,
        });

        let result = gpu_expr::eval_expr(cmp, &arena, &gpu_df).unwrap();
        let vals: Vec<bool> = gpu_error::gpu_result(result.to_vec()).unwrap();
        assert_eq!(vals, vec![false, false, false, true, true]);
    }

    #[test]
    #[cfg(feature = "gpu-tests")]
    fn expr_cast() {
        use polars_plan::plans::AExpr;
        use polars_core::prelude::DataType;
        use polars_core::chunked_array::cast::CastOptions;
        use polars_utils::arena::Arena;

        let df = df!("x" => [1i32, 2, 3]).unwrap();
        let gpu_df = GpuDataFrame::from_polars(&df).unwrap();

        let mut arena = Arena::new();
        let col_node = arena.add(AExpr::Column("x".into()));
        let cast_node = arena.add(AExpr::Cast {
            expr: col_node,
            dtype: DataType::Float64,
            options: CastOptions::NonStrict,
        });

        let result = gpu_expr::eval_expr(cast_node, &arena, &gpu_df).unwrap();
        let vals: Vec<f64> = gpu_error::gpu_result(result.to_vec()).unwrap();
        assert_eq!(vals, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    #[cfg(feature = "gpu-tests")]
    fn expr_len() {
        use polars_plan::plans::AExpr;
        use polars_utils::arena::Arena;

        let df = df!("x" => [1i32, 2, 3, 4, 5]).unwrap();
        let gpu_df = GpuDataFrame::from_polars(&df).unwrap();

        let mut arena = Arena::new();
        let len_node = arena.add(AExpr::Len);

        let result = gpu_expr::eval_expr(len_node, &arena, &gpu_df).unwrap();
        let vals: Vec<u32> = gpu_error::gpu_result(result.to_vec()).unwrap();
        assert_eq!(vals, vec![5u32]);
    }
}
