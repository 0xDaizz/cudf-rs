//! GPU execution engine for Polars using NVIDIA libcudf.

pub mod convert;
pub mod engine;
pub mod error;
pub mod expr;
pub mod gpu_frame;
pub mod types;

#[cfg(feature = "lazy")]
pub use engine::collect_gpu;
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
            assert!(
                (o - r).abs() < f64::EPSILON,
                "mismatch at index {}: {} vs {}",
                i,
                o,
                r
            );
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
            assert_eq!(
                orig_id.get(i),
                result_id.get(i),
                "id mismatch at index {}",
                i
            );
        }
        let orig_val = df.column("value").unwrap().f64().unwrap();
        let result_val = back.column("value").unwrap().f64().unwrap();
        for i in 0..orig_val.len() {
            let o = orig_val.get(i).unwrap();
            let r = result_val.get(i).unwrap();
            assert!(
                (o - r).abs() < f64::EPSILON,
                "value mismatch at index {}: {} vs {}",
                i,
                o,
                r
            );
        }
        let orig_name = df.column("name").unwrap().str().unwrap();
        let result_name = back.column("name").unwrap().str().unwrap();
        for i in 0..orig_name.len() {
            assert_eq!(
                orig_name.get(i),
                result_name.get(i),
                "name mismatch at index {}",
                i
            );
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
    use crate::error as gpu_error;
    use crate::expr as gpu_expr;
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
        )
        .unwrap();
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
        let mask = gpu_error::gpu_result(x_col.binary_op_scalar(
            &threshold,
            cudf::BinaryOp::Greater,
            cudf::types::DataType::new(cudf::types::TypeId::Bool8),
        ))
        .unwrap();

        let filtered = gpu_df.apply_boolean_mask(&mask).unwrap();
        assert_eq!(filtered.height(), 3);
        let back = filtered.to_polars().unwrap();
        let vals: Vec<i32> = back
            .column("x")
            .unwrap()
            .i32()
            .unwrap()
            .into_no_null_iter()
            .collect();
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
        let vals: Vec<i32> = back
            .column("x")
            .unwrap()
            .i32()
            .unwrap()
            .into_no_null_iter()
            .collect();
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
        let vals: Vec<i32> = back
            .column("x")
            .unwrap()
            .i32()
            .unwrap()
            .into_no_null_iter()
            .collect();
        assert_eq!(vals, vec![40, 50]);
    }

    // ── Expression evaluation tests ──

    #[test]
    #[cfg(feature = "gpu-tests")]
    fn expr_binary_add() {
        use polars_plan::dsl::Operator;
        use polars_plan::plans::AExpr;
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
        use polars_plan::dsl::Operator;
        use polars_plan::plans::AExpr;
        use polars_plan::plans::{DynLiteralValue, LiteralValue};
        use polars_utils::arena::Arena;

        let df = df!("x" => [1i32, 2, 3, 4, 5]).unwrap();
        let gpu_df = GpuDataFrame::from_polars(&df).unwrap();

        let mut arena = Arena::new();
        let col_node = arena.add(AExpr::Column("x".into()));
        let lit_node = arena.add(AExpr::Literal(LiteralValue::Dyn(DynLiteralValue::Int(3))));
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
        use polars_core::chunked_array::cast::CastOptions;
        use polars_core::prelude::DataType;
        use polars_plan::plans::AExpr;
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
        assert_eq!(vals, vec![5u32; 5]);
    }

    // ── M3: Sort tests ──

    #[test]
    #[cfg(feature = "gpu-tests")]
    fn gpu_frame_sort_ascending() {
        use cudf::sorting::{NullOrder, SortOrder};

        let df = df!("x" => [3i32, 1, 4, 1, 5]).unwrap();
        let gpu_df = GpuDataFrame::from_polars(&df).unwrap();

        let key = gpu_df.column_by_name("x").unwrap();
        let sorted = gpu_df
            .sort_by_key(vec![key], &[SortOrder::Ascending], &[NullOrder::After])
            .unwrap();

        let back = sorted.to_polars().unwrap();
        let vals: Vec<i32> = back
            .column("x")
            .unwrap()
            .i32()
            .unwrap()
            .into_no_null_iter()
            .collect();
        assert_eq!(vals, vec![1, 1, 3, 4, 5]);
    }

    #[test]
    #[cfg(feature = "gpu-tests")]
    fn gpu_frame_sort_descending() {
        use cudf::sorting::{NullOrder, SortOrder};

        let df = df!("x" => [3i32, 1, 4, 1, 5]).unwrap();
        let gpu_df = GpuDataFrame::from_polars(&df).unwrap();

        let key = gpu_df.column_by_name("x").unwrap();
        let sorted = gpu_df
            .sort_by_key(vec![key], &[SortOrder::Descending], &[NullOrder::After])
            .unwrap();

        let back = sorted.to_polars().unwrap();
        let vals: Vec<i32> = back
            .column("x")
            .unwrap()
            .i32()
            .unwrap()
            .into_no_null_iter()
            .collect();
        assert_eq!(vals, vec![5, 4, 3, 1, 1]);
    }

    // ── M3: GroupBy tests ──

    #[test]
    #[cfg(feature = "gpu-tests")]
    fn gpu_frame_groupby_sum() {
        use cudf::aggregation::AggregationKind;

        let df = df!(
            "cat" => [1i32, 1, 2, 2, 3],
            "val" => [10.0f64, 20.0, 30.0, 40.0, 50.0]
        )
        .unwrap();
        let gpu_df = GpuDataFrame::from_polars(&df).unwrap();

        let key_col = gpu_df.column_by_name("cat").unwrap();
        let val_col = gpu_df.column_by_name("val").unwrap();

        let result = gpu_df
            .groupby(
                vec![key_col],
                vec!["cat".to_string()],
                vec![val_col],
                vec![(0, AggregationKind::Sum)],
                vec!["val_sum".to_string()],
                true, // maintain_order
            )
            .unwrap();

        assert_eq!(result.height(), 3); // 3 groups
        let back = result.to_polars().unwrap();
        assert_eq!(back.width(), 2); // cat + val_sum

        // Verify actual sum values (sorted by key)
        let cats: Vec<i32> = back
            .column("cat")
            .unwrap()
            .i32()
            .unwrap()
            .into_no_null_iter()
            .collect();
        let sums: Vec<f64> = back
            .column("val_sum")
            .unwrap()
            .f64()
            .unwrap()
            .into_no_null_iter()
            .collect();
        assert_eq!(cats, vec![1, 2, 3]);
        assert_eq!(sums, vec![30.0, 70.0, 50.0]);
    }

    #[test]
    #[cfg(feature = "gpu-tests")]
    fn gpu_frame_groupby_mean_count() {
        use cudf::aggregation::AggregationKind;

        let df = df!(
            "grp" => [1i32, 1, 2, 2],
            "a" => [10.0f64, 20.0, 30.0, 40.0],
            "b" => [100i32, 200, 300, 400]
        )
        .unwrap();
        let gpu_df = GpuDataFrame::from_polars(&df).unwrap();

        let key_col = gpu_df.column_by_name("grp").unwrap();
        let a_col = gpu_df.column_by_name("a").unwrap();
        let b_col = gpu_df.column_by_name("b").unwrap();

        let result = gpu_df
            .groupby(
                vec![key_col],
                vec!["grp".to_string()],
                vec![a_col, b_col],
                vec![(0, AggregationKind::Mean), (1, AggregationKind::Count)],
                vec!["a_mean".to_string(), "b_count".to_string()],
                false, // maintain_order
            )
            .unwrap();

        assert_eq!(result.height(), 2); // 2 groups
        assert_eq!(result.width(), 3); // grp + a_mean + b_count
    }

    // ── M3: Distinct tests ──

    #[test]
    #[cfg(feature = "gpu-tests")]
    fn gpu_frame_distinct_all_columns() {
        use cudf::stream_compaction::DuplicateKeepOption;

        let df = df!(
            "x" => [1i32, 2, 2, 3, 3, 3],
            "y" => [10i32, 20, 20, 30, 30, 30]
        )
        .unwrap();
        let gpu_df = GpuDataFrame::from_polars(&df).unwrap();

        let result = gpu_df
            .distinct(None, DuplicateKeepOption::First, false)
            .unwrap();
        assert_eq!(result.height(), 3); // 3 unique rows

        // Verify actual distinct values (sort by x to handle non-deterministic order)
        let x_col = result.column_by_name("x").unwrap();
        let sorted = result
            .sort_by_key(
                vec![x_col],
                &[cudf::sorting::SortOrder::Ascending],
                &[cudf::sorting::NullOrder::After],
            )
            .unwrap();
        let back = sorted.to_polars().unwrap();
        let xs: Vec<i32> = back
            .column("x")
            .unwrap()
            .i32()
            .unwrap()
            .into_no_null_iter()
            .collect();
        let ys: Vec<i32> = back
            .column("y")
            .unwrap()
            .i32()
            .unwrap()
            .into_no_null_iter()
            .collect();
        assert_eq!(xs, vec![1, 2, 3]);
        assert_eq!(ys, vec![10, 20, 30]);
    }

    #[test]
    #[cfg(feature = "gpu-tests")]
    fn gpu_frame_distinct_subset() {
        use cudf::stream_compaction::DuplicateKeepOption;

        let df = df!(
            "x" => [1i32, 1, 2, 2],
            "y" => [10i32, 20, 30, 40]
        )
        .unwrap();
        let gpu_df = GpuDataFrame::from_polars(&df).unwrap();

        let result = gpu_df
            .distinct(Some(&["x"]), DuplicateKeepOption::First, true)
            .unwrap();
        assert_eq!(result.height(), 2); // 2 unique x values
        assert_eq!(result.width(), 2); // both columns preserved
    }
}

#[cfg(test)]
mod m4_tests {
    use crate::error as gpu_error;
    use crate::gpu_frame::GpuDataFrame;
    use polars_core::prelude::*;

    // ── Join tests ──

    #[test]
    #[cfg(feature = "gpu-tests")]
    fn test_inner_join() {
        // Left: id=[1,2,3,4], val=[10,20,30,40]
        // Right: id=[2,3,5], score=[200,300,500]
        // Inner join on id: expect matches at id=2,3
        let left_df = df!(
            "id" => [1i32, 2, 3, 4],
            "val" => [10i32, 20, 30, 40]
        )
        .unwrap();
        let right_df = df!(
            "id" => [2i32, 3, 5],
            "score" => [200i32, 300, 500]
        )
        .unwrap();

        let left_gpu = GpuDataFrame::from_polars(&left_df).unwrap();
        let right_gpu = GpuDataFrame::from_polars(&right_df).unwrap();

        // Build key tables
        let left_keys = vec![left_gpu.column_by_name("id").unwrap()];
        let right_keys = vec![right_gpu.column_by_name("id").unwrap()];
        let left_keys_table = gpu_error::gpu_result(cudf::Table::new(left_keys)).unwrap();
        let right_keys_table = gpu_error::gpu_result(cudf::Table::new(right_keys)).unwrap();

        let result = gpu_error::gpu_result(left_keys_table.inner_join(&right_keys_table)).unwrap();
        let left_gathered =
            gpu_error::gpu_result(left_gpu.inner_table().gather(&result.left_indices)).unwrap();
        let right_gathered =
            gpu_error::gpu_result(right_gpu.inner_table().gather(&result.right_indices)).unwrap();

        // Build combined result
        let mut cols = Vec::new();
        let mut names = Vec::new();
        for i in 0..left_gathered.num_columns() {
            cols.push(gpu_error::gpu_result(left_gathered.column(i)).unwrap());
            names.push(left_gpu.names()[i].clone());
        }
        for i in 0..right_gathered.num_columns() {
            let rname = &right_gpu.names()[i];
            if names.contains(rname) {
                names.push(format!("{}_right", rname));
            } else {
                names.push(rname.clone());
            }
            cols.push(gpu_error::gpu_result(right_gathered.column(i)).unwrap());
        }

        let joined = GpuDataFrame::from_columns(cols, names).unwrap();
        assert_eq!(joined.height(), 2); // 2 matching rows

        let back = joined.to_polars().unwrap();
        // Sort by id for deterministic check
        let back = back.sort(["id"], Default::default()).unwrap();
        let ids: Vec<i32> = back
            .column("id")
            .unwrap()
            .i32()
            .unwrap()
            .into_no_null_iter()
            .collect();
        let vals: Vec<i32> = back
            .column("val")
            .unwrap()
            .i32()
            .unwrap()
            .into_no_null_iter()
            .collect();
        let scores: Vec<i32> = back
            .column("score")
            .unwrap()
            .i32()
            .unwrap()
            .into_no_null_iter()
            .collect();
        assert_eq!(ids, vec![2, 3]);
        assert_eq!(vals, vec![20, 30]);
        assert_eq!(scores, vec![200, 300]);
    }

    #[test]
    #[cfg(feature = "gpu-tests")]
    fn test_left_join() {
        let left_df = df!(
            "id" => [1i32, 2, 3, 4],
            "val" => [10i32, 20, 30, 40]
        )
        .unwrap();
        let right_df = df!(
            "id" => [2i32, 3, 5],
            "score" => [200i32, 300, 500]
        )
        .unwrap();

        let left_gpu = GpuDataFrame::from_polars(&left_df).unwrap();
        let right_gpu = GpuDataFrame::from_polars(&right_df).unwrap();

        let left_keys = vec![left_gpu.column_by_name("id").unwrap()];
        let right_keys = vec![right_gpu.column_by_name("id").unwrap()];
        let left_keys_table = gpu_error::gpu_result(cudf::Table::new(left_keys)).unwrap();
        let right_keys_table = gpu_error::gpu_result(cudf::Table::new(right_keys)).unwrap();

        let result = gpu_error::gpu_result(left_keys_table.left_join(&right_keys_table)).unwrap();
        let left_gathered =
            gpu_error::gpu_result(left_gpu.inner_table().gather(&result.left_indices)).unwrap();
        let right_gathered =
            gpu_error::gpu_result(right_gpu.inner_table().gather(&result.right_indices)).unwrap();

        // Left join: all 4 left rows preserved
        assert_eq!(left_gathered.num_rows(), 4);
        assert_eq!(right_gathered.num_rows(), 4);

        // Verify actual values: combine left+right gathered, then sort by id
        let left_result_df = GpuDataFrame::from_table(left_gathered, left_gpu.names().to_vec());
        let right_result_df = GpuDataFrame::from_table(right_gathered, right_gpu.names().to_vec());
        let left_back = left_result_df.to_polars().unwrap();
        let right_back = right_result_df.to_polars().unwrap();

        // Combine left and right columns into one DataFrame, then sort
        let mut combined = left_back.clone();
        combined
            .with_column(right_back.column("score").unwrap().clone())
            .unwrap();
        let combined = combined.sort(["id"], Default::default()).unwrap();

        let ids: Vec<i32> = combined
            .column("id")
            .unwrap()
            .i32()
            .unwrap()
            .into_no_null_iter()
            .collect();
        assert_eq!(ids, vec![1, 2, 3, 4]); // all left rows preserved
        let vals: Vec<i32> = combined
            .column("val")
            .unwrap()
            .i32()
            .unwrap()
            .into_no_null_iter()
            .collect();
        assert_eq!(vals, vec![10, 20, 30, 40]);
        let scores: Vec<Option<i32>> = combined
            .column("score")
            .unwrap()
            .i32()
            .unwrap()
            .into_iter()
            .collect();
        // id=1 -> None, id=2 -> 200, id=3 -> 300, id=4 -> None
        assert_eq!(scores, vec![None, Some(200), Some(300), None]);
    }

    #[test]
    #[cfg(feature = "gpu-tests")]
    fn test_left_semi_join() {
        let left_df = df!(
            "id" => [1i32, 2, 3, 4],
            "val" => [10i32, 20, 30, 40]
        )
        .unwrap();
        let right_df = df!(
            "id" => [2i32, 3, 5]
        )
        .unwrap();

        let left_gpu = GpuDataFrame::from_polars(&left_df).unwrap();
        let right_gpu = GpuDataFrame::from_polars(&right_df).unwrap();

        let left_keys = vec![left_gpu.column_by_name("id").unwrap()];
        let right_keys = vec![right_gpu.column_by_name("id").unwrap()];
        let left_keys_table = gpu_error::gpu_result(cudf::Table::new(left_keys)).unwrap();
        let right_keys_table = gpu_error::gpu_result(cudf::Table::new(right_keys)).unwrap();

        let result =
            gpu_error::gpu_result(left_keys_table.left_semi_join(&right_keys_table)).unwrap();
        let gathered =
            gpu_error::gpu_result(left_gpu.inner_table().gather(&result.left_indices)).unwrap();
        let gathered_df = GpuDataFrame::from_table(gathered, left_gpu.names().to_vec());

        // Sort for deterministic output
        let id_col = gathered_df.column_by_name("id").unwrap();
        let sorted = gathered_df
            .sort_by_key(
                vec![id_col],
                &[cudf::sorting::SortOrder::Ascending],
                &[cudf::sorting::NullOrder::After],
            )
            .unwrap();

        assert_eq!(sorted.height(), 2); // only rows with id 2, 3
        let back = sorted.to_polars().unwrap();
        let ids: Vec<i32> = back
            .column("id")
            .unwrap()
            .i32()
            .unwrap()
            .into_no_null_iter()
            .collect();
        assert_eq!(ids, vec![2, 3]);
    }

    #[test]
    #[cfg(feature = "gpu-tests")]
    fn test_left_anti_join() {
        let left_df = df!(
            "id" => [1i32, 2, 3, 4],
            "val" => [10i32, 20, 30, 40]
        )
        .unwrap();
        let right_df = df!(
            "id" => [2i32, 3, 5]
        )
        .unwrap();

        let left_gpu = GpuDataFrame::from_polars(&left_df).unwrap();
        let right_gpu = GpuDataFrame::from_polars(&right_df).unwrap();

        let left_keys = vec![left_gpu.column_by_name("id").unwrap()];
        let right_keys = vec![right_gpu.column_by_name("id").unwrap()];
        let left_keys_table = gpu_error::gpu_result(cudf::Table::new(left_keys)).unwrap();
        let right_keys_table = gpu_error::gpu_result(cudf::Table::new(right_keys)).unwrap();

        let result =
            gpu_error::gpu_result(left_keys_table.left_anti_join(&right_keys_table)).unwrap();
        let gathered =
            gpu_error::gpu_result(left_gpu.inner_table().gather(&result.left_indices)).unwrap();
        let gathered_df = GpuDataFrame::from_table(gathered, left_gpu.names().to_vec());

        let id_col = gathered_df.column_by_name("id").unwrap();
        let sorted = gathered_df
            .sort_by_key(
                vec![id_col],
                &[cudf::sorting::SortOrder::Ascending],
                &[cudf::sorting::NullOrder::After],
            )
            .unwrap();

        assert_eq!(sorted.height(), 2); // rows with id 1, 4 (not in right)
        let back = sorted.to_polars().unwrap();
        let ids: Vec<i32> = back
            .column("id")
            .unwrap()
            .i32()
            .unwrap()
            .into_no_null_iter()
            .collect();
        assert_eq!(ids, vec![1, 4]);
    }

    #[test]
    #[cfg(feature = "gpu-tests")]
    fn test_cross_join() {
        let left_df = df!(
            "a" => [1i32, 2]
        )
        .unwrap();
        let right_df = df!(
            "b" => [10i32, 20, 30]
        )
        .unwrap();

        let left_gpu = GpuDataFrame::from_polars(&left_df).unwrap();
        let right_gpu = GpuDataFrame::from_polars(&right_df).unwrap();

        let cross =
            gpu_error::gpu_result(left_gpu.inner_table().cross_join(right_gpu.inner_table()))
                .unwrap();
        assert_eq!(cross.num_rows(), 6); // 2 * 3
        assert_eq!(cross.num_columns(), 2); // a, b

        // Verify actual values
        let names = vec!["a".to_string(), "b".to_string()];
        let result_df = GpuDataFrame::from_table(cross, names);
        let back = result_df.to_polars().unwrap();
        let back = back.sort(["a", "b"], Default::default()).unwrap();
        let a_vals: Vec<i32> = back
            .column("a")
            .unwrap()
            .i32()
            .unwrap()
            .into_no_null_iter()
            .collect();
        let b_vals: Vec<i32> = back
            .column("b")
            .unwrap()
            .i32()
            .unwrap()
            .into_no_null_iter()
            .collect();
        // Cross join: each left row x each right row, sorted by (a, b)
        assert_eq!(a_vals, vec![1, 1, 1, 2, 2, 2]);
        assert_eq!(b_vals, vec![10, 20, 30, 10, 20, 30]);
    }

    // ── Union (vertical concat) test ──

    #[test]
    #[cfg(feature = "gpu-tests")]
    fn test_union_concat() {
        let df1 = df!(
            "x" => [1i32, 2, 3],
            "y" => [10i32, 20, 30]
        )
        .unwrap();
        let df2 = df!(
            "x" => [4i32, 5],
            "y" => [40i32, 50]
        )
        .unwrap();

        let gpu1 = GpuDataFrame::from_polars(&df1).unwrap();
        let gpu2 = GpuDataFrame::from_polars(&df2).unwrap();

        let table_refs = vec![gpu1.inner_table(), gpu2.inner_table()];
        let concatenated =
            gpu_error::gpu_result(cudf::concatenate::concatenate_tables(&table_refs)).unwrap();
        let result = GpuDataFrame::from_table(concatenated, gpu1.names().to_vec());

        assert_eq!(result.height(), 5);
        assert_eq!(result.width(), 2);

        let back = result.to_polars().unwrap();
        let xs: Vec<i32> = back
            .column("x")
            .unwrap()
            .i32()
            .unwrap()
            .into_no_null_iter()
            .collect();
        let ys: Vec<i32> = back
            .column("y")
            .unwrap()
            .i32()
            .unwrap()
            .into_no_null_iter()
            .collect();
        assert_eq!(xs, vec![1, 2, 3, 4, 5]);
        assert_eq!(ys, vec![10, 20, 30, 40, 50]);
    }

    // ── HConcat (horizontal concat) test ──

    #[test]
    #[cfg(feature = "gpu-tests")]
    fn test_hconcat() {
        let df1 = df!("a" => [1i32, 2, 3]).unwrap();
        let df2 = df!("b" => [10i32, 20, 30]).unwrap();

        let gpu1 = GpuDataFrame::from_polars(&df1).unwrap();
        let gpu2 = GpuDataFrame::from_polars(&df2).unwrap();

        // Combine columns
        let mut all_cols = Vec::new();
        let mut all_names = Vec::new();
        for i in 0..gpu1.width() {
            all_cols.push(gpu1.column(i).unwrap());
            all_names.push(gpu1.names()[i].clone());
        }
        for i in 0..gpu2.width() {
            all_cols.push(gpu2.column(i).unwrap());
            all_names.push(gpu2.names()[i].clone());
        }

        let combined = GpuDataFrame::from_columns(all_cols, all_names).unwrap();
        assert_eq!(combined.width(), 2);
        assert_eq!(combined.height(), 3);

        let back = combined.to_polars().unwrap();
        let a_vals: Vec<i32> = back
            .column("a")
            .unwrap()
            .i32()
            .unwrap()
            .into_no_null_iter()
            .collect();
        let b_vals: Vec<i32> = back
            .column("b")
            .unwrap()
            .i32()
            .unwrap()
            .into_no_null_iter()
            .collect();
        assert_eq!(a_vals, vec![1, 2, 3]);
        assert_eq!(b_vals, vec![10, 20, 30]);
    }

    // ── Ternary (if-then-else / copy_if_else) test ──

    #[test]
    #[cfg(feature = "gpu-tests")]
    fn test_ternary_copy_if_else() {
        // mask = [true, false, true, false, true]
        // truthy = [10, 20, 30, 40, 50]
        // falsy = [100, 200, 300, 400, 500]
        // result = [10, 200, 30, 400, 50]
        let mask =
            gpu_error::gpu_result(cudf::Column::from_slice(&[true, false, true, false, true]))
                .unwrap();
        let truthy =
            gpu_error::gpu_result(cudf::Column::from_slice(&[10i32, 20, 30, 40, 50])).unwrap();
        let falsy =
            gpu_error::gpu_result(cudf::Column::from_slice(&[100i32, 200, 300, 400, 500])).unwrap();

        let result = gpu_error::gpu_result(truthy.copy_if_else(&falsy, &mask)).unwrap();
        let vals: Vec<i32> = gpu_error::gpu_result(result.to_vec()).unwrap();
        assert_eq!(vals, vec![10, 200, 30, 400, 50]);
    }

    // ── Function tests (IsNull, IsNotNull, IsNan, Abs) ──

    #[test]
    #[cfg(feature = "gpu-tests")]
    fn test_is_null() {
        let opts: Vec<Option<i32>> = vec![Some(1), None, Some(3), None, Some(5)];
        let col = gpu_error::gpu_result(cudf::Column::from_optional_i32(&opts)).unwrap();
        let result = gpu_error::gpu_result(col.is_null()).unwrap();
        let vals: Vec<bool> = gpu_error::gpu_result(result.to_vec()).unwrap();
        assert_eq!(vals, vec![false, true, false, true, false]);
    }

    #[test]
    #[cfg(feature = "gpu-tests")]
    fn test_is_valid() {
        let opts: Vec<Option<i32>> = vec![Some(1), None, Some(3), None, Some(5)];
        let col = gpu_error::gpu_result(cudf::Column::from_optional_i32(&opts)).unwrap();
        let result = gpu_error::gpu_result(col.is_valid()).unwrap();
        let vals: Vec<bool> = gpu_error::gpu_result(result.to_vec()).unwrap();
        assert_eq!(vals, vec![true, false, true, false, true]);
    }

    #[test]
    #[cfg(feature = "gpu-tests")]
    fn test_abs() {
        use cudf::unary::UnaryOp;

        let col = gpu_error::gpu_result(cudf::Column::from_slice(&[-3i32, -1, 0, 2, 5])).unwrap();
        let result = gpu_error::gpu_result(col.unary_op(UnaryOp::Abs)).unwrap();
        let vals: Vec<i32> = gpu_error::gpu_result(result.to_vec()).unwrap();
        assert_eq!(vals, vec![3, 1, 0, 2, 5]);
    }

    #[test]
    #[cfg(feature = "gpu-tests")]
    fn test_not() {
        use cudf::unary::UnaryOp;

        let col = gpu_error::gpu_result(cudf::Column::from_slice(&[true, false, true])).unwrap();
        let result = gpu_error::gpu_result(col.unary_op(UnaryOp::Not)).unwrap();
        let vals: Vec<bool> = gpu_error::gpu_result(result.to_vec()).unwrap();
        assert_eq!(vals, vec![false, true, false]);
    }
}
