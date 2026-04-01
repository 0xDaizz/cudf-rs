//! GPU execution engine: walks the IR tree and executes nodes on GPU.

use polars_core::prelude::*;
use polars_error::{PolarsResult, polars_bail};
use polars_plan::plans::{AExpr, IR, IRAggExpr, IRPlan};
use polars_utils::arena::{Arena, Node};

use cudf::aggregation::AggregationKind;
use cudf::sorting::{NullOrder, SortOrder};
use cudf::stream_compaction::DuplicateKeepOption;

use polars_ops::prelude::JoinType;

use crate::error::gpu_result;
use crate::expr;
use crate::gpu_frame::GpuDataFrame;

/// Execute an IR node recursively, producing a GPU-resident data frame.
pub fn execute_node(
    node: Node,
    lp_arena: &Arena<IR>,
    expr_arena: &Arena<AExpr>,
) -> PolarsResult<GpuDataFrame> {
    match lp_arena.get(node) {
        IR::DataFrameScan {
            df, output_schema, ..
        } => {
            let gpu_df = GpuDataFrame::from_polars(df)?;
            // If there's an output_schema (projection), apply it
            if let Some(schema) = output_schema {
                let names: Vec<&str> = schema.iter_names().map(|n| n.as_str()).collect();
                gpu_df.select_columns(&names)
            } else {
                Ok(gpu_df)
            }
        }

        IR::Filter { input, predicate } => {
            let input_node = *input;
            let pred_node = predicate.node();
            let table = execute_node(input_node, lp_arena, expr_arena)?;
            let mask = expr::eval_expr(pred_node, expr_arena, &table)?;
            table.apply_boolean_mask(&mask)
        }

        IR::Select {
            input, expr: exprs, ..
        } => {
            let input_node = *input;
            let exprs = exprs.clone();
            let table = execute_node(input_node, lp_arena, expr_arena)?;

            let mut columns = Vec::with_capacity(exprs.len());
            let mut names = Vec::with_capacity(exprs.len());
            for e in &exprs {
                let col = expr::eval_expr(e.node(), expr_arena, &table)?;
                columns.push(col);
                names.push(e.output_name().to_string());
            }
            GpuDataFrame::from_columns(columns, names)
        }

        IR::SimpleProjection { input, columns } => {
            let input_node = *input;
            let col_names: Vec<&str> = columns.iter_names().map(|n| n.as_str()).collect();
            let table = execute_node(input_node, lp_arena, expr_arena)?;
            table.select_columns(&col_names)
        }

        IR::Slice { input, offset, len } => {
            let input_node = *input;
            let offset = *offset;
            let len = *len as usize;
            let table = execute_node(input_node, lp_arena, expr_arena)?;
            table.slice(offset, len)
        }

        IR::HStack {
            input,
            exprs,
            schema,
            ..
        } => {
            let input_node = *input;
            let exprs = exprs.clone();
            let schema = schema.clone();
            let table = execute_node(input_node, lp_arena, expr_arena)?;

            // HStack adds new columns to the existing frame.
            // Use Option<Column> to allow zero-copy reordering without dummy GPU allocations.
            let existing_width = table.width();
            let mut all_columns: Vec<Option<cudf::Column>> =
                Vec::with_capacity(existing_width + exprs.len());
            let mut all_names = Vec::with_capacity(existing_width + exprs.len());

            for i in 0..existing_width {
                all_columns.push(Some(table.column(i)?));
                all_names.push(table.names()[i].clone());
            }

            for e in &exprs {
                let col = expr::eval_expr(e.node(), expr_arena, &table)?;
                let name = e.output_name().to_string();

                if let Some(pos) = all_names.iter().position(|n| n == &name) {
                    all_columns[pos] = Some(col);
                } else {
                    all_columns.push(Some(col));
                    all_names.push(name);
                }
            }

            // Reorder to match the output schema
            let schema_names: Vec<&str> = schema.iter_names().map(|n| n.as_str()).collect();
            let mut ordered_columns = Vec::with_capacity(schema_names.len());
            let mut ordered_names = Vec::with_capacity(schema_names.len());
            for &sn in &schema_names {
                if let Some(pos) = all_names.iter().position(|n| n == sn) {
                    let col = all_columns[pos].take().ok_or_else(|| {
                        polars_err!(ColumnNotFound: "duplicate reference to column '{}' in HStack schema", sn)
                    })?;
                    ordered_columns.push(col);
                    ordered_names.push(sn.to_string());
                } else {
                    polars_bail!(ColumnNotFound: "{}", sn);
                }
            }

            GpuDataFrame::from_columns(ordered_columns, ordered_names)
        }

        IR::Sort {
            input,
            by_column,
            slice,
            sort_options,
        } => {
            let input_node = *input;
            let by_column = by_column.clone();
            let slice = *slice;
            let sort_options = sort_options.clone();
            let table = execute_node(input_node, lp_arena, expr_arena)?;

            // Evaluate sort key expressions
            let sort_keys: Vec<cudf::Column> = by_column
                .iter()
                .map(|e| expr::eval_expr(e.node(), expr_arena, &table))
                .collect::<PolarsResult<_>>()?;

            let ncols = sort_keys.len();

            // Build sort orders, broadcasting single-element vecs
            let orders: Vec<SortOrder> = if sort_options.descending.len() == 1 {
                vec![
                    if sort_options.descending[0] {
                        SortOrder::Descending
                    } else {
                        SortOrder::Ascending
                    };
                    ncols
                ]
            } else {
                sort_options
                    .descending
                    .iter()
                    .map(|d| {
                        if *d {
                            SortOrder::Descending
                        } else {
                            SortOrder::Ascending
                        }
                    })
                    .collect()
            };

            let null_orders: Vec<NullOrder> = if sort_options.nulls_last.len() == 1 {
                vec![
                    if sort_options.nulls_last[0] {
                        NullOrder::After
                    } else {
                        NullOrder::Before
                    };
                    ncols
                ]
            } else {
                sort_options
                    .nulls_last
                    .iter()
                    .map(|n| {
                        if *n {
                            NullOrder::After
                        } else {
                            NullOrder::Before
                        }
                    })
                    .collect()
            };

            let sorted = table.sort_by_key(sort_keys, &orders, &null_orders)?;

            // Apply optional slice
            if let Some((offset, len)) = slice {
                sorted.slice(offset, len)
            } else {
                Ok(sorted)
            }
        }

        IR::GroupBy {
            input,
            keys,
            aggs,
            schema,
            apply,
            maintain_order,
            ..
        } => {
            if apply.is_some() {
                polars_bail!(ComputeError: "GPU engine: custom apply in GroupBy not supported");
            }

            let input_node = *input;
            let keys = keys.clone();
            let aggs = aggs.clone();
            let schema = schema.clone();
            let table = execute_node(input_node, lp_arena, expr_arena)?;

            // Evaluate key expressions
            let mut key_columns = Vec::with_capacity(keys.len());
            let mut key_names = Vec::with_capacity(keys.len());
            for k in &keys {
                let col = expr::eval_expr(k.node(), expr_arena, &table)?;
                key_columns.push(col);
                key_names.push(k.output_name().to_string());
            }

            // Parse aggregation expressions
            let mut value_columns: Vec<cudf::Column> = Vec::new();
            let mut agg_requests: Vec<(usize, AggregationKind)> = Vec::new();
            let mut agg_names: Vec<String> = Vec::new();

            for agg_expr_ir in &aggs {
                let agg_name = agg_expr_ir.output_name().to_string();
                let (input_node, agg_kind) = extract_agg_info(agg_expr_ir.node(), expr_arena)?;

                // Evaluate the input column for this aggregation
                let input_col = expr::eval_expr(input_node, expr_arena, &table)?;

                // Always add as a new column entry
                let val_idx = value_columns.len();
                value_columns.push(input_col);

                agg_requests.push((val_idx, agg_kind));
                agg_names.push(agg_name);
            }

            let result = table.groupby(
                key_columns,
                key_names,
                value_columns,
                agg_requests,
                agg_names,
                *maintain_order,
            )?;

            // Reorder to match output schema if needed
            let schema_names: Vec<&str> = schema.iter_names().map(|n| n.as_str()).collect();
            let result_names = result.names().to_vec();
            if result_names.iter().map(|s| s.as_str()).collect::<Vec<_>>() == schema_names {
                Ok(result)
            } else {
                result.select_columns(&schema_names)
            }
        }

        IR::Distinct { input, options } => {
            let input_node = *input;
            let options = options.clone();
            let table = execute_node(input_node, lp_arena, expr_arena)?;

            let keep = match options.keep_strategy {
                UniqueKeepStrategy::First => DuplicateKeepOption::First,
                UniqueKeepStrategy::Last => DuplicateKeepOption::Last,
                UniqueKeepStrategy::None => DuplicateKeepOption::None,
                UniqueKeepStrategy::Any => DuplicateKeepOption::Any,
            };

            let subset: Option<Vec<&str>> = options
                .subset
                .as_ref()
                .map(|s| s.iter().map(|n| n.as_str()).collect());

            let result = table.distinct(subset.as_deref(), keep, options.maintain_order)?;

            // Apply optional slice
            if let Some((offset, len)) = options.slice {
                result.slice(offset, len)
            } else {
                Ok(result)
            }
        }

        IR::Scan {
            sources,
            scan_type,
            unified_scan_args,
            output_schema,
            predicate,
            ..
        } => {
            use polars_plan::dsl::FileScanIR;

            let unified_scan_args = unified_scan_args.clone();
            let output_schema = output_schema.clone();
            let predicate = predicate.clone();

            match scan_type.as_ref() {
                FileScanIR::Parquet { .. } => {
                    let paths = sources.as_paths().ok_or_else(
                        || polars_err!(ComputeError: "GPU engine: Scan requires file paths"),
                    )?;

                    if paths.is_empty() {
                        polars_bail!(ComputeError: "GPU engine: Scan has no source paths");
                    }

                    // Read the first (or only) parquet file
                    let path_str: String = AsRef::<str>::as_ref(&paths[0]).to_string();

                    // Determine which columns to read
                    let col_names: Vec<String> =
                        if let Some(ref projection) = unified_scan_args.projection {
                            projection.iter().map(|c| c.to_string()).collect()
                        } else {
                            vec![]
                        };

                    let mut reader = cudf::io::parquet::ParquetReader::new(&path_str);
                    if !col_names.is_empty() {
                        reader = reader.columns(col_names.clone());
                    }

                    // Apply row limit/skip from pre_slice
                    if let Some(ref slice) = unified_scan_args.pre_slice {
                        use polars_utils::slice_enum::Slice as SliceEnum;
                        match slice {
                            SliceEnum::Positive { offset, len } => {
                                if *offset > 0 {
                                    reader = reader.skip_rows(*offset);
                                }
                                reader = reader.num_rows(*len);
                            }
                            SliceEnum::Negative { .. } => {
                                // Negative offset (tail): read all, then slice at the end
                            }
                        }
                    }

                    let gpu_table = gpu_result(reader.read_with_metadata())?;
                    let names: Vec<String> = gpu_table.column_names.clone();
                    let gpu_df = GpuDataFrame::from_table(gpu_table.table, names);

                    // Apply column projection from output_schema
                    let gpu_df = if let Some(ref schema) = output_schema {
                        let proj_names: Vec<&str> =
                            schema.iter_names().map(|n| n.as_str()).collect();
                        gpu_df.select_columns(&proj_names)?
                    } else {
                        gpu_df
                    };

                    // Apply predicate pushdown if present
                    if let Some(ref pred) = predicate {
                        let mask = expr::eval_expr(pred.node(), expr_arena, &gpu_df)?;
                        gpu_df.apply_boolean_mask(&mask)
                    } else {
                        Ok(gpu_df)
                    }
                }
                _ => {
                    polars_bail!(ComputeError: "GPU engine: only Parquet scan is supported")
                }
            }
        }

        IR::Join {
            input_left,
            input_right,
            schema,
            left_on,
            right_on,
            options,
        } => {
            let input_left = *input_left;
            let input_right = *input_right;
            let schema = schema.clone();
            let left_on = left_on.clone();
            let right_on = right_on.clone();
            let options = options.clone();

            let left_table = execute_node(input_left, lp_arena, expr_arena)?;
            let right_table = execute_node(input_right, lp_arena, expr_arena)?;

            // Evaluate key expressions
            let left_keys: Vec<cudf::Column> = left_on
                .iter()
                .map(|e| expr::eval_expr(e.node(), expr_arena, &left_table))
                .collect::<PolarsResult<_>>()?;
            let right_keys: Vec<cudf::Column> = right_on
                .iter()
                .map(|e| expr::eval_expr(e.node(), expr_arena, &right_table))
                .collect::<PolarsResult<_>>()?;

            let left_keys_table = gpu_result(cudf::Table::new(left_keys))?;
            let right_keys_table = gpu_result(cudf::Table::new(right_keys))?;

            let join_type = &options.args.how;
            let suffix = options
                .args
                .suffix
                .as_deref()
                .unwrap_or("_right")
                .to_string();

            match join_type {
                JoinType::Inner => {
                    let result = gpu_result(left_keys_table.inner_join(&right_keys_table))?;
                    build_joined_table(
                        &left_table,
                        &right_table,
                        &result.left_indices,
                        &result.right_indices,
                        &schema,
                        &suffix,
                    )
                }
                JoinType::Left => {
                    let result = gpu_result(left_keys_table.left_join(&right_keys_table))?;
                    build_joined_table(
                        &left_table,
                        &right_table,
                        &result.left_indices,
                        &result.right_indices,
                        &schema,
                        &suffix,
                    )
                }
                JoinType::Full => {
                    let result = gpu_result(left_keys_table.full_join(&right_keys_table))?;
                    build_joined_table(
                        &left_table,
                        &right_table,
                        &result.left_indices,
                        &result.right_indices,
                        &schema,
                        &suffix,
                    )
                }
                JoinType::Semi => {
                    let result = gpu_result(left_keys_table.left_semi_join(&right_keys_table))?;
                    let gathered =
                        gpu_result(left_table.inner_table().gather(&result.left_indices))?;
                    let result_df = GpuDataFrame::from_table(gathered, left_table.names().to_vec());
                    // Reorder to match output schema
                    let schema_names: Vec<&str> = schema.iter_names().map(|n| n.as_str()).collect();
                    result_df.select_columns(&schema_names)
                }
                JoinType::Anti => {
                    let result = gpu_result(left_keys_table.left_anti_join(&right_keys_table))?;
                    let gathered =
                        gpu_result(left_table.inner_table().gather(&result.left_indices))?;
                    let result_df = GpuDataFrame::from_table(gathered, left_table.names().to_vec());
                    let schema_names: Vec<&str> = schema.iter_names().map(|n| n.as_str()).collect();
                    result_df.select_columns(&schema_names)
                }
                JoinType::Cross => {
                    let cross_result = gpu_result(
                        left_table
                            .inner_table()
                            .cross_join(right_table.inner_table()),
                    )?;
                    // Build names: left names + right names (with suffix for conflicts)
                    let mut all_names = Vec::new();
                    let left_name_set: std::collections::HashSet<&str> =
                        left_table.names().iter().map(|s| s.as_str()).collect();
                    for n in left_table.names() {
                        all_names.push(n.clone());
                    }
                    for n in right_table.names() {
                        if left_name_set.contains(n.as_str()) {
                            all_names.push(format!("{}{}", n, suffix));
                        } else {
                            all_names.push(n.clone());
                        }
                    }
                    let result_df = GpuDataFrame::from_table(cross_result, all_names);
                    let schema_names: Vec<&str> = schema.iter_names().map(|n| n.as_str()).collect();
                    result_df.select_columns(&schema_names)
                }
                _ => {
                    polars_bail!(ComputeError: "GPU engine: unsupported join type {:?}", join_type)
                }
            }
        }

        IR::Union { inputs, options } => {
            let inputs = inputs.clone();
            let options = *options;

            if inputs.is_empty() {
                polars_bail!(ComputeError: "GPU engine: Union with no inputs");
            }

            let tables: Vec<GpuDataFrame> = inputs
                .iter()
                .map(|node| execute_node(*node, lp_arena, expr_arena))
                .collect::<PolarsResult<_>>()?;

            // Use the first table's names as reference
            let names = tables[0].names().to_vec();

            // Reorder remaining tables' columns to match the first table's column order.
            // Different upstream branches may produce columns in different orders.
            let mut tables = tables;
            let ref_names: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
            for table in tables.iter_mut().skip(1) {
                *table = table.select_columns(&ref_names)?;
            }

            let table_refs: Vec<&cudf::Table> = tables.iter().map(|t| t.inner_table()).collect();
            let concatenated = gpu_result(cudf::concatenate::concatenate_tables(&table_refs))?;
            let result = GpuDataFrame::from_table(concatenated, names);

            // Apply optional slice
            if let Some((offset, len)) = options.slice {
                result.slice(offset, len)
            } else {
                Ok(result)
            }
        }

        IR::HConcat { inputs, schema, .. } => {
            let inputs = inputs.clone();
            let schema = schema.clone();

            let tables: Vec<GpuDataFrame> = inputs
                .iter()
                .map(|node| execute_node(*node, lp_arena, expr_arena))
                .collect::<PolarsResult<_>>()?;

            // Validate that all inputs have the same height.
            // GPU HConcat cannot pad shorter tables with nulls without knowing column types.
            let heights: Vec<usize> = tables.iter().map(|t| t.height()).collect();
            if heights.windows(2).any(|w| w[0] != w[1]) {
                polars_bail!(ComputeError: "GPU HConcat requires all inputs to have the same height, got {:?}", heights);
            }

            // Collect all columns from all tables
            let mut all_columns = Vec::new();
            let mut all_names = Vec::new();
            for t in &tables {
                for i in 0..t.width() {
                    all_columns.push(t.column(i)?);
                    all_names.push(t.names()[i].clone());
                }
            }

            let combined = GpuDataFrame::from_columns(all_columns, all_names)?;
            let schema_names: Vec<&str> = schema.iter_names().map(|n| n.as_str()).collect();
            combined.select_columns(&schema_names)
        }

        other => {
            let kind: &'static str = other.into();
            polars_bail!(ComputeError: "GPU engine: unsupported IR node {}", kind)
        }
    }
}

/// Extract the input node and aggregation kind from an AExpr that wraps an aggregation.
///
/// Walks through Alias nodes to find the underlying AExpr::Agg.
fn extract_agg_info(
    node: Node,
    expr_arena: &Arena<AExpr>,
) -> PolarsResult<(Node, AggregationKind)> {
    match expr_arena.get(node) {
        AExpr::Agg(agg) => {
            let (input, kind) = map_ir_agg(agg)?;
            Ok((input, kind))
        }
        _ => polars_bail!(ComputeError: "GPU engine: expected aggregation expression in GroupBy"),
    }
}

/// Map an IRAggExpr to its input node and cudf AggregationKind.
fn map_ir_agg(agg: &IRAggExpr) -> PolarsResult<(Node, AggregationKind)> {
    match agg {
        IRAggExpr::Sum(input) => Ok((*input, AggregationKind::Sum)),
        IRAggExpr::Min { input, .. } => Ok((*input, AggregationKind::Min)),
        IRAggExpr::Max { input, .. } => Ok((*input, AggregationKind::Max)),
        IRAggExpr::Mean(input) => Ok((*input, AggregationKind::Mean)),
        IRAggExpr::Median(input) => Ok((*input, AggregationKind::Median)),
        IRAggExpr::Count {
            input,
            include_nulls,
        } => {
            if *include_nulls {
                Ok((*input, AggregationKind::Count))
            } else {
                Ok((*input, AggregationKind::CountValid))
            }
        }
        IRAggExpr::NUnique(input) => Ok((*input, AggregationKind::Nunique)),
        IRAggExpr::First(input) => Ok((*input, AggregationKind::NthElement { n: 0 })),
        IRAggExpr::Last(input) => Ok((*input, AggregationKind::NthElement { n: -1 })),
        IRAggExpr::Std(input, ddof) => Ok((*input, AggregationKind::Std { ddof: *ddof as i32 })),
        IRAggExpr::Var(input, ddof) => {
            Ok((*input, AggregationKind::Variance { ddof: *ddof as i32 }))
        }
        IRAggExpr::Quantile { .. } => {
            polars_bail!(ComputeError: "GPU engine: Quantile aggregation not yet supported")
        }
        other => {
            polars_bail!(ComputeError: "GPU engine: unsupported aggregation type: {:?}", other)
        }
    }
}

/// Build a joined table from left/right gather maps, applying suffix for name conflicts.
fn build_joined_table(
    left: &GpuDataFrame,
    right: &GpuDataFrame,
    left_indices: &cudf::Column,
    right_indices: &cudf::Column,
    schema: &std::sync::Arc<polars_core::prelude::Schema>,
    suffix: &str,
) -> PolarsResult<GpuDataFrame> {
    let left_gathered = gpu_result(left.inner_table().gather(left_indices))?;
    let right_gathered = gpu_result(right.inner_table().gather(right_indices))?;

    // Build column names: left names + right names (with suffix for conflicts)
    let left_name_set: std::collections::HashSet<&str> =
        left.names().iter().map(|s| s.as_str()).collect();
    let mut all_names = Vec::new();
    for n in left.names() {
        all_names.push(n.clone());
    }
    for n in right.names() {
        if left_name_set.contains(n.as_str()) {
            all_names.push(format!("{}{}", n, suffix));
        } else {
            all_names.push(n.clone());
        }
    }

    // Combine columns from both gathered tables
    let left_ncols = left_gathered.num_columns();
    let right_ncols = right_gathered.num_columns();
    let mut all_columns = Vec::with_capacity(left_ncols + right_ncols);
    for i in 0..left_ncols {
        all_columns.push(gpu_result(left_gathered.column(i))?);
    }
    for i in 0..right_ncols {
        all_columns.push(gpu_result(right_gathered.column(i))?);
    }

    let combined = GpuDataFrame::from_columns(all_columns, all_names)?;

    // Reorder to match the output schema
    let schema_names: Vec<&str> = schema.iter_names().map(|n| n.as_str()).collect();
    combined.select_columns(&schema_names)
}

/// Execute an optimized IR plan on GPU, returning a Polars DataFrame.
pub fn execute_plan(plan: IRPlan) -> PolarsResult<DataFrame> {
    let result = execute_node(plan.lp_top, &plan.lp_arena, &plan.expr_arena)?;
    result.to_polars()
}
