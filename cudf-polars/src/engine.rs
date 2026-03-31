//! GPU execution engine: walks the IR tree and executes nodes on GPU.

use polars_core::prelude::*;
use polars_error::{PolarsResult, polars_bail};
use polars_plan::plans::{AExpr, IRAggExpr, IR, IRPlan};
use polars_utils::arena::{Arena, Node};

use cudf::aggregation::AggregationKind;
use cudf::sorting::{SortOrder, NullOrder};
use cudf::stream_compaction::DuplicateKeepOption;

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
            df,
            output_schema,
            ..
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
            input,
            expr: exprs,
            ..
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

        IR::Slice {
            input,
            offset,
            len,
        } => {
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
            let existing_width = table.width();
            let mut all_columns = Vec::with_capacity(existing_width + exprs.len());
            let mut all_names = Vec::with_capacity(existing_width + exprs.len());

            for i in 0..existing_width {
                all_columns.push(table.column(i)?);
                all_names.push(table.names()[i].clone());
            }

            for e in &exprs {
                let col = expr::eval_expr(e.node(), expr_arena, &table)?;
                let name = e.output_name().to_string();

                if let Some(pos) = all_names.iter().position(|n| n == &name) {
                    all_columns[pos] = col;
                } else {
                    all_columns.push(col);
                    all_names.push(name);
                }
            }

            // Reorder to match the output schema
            let schema_names: Vec<&str> = schema.iter_names().map(|n| n.as_str()).collect();
            let mut ordered_columns = Vec::with_capacity(schema_names.len());
            let mut ordered_names = Vec::with_capacity(schema_names.len());
            for &sn in &schema_names {
                if let Some(pos) = all_names.iter().position(|n| n == sn) {
                    ordered_columns.push(std::mem::replace(
                        &mut all_columns[pos],
                        crate::error::gpu_result(cudf::Column::from_slice(&[0i32])).unwrap(),
                    ));
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
                let (input_node, agg_kind) =
                    extract_agg_info(agg_expr_ir.node(), expr_arena)?;

                // Evaluate the input column for this aggregation
                let input_col = expr::eval_expr(input_node, expr_arena, &table)?;

                // Always add as a new column entry
                let val_idx = value_columns.len();
                value_columns.push(input_col);

                agg_requests.push((val_idx, agg_kind));
                agg_names.push(agg_name);
            }

            let result =
                table.groupby(key_columns, key_names, value_columns, agg_requests, agg_names, *maintain_order)?;

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

            let result =
                table.distinct(subset.as_deref(), keep, options.maintain_order)?;

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
            file_options,
            output_schema,
            predicate,
            ..
        } => {
            use polars_plan::plans::FileScan;

            let file_options = file_options.clone();
            let output_schema = output_schema.clone();
            let predicate = predicate.clone();

            match scan_type {
                FileScan::Parquet { .. } => {
                    let paths = sources.as_paths().ok_or_else(|| {
                        polars_err!(ComputeError: "GPU engine: Scan requires file paths")
                    })?;

                    if paths.is_empty() {
                        polars_bail!(ComputeError: "GPU engine: Scan has no source paths");
                    }

                    // Read the first (or only) parquet file
                    let path_str = paths[0].to_string_lossy().to_string();

                    // Determine which columns to read
                    let col_names: Vec<String> = if let Some(ref with_cols) = file_options.with_columns
                    {
                        with_cols.iter().map(|c| c.to_string()).collect()
                    } else {
                        vec![]
                    };

                    let mut reader = cudf::io::parquet::ParquetReader::new(&path_str);
                    if !col_names.is_empty() {
                        reader = reader.columns(col_names.clone());
                    }

                    // Apply row limit/skip from slice
                    if let Some((offset, len)) = file_options.slice {
                        if offset > 0 {
                            reader = reader.skip_rows(offset as usize);
                        }
                        reader = reader.num_rows(len);
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
        AExpr::Alias(inner, _) => extract_agg_info(*inner, expr_arena),
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
        IRAggExpr::Count(input, include_nulls) => {
            // TODO: cudf-rs AggregationKind has no CountValid variant.
            // Count always includes nulls at the libcudf level.
            // The exclude-nulls logic is handled in eval_agg_expr instead.
            let _ = include_nulls;
            Ok((*input, AggregationKind::Count))
        }
        IRAggExpr::NUnique(input) => Ok((*input, AggregationKind::Nunique)),
        IRAggExpr::First(input) => Ok((*input, AggregationKind::NthElement { n: 0 })),
        IRAggExpr::Last(input) => Ok((*input, AggregationKind::NthElement { n: -1 })),
        IRAggExpr::Std(input, ddof) => Ok((*input, AggregationKind::Std { ddof: *ddof as i32 })),
        IRAggExpr::Var(input, ddof) => Ok((*input, AggregationKind::Variance { ddof: *ddof as i32 })),
        IRAggExpr::Quantile { .. } => {
            polars_bail!(ComputeError: "GPU engine: Quantile aggregation not yet supported")
        }
        _ => polars_bail!(ComputeError: "GPU engine: unsupported aggregation type"),
    }
}

/// Execute an optimized IR plan on GPU, returning a Polars DataFrame.
pub fn execute_plan(plan: IRPlan) -> PolarsResult<DataFrame> {
    let result = execute_node(plan.lp_top, &plan.lp_arena, &plan.expr_arena)?;
    result.to_polars()
}
