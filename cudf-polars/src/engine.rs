//! GPU execution engine: walks the IR tree and executes nodes on GPU.

use polars_core::prelude::*;
use polars_error::{PolarsResult, polars_bail};
use polars_plan::plans::{AExpr, IR, IRPlan};
use polars_utils::arena::{Arena, Node};

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

        other => {
            let kind: &'static str = other.into();
            polars_bail!(ComputeError: "GPU engine: unsupported IR node {}", kind)
        }
    }
}

/// Execute an optimized IR plan on GPU, returning a Polars DataFrame.
pub fn execute_plan(plan: IRPlan) -> PolarsResult<DataFrame> {
    let result = execute_node(plan.lp_top, &plan.lp_arena, &plan.expr_arena)?;
    result.to_polars()
}
