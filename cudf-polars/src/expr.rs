//! Expression evaluation on GPU.
//!
//! Walks the `AExpr` arena and evaluates each node, producing a GPU column.

use cudf::Column as GpuColumn;
use cudf::types::{DataType as GpuDataType, TypeId as GpuTypeId};
use polars_error::{PolarsResult, polars_bail, polars_err};
use polars_plan::plans::AExpr;
use polars_plan::plans::LiteralValue;
use polars_utils::arena::{Arena, Node};

use crate::error::gpu_result;
use crate::gpu_frame::GpuDataFrame;
use crate::types::{is_comparison, map_dtype, map_operator};

/// Compute the output type for arithmetic operations (supertype promotion).
fn arithmetic_output_type(left_type: &GpuDataType, right_type: &GpuDataType) -> GpuDataType {
    let l = left_type.id();
    let r = right_type.id();

    if l == GpuTypeId::Float64 || r == GpuTypeId::Float64 {
        GpuDataType::new(GpuTypeId::Float64)
    } else if l == GpuTypeId::Float32 || r == GpuTypeId::Float32 {
        GpuDataType::new(GpuTypeId::Float32)
    } else if l == GpuTypeId::Int64 || r == GpuTypeId::Int64 {
        GpuDataType::new(GpuTypeId::Int64)
    } else {
        *left_type
    }
}

/// Evaluate an expression node, producing a GPU column.
pub fn eval_expr(
    node: Node,
    expr_arena: &Arena<AExpr>,
    df: &GpuDataFrame,
) -> PolarsResult<GpuColumn> {
    match expr_arena.get(node) {
        AExpr::Column(name) => df.column_by_name(name.as_str()),

        AExpr::Literal(lit) => literal_to_gpu_column(lit, df.height()),

        AExpr::BinaryExpr { left, op, right } => {
            let left_node = *left;
            let right_node = *right;
            let op = *op;
            let left_col = eval_expr(left_node, expr_arena, df)?;
            let right_col = eval_expr(right_node, expr_arena, df)?;

            let gpu_op = map_operator(op)?;
            let l_type = left_col.data_type();
            let r_type = right_col.data_type();
            let output_type = if is_comparison(op) {
                GpuDataType::new(GpuTypeId::Bool8)
            } else {
                arithmetic_output_type(&l_type, &r_type)
            };
            gpu_result(left_col.binary_op(&right_col, gpu_op, output_type))
        }

        AExpr::Cast { expr, dtype, .. } => {
            let expr_node = *expr;
            let dtype = dtype.clone();
            let col = eval_expr(expr_node, expr_arena, df)?;
            let gpu_dtype = map_dtype(&dtype)?;
            gpu_result(col.cast(gpu_dtype))
        }

        AExpr::Len => {
            let height = df.height();
            let values: Vec<u32> = vec![height as u32; height];
            gpu_result(GpuColumn::from_slice(&values))
        }

        AExpr::Alias(inner, _name) => {
            eval_expr(*inner, expr_arena, df)
        }

        _ => polars_bail!(ComputeError: "GPU engine: unsupported expression node"),
    }
}

/// Convert a Polars `LiteralValue` to a GPU column of the given height.
fn literal_to_gpu_column(lit: &LiteralValue, height: usize) -> PolarsResult<GpuColumn> {
    match lit {
        // TODO: Null type should match the context (e.g., the other operand's type in BinaryExpr).
        // Currently always creates i32 nullable column. This may cause type mismatches.
        LiteralValue::Null => {
            let opts: Vec<Option<i32>> = vec![None; height];
            gpu_result(GpuColumn::from_optional_i32(&opts))
        }
        LiteralValue::Boolean(v) => {
            let data = vec![*v; height];
            gpu_result(GpuColumn::from_slice(&data))
        }
        LiteralValue::Int32(v) => {
            let data = vec![*v; height];
            gpu_result(GpuColumn::from_slice(&data))
        }
        LiteralValue::Int64(v) => {
            let data = vec![*v; height];
            gpu_result(GpuColumn::from_slice(&data))
        }
        LiteralValue::UInt32(v) => {
            let data = vec![*v; height];
            gpu_result(GpuColumn::from_slice(&data))
        }
        LiteralValue::UInt64(v) => {
            let data = vec![*v; height];
            gpu_result(GpuColumn::from_slice(&data))
        }
        LiteralValue::Float32(v) => {
            let data = vec![*v; height];
            gpu_result(GpuColumn::from_slice(&data))
        }
        LiteralValue::Float64(v) => {
            let data = vec![*v; height];
            gpu_result(GpuColumn::from_slice(&data))
        }
        LiteralValue::String(s) => {
            let strings: Vec<&str> = vec![s.as_str(); height];
            gpu_result(GpuColumn::from_strings(&strings))
        }
        LiteralValue::Int(v) => {
            let val = i64::try_from(*v)
                .map_err(|_| polars_err!(ComputeError: "integer literal {} exceeds i64 range", v))?;
            let data = vec![val; height];
            gpu_result(GpuColumn::from_slice(&data))
        }
        LiteralValue::Float(v) => {
            let data = vec![*v; height];
            gpu_result(GpuColumn::from_slice(&data))
        }
        _ => polars_bail!(ComputeError: "GPU engine: unsupported literal type"),
    }
}
