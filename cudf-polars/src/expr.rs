//! Expression evaluation on GPU.
//!
//! Walks the `AExpr` arena and evaluates each node, producing a GPU column.

use cudf::Column as GpuColumn;
use cudf::reduction::ReduceOp;
use cudf::types::{DataType as GpuDataType, TypeId as GpuTypeId};
use polars_error::{PolarsResult, polars_bail, polars_err};
use polars_plan::plans::{AExpr, IRAggExpr};
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

        AExpr::Agg(agg_expr) => eval_agg_expr(agg_expr, expr_arena, df),

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

/// Evaluate a standalone aggregation expression (reduce column to scalar, broadcast to height).
fn eval_agg_expr(
    agg: &IRAggExpr,
    expr_arena: &Arena<AExpr>,
    df: &GpuDataFrame,
) -> PolarsResult<GpuColumn> {
    let height = df.height();
    match agg {
        IRAggExpr::Sum(input) => reduce_and_broadcast(*input, ReduceOp::Sum, expr_arena, df, height),
        IRAggExpr::Min { input, .. } => {
            reduce_and_broadcast(*input, ReduceOp::Min, expr_arena, df, height)
        }
        IRAggExpr::Max { input, .. } => {
            reduce_and_broadcast(*input, ReduceOp::Max, expr_arena, df, height)
        }
        IRAggExpr::Mean(input) => {
            reduce_and_broadcast(*input, ReduceOp::Mean, expr_arena, df, height)
        }
        IRAggExpr::Median(input) => {
            reduce_and_broadcast(*input, ReduceOp::Median, expr_arena, df, height)
        }
        IRAggExpr::Std(input, _ddof) => {
            // TODO: ReduceOp::Std does not accept a ddof parameter.
            // cudf reduction always uses ddof=1. If _ddof != 1, results will be incorrect.
            reduce_and_broadcast(*input, ReduceOp::Std, expr_arena, df, height)
        }
        IRAggExpr::Var(input, _ddof) => {
            // TODO: ReduceOp::Variance does not accept a ddof parameter.
            // cudf reduction always uses ddof=1. If _ddof != 1, results will be incorrect.
            reduce_and_broadcast(*input, ReduceOp::Variance, expr_arena, df, height)
        }
        IRAggExpr::Count(input, include_nulls) => {
            if *include_nulls {
                // Count all rows including nulls
                let count = height as u32;
                let data = vec![count; height];
                gpu_result(GpuColumn::from_slice(&data))
            } else {
                // Count only non-null rows: height - null_count
                let col = eval_expr(*input, expr_arena, df)?;
                let valid_count = (height - col.null_count()) as u32;
                let data = vec![valid_count; height];
                gpu_result(GpuColumn::from_slice(&data))
            }
        }
        IRAggExpr::NUnique(input) => {
            let col = eval_expr(*input, expr_arena, df)?;
            let n = gpu_result(col.distinct_count())? as u32;
            let data = vec![n; height];
            gpu_result(GpuColumn::from_slice(&data))
        }
        IRAggExpr::First(input) | IRAggExpr::Last(input) => {
            // In standalone context, just evaluate the input
            eval_expr(*input, expr_arena, df)
        }
        _ => polars_bail!(ComputeError: "GPU engine: unsupported standalone aggregation"),
    }
}

/// Reduce a column and broadcast the scalar result to `height` rows.
fn reduce_and_broadcast(
    input_node: Node,
    op: ReduceOp,
    expr_arena: &Arena<AExpr>,
    df: &GpuDataFrame,
    height: usize,
) -> PolarsResult<GpuColumn> {
    let col = eval_expr(input_node, expr_arena, df)?;
    let dtype = col.data_type();

    // For mean/median/std/var, output is always float64
    let output_type = match op {
        ReduceOp::Mean | ReduceOp::Median | ReduceOp::Std | ReduceOp::Variance => {
            GpuDataType::new(GpuTypeId::Float64)
        }
        _ => dtype,
    };

    let scalar = gpu_result(col.reduce(op, output_type))?;

    // Extract scalar value and broadcast to all rows
    broadcast_scalar(&scalar, height)
}

/// Broadcast a scalar value to a column of the given height.
fn broadcast_scalar(scalar: &cudf::Scalar, height: usize) -> PolarsResult<GpuColumn> {
    let dtype = scalar.data_type();
    let tid = dtype.id();

    // If the scalar is not valid (null), create a null column of the correct type
    if !scalar.is_valid() {
        return match tid {
            GpuTypeId::Float64 => {
                let opts: Vec<Option<f64>> = vec![None; height];
                gpu_result(GpuColumn::from_optional_f64(&opts))
            }
            GpuTypeId::Float32 => {
                let opts: Vec<Option<f32>> = vec![None; height];
                gpu_result(GpuColumn::from_optional_f32(&opts))
            }
            GpuTypeId::Int64 => {
                let opts: Vec<Option<i64>> = vec![None; height];
                gpu_result(GpuColumn::from_optional_i64(&opts))
            }
            GpuTypeId::Uint32 => {
                let opts: Vec<Option<u32>> = vec![None; height];
                gpu_result(GpuColumn::from_optional_u32(&opts))
            }
            GpuTypeId::Uint64 => {
                let opts: Vec<Option<u64>> = vec![None; height];
                gpu_result(GpuColumn::from_optional_u64(&opts))
            }
            _ => {
                let opts: Vec<Option<i32>> = vec![None; height];
                gpu_result(GpuColumn::from_optional_i32(&opts))
            }
        };
    }

    match tid {
        GpuTypeId::Int8 => {
            let v: i8 = gpu_result(scalar.value())?;
            gpu_result(GpuColumn::from_slice(&vec![v; height]))
        }
        GpuTypeId::Int16 => {
            let v: i16 = gpu_result(scalar.value())?;
            gpu_result(GpuColumn::from_slice(&vec![v; height]))
        }
        GpuTypeId::Int32 => {
            let v: i32 = gpu_result(scalar.value())?;
            gpu_result(GpuColumn::from_slice(&vec![v; height]))
        }
        GpuTypeId::Int64 => {
            let v: i64 = gpu_result(scalar.value())?;
            gpu_result(GpuColumn::from_slice(&vec![v; height]))
        }
        GpuTypeId::Uint8 => {
            let v: u8 = gpu_result(scalar.value())?;
            gpu_result(GpuColumn::from_slice(&vec![v; height]))
        }
        GpuTypeId::Uint16 => {
            let v: u16 = gpu_result(scalar.value())?;
            gpu_result(GpuColumn::from_slice(&vec![v; height]))
        }
        GpuTypeId::Uint32 => {
            let v: u32 = gpu_result(scalar.value())?;
            gpu_result(GpuColumn::from_slice(&vec![v; height]))
        }
        GpuTypeId::Uint64 => {
            let v: u64 = gpu_result(scalar.value())?;
            gpu_result(GpuColumn::from_slice(&vec![v; height]))
        }
        GpuTypeId::Float32 => {
            let v: f32 = gpu_result(scalar.value())?;
            gpu_result(GpuColumn::from_slice(&vec![v; height]))
        }
        GpuTypeId::Float64 => {
            let v: f64 = gpu_result(scalar.value())?;
            gpu_result(GpuColumn::from_slice(&vec![v; height]))
        }
        _ => polars_bail!(ComputeError: "GPU engine: cannot broadcast scalar of type {:?}", tid),
    }
}
