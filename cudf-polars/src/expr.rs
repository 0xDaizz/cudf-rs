//! Expression evaluation on GPU.
//!
//! Walks the `AExpr` arena and evaluates each node, producing a GPU column.

use cudf::Column as GpuColumn;
use cudf::reduction::ReduceOp;
use cudf::types::{DataType as GpuDataType, TypeId as GpuTypeId};
use cudf::unary::UnaryOp;
use polars_error::{PolarsResult, polars_bail};
use polars_plan::plans::IRBooleanFunction;
use polars_plan::plans::IRFunctionExpr;
use polars_plan::plans::LiteralValue;
use polars_plan::plans::{AExpr, IRAggExpr};
use polars_utils::arena::{Arena, Node};

use crate::error::gpu_result;
use crate::gpu_frame::GpuDataFrame;
use crate::types::{is_comparison, map_dtype, map_operator};

/// Returns true if the type ID is a temporal type (timestamp or duration).
fn is_temporal(tid: GpuTypeId) -> bool {
    matches!(
        tid,
        GpuTypeId::TimestampDays
            | GpuTypeId::TimestampSeconds
            | GpuTypeId::TimestampMilliseconds
            | GpuTypeId::TimestampMicroseconds
            | GpuTypeId::TimestampNanoseconds
            | GpuTypeId::DurationDays
            | GpuTypeId::DurationSeconds
            | GpuTypeId::DurationMilliseconds
            | GpuTypeId::DurationMicroseconds
            | GpuTypeId::DurationNanoseconds
    )
}

/// Compute the output type for arithmetic operations (supertype promotion).
fn arithmetic_output_type(left_type: &GpuDataType, right_type: &GpuDataType) -> GpuDataType {
    use GpuTypeId::*;
    let l = left_type.id();
    let r = right_type.id();

    // Temporal types: let cudf handle the type promotion natively.
    // Return the left type as a hint; cudf's binary_op will determine the actual output.
    if is_temporal(l) || is_temporal(r) {
        return left_type.clone();
    }

    // Bool + Bool stays Bool (bitwise AND/OR/XOR on booleans)
    if l == Bool8 && r == Bool8 {
        return GpuDataType::new(Bool8);
    }

    // Float types dominate
    if l == Float64 || r == Float64 {
        return GpuDataType::new(Float64);
    }
    if l == Float32 || r == Float32 {
        return GpuDataType::new(Float32);
    }

    // Unsigned 64/32 promote to Int64 to avoid overflow
    if l == Uint64 || r == Uint64 || l == Uint32 || r == Uint32 {
        return GpuDataType::new(Int64);
    }

    // Int64 dominates remaining integers
    if l == Int64 || r == Int64 {
        return GpuDataType::new(Int64);
    }

    // Everything else (Int8, Int16, UInt8, UInt16, Int32, Bool) → Int32
    GpuDataType::new(Int32)
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
            let gpu_op = map_operator(op)?;

            // Scalar optimization: avoid broadcasting a literal to a full column.
            // Use binary_op_scalar (or binaryop::binary_op for scalar-column order)
            // to let libcudf handle the broadcast internally on GPU.
            // Falls back to column-column path for unsupported scalar types (Null, String, etc.).
            let left_is_literal = matches!(expr_arena.get(left_node), AExpr::Literal(_));
            let right_is_literal = matches!(expr_arena.get(right_node), AExpr::Literal(_));

            let try_scalar = || -> PolarsResult<GpuColumn> {
                if left_is_literal && !right_is_literal {
                    let scalar = literal_to_scalar(expr_arena.get(left_node))?;
                    let right_col = eval_expr(right_node, expr_arena, df)?;
                    let output_type = if is_comparison(op) {
                        GpuDataType::new(GpuTypeId::Bool8)
                    } else {
                        arithmetic_output_type(&scalar.data_type(), &right_col.data_type())
                    };
                    return gpu_result(cudf::binaryop::binary_op(
                        &scalar,
                        &right_col,
                        gpu_op,
                        output_type,
                    ));
                }
                if right_is_literal && !left_is_literal {
                    let left_col = eval_expr(left_node, expr_arena, df)?;
                    let scalar = literal_to_scalar(expr_arena.get(right_node))?;
                    let output_type = if is_comparison(op) {
                        GpuDataType::new(GpuTypeId::Bool8)
                    } else {
                        arithmetic_output_type(&left_col.data_type(), &scalar.data_type())
                    };
                    return gpu_result(left_col.binary_op_scalar(&scalar, gpu_op, output_type));
                }
                polars_bail!(ComputeError: "not a scalar case")
            };

            if let Ok(result) = try_scalar() {
                return Ok(result);
            }

            // Both are columns (or unsupported scalar type) — fall back to column-column op
            let left_col = eval_expr(left_node, expr_arena, df)?;
            let right_col = eval_expr(right_node, expr_arena, df)?;
            let output_type = if is_comparison(op) {
                GpuDataType::new(GpuTypeId::Bool8)
            } else {
                arithmetic_output_type(&left_col.data_type(), &right_col.data_type())
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

        AExpr::Agg(agg_expr) => eval_agg_expr(agg_expr, expr_arena, df),

        AExpr::Ternary {
            predicate,
            truthy,
            falsy,
        } => {
            let pred_node = *predicate;
            let true_node = *truthy;
            let false_node = *falsy;
            let cond = eval_expr(pred_node, expr_arena, df)?;
            let t = eval_expr(true_node, expr_arena, df)?;
            let f = eval_expr(false_node, expr_arena, df)?;
            // copy_if_else: where mask is true use self (truthy), else use other (falsy)
            gpu_result(t.copy_if_else(&f, &cond))
        }

        AExpr::Function {
            input, function, ..
        } => {
            let input = input.clone();
            let function = function.clone();
            eval_ir_function(&input, &function, expr_arena, df)
        }

        AExpr::Over {
            function,
            partition_by,
            order_by,
            mapping,
        } => {
            use polars_plan::dsl::options::WindowMapping;

            if order_by.is_some() {
                polars_bail!(ComputeError: "GPU engine: window functions with order_by not yet supported");
            }

            let function = *function;
            let partition_by = partition_by.clone();
            let mapping = *mapping;

            match mapping {
                WindowMapping::GroupsToRows => {
                    eval_window_groups_to_rows(function, &partition_by, expr_arena, df)
                }
                other => {
                    polars_bail!(ComputeError: "GPU engine: unsupported window mapping {:?}", other)
                }
            }
        }

        other => polars_bail!(ComputeError: "GPU engine: unsupported expression: {:?}", other),
    }
}

/// Evaluate a window function with `GroupsToRows` mapping.
///
/// Strategy: groupby partition keys → aggregate → left join back to original rows.
/// This broadcasts the per-group aggregate result to every row in that group.
fn eval_window_groups_to_rows(
    function: Node,
    partition_by: &[Node],
    expr_arena: &Arena<AExpr>,
    df: &GpuDataFrame,
) -> PolarsResult<GpuColumn> {
    use cudf::Table as GpuTable;

    // 1. Evaluate partition key columns
    let mut key_columns = Vec::with_capacity(partition_by.len());
    for &key_node in partition_by {
        let col = eval_expr(key_node, expr_arena, df)?;
        key_columns.push(col);
    }

    // 2. Extract the aggregation info from the function node
    let (input_node, agg_kind) = crate::engine::extract_agg_info(function, expr_arena)?;

    // 3. Evaluate the input column for the aggregation
    let value_col = eval_expr(input_node, expr_arena, df)?;

    // 4. Perform groupby aggregation: keys → aggregate
    let keys_table = gpu_result(GpuTable::new(key_columns.clone()))?;
    let values_table = gpu_result(GpuTable::new(vec![value_col]))?;

    let gb = cudf::groupby::GroupBy::new(&keys_table).agg(0, agg_kind);
    // Result: [key_col_0, key_col_1, ..., agg_result_col]
    let agg_result = gpu_result(gb.execute(&values_table))?;

    // 5. Extract the aggregated keys and the agg result column from the result table
    let n_keys = partition_by.len();
    let mut agg_key_cols = Vec::with_capacity(n_keys);
    for i in 0..n_keys {
        agg_key_cols.push(gpu_result(agg_result.column(i))?);
    }
    let agg_value_col = gpu_result(agg_result.column(n_keys))?;

    // 6. Left join: original keys LEFT JOIN aggregated keys → get per-row agg values
    let agg_keys_table = gpu_result(GpuTable::new(agg_key_cols))?;
    let join_result = gpu_result(keys_table.left_join(&agg_keys_table))?;

    // 7. Gather the agg result column using right_indices to broadcast to original rows
    let agg_as_table = gpu_result(GpuTable::new(vec![agg_value_col]))?;
    let gathered = gpu_result(agg_as_table.gather(&join_result.right_indices))?;
    let agg_col = gpu_result(gathered.column(0))?;

    // 8. Restore original row order using scatter (O(n)) instead of sort (O(n log n)).
    //    left_indices[i] = the original row position for join result row i.
    //    We scatter the gathered agg values into their correct positions.
    let original_height = df.height();
    let agg_dtype = agg_col.data_type();
    let scatter_source = gpu_result(GpuTable::new(vec![agg_col]))?;
    let target_col = null_column_for_type(agg_dtype, original_height)?;
    let target_table = gpu_result(GpuTable::new(vec![target_col]))?;
    let scattered = gpu_result(scatter_source.scatter(&join_result.left_indices, &target_table))?;
    let result_col = gpu_result(scattered.column(0))?;

    Ok(result_col)
}

/// Convert a Polars `LiteralValue` to a GPU column of the given height.
fn literal_to_gpu_column(lit: &LiteralValue, height: usize) -> PolarsResult<GpuColumn> {
    use polars_core::prelude::*;
    use polars_plan::plans::DynLiteralValue;

    match lit {
        LiteralValue::Scalar(scalar) => {
            if scalar.is_null() {
                // Null scalar: create a null column of the appropriate type
                let gpu_dtype = crate::types::map_dtype(scalar.dtype())?;
                let tid = gpu_dtype.id();
                return match tid {
                    GpuTypeId::Bool8 => {
                        let opts: Vec<Option<bool>> = vec![None; height];
                        gpu_result(GpuColumn::from_optional_bool(&opts))
                    }
                    GpuTypeId::Int8 => {
                        let opts: Vec<Option<i8>> = vec![None; height];
                        gpu_result(GpuColumn::from_optional_i8(&opts))
                    }
                    GpuTypeId::Int16 => {
                        let opts: Vec<Option<i16>> = vec![None; height];
                        gpu_result(GpuColumn::from_optional_i16(&opts))
                    }
                    GpuTypeId::Int32 => {
                        let opts: Vec<Option<i32>> = vec![None; height];
                        gpu_result(GpuColumn::from_optional_i32(&opts))
                    }
                    GpuTypeId::Int64 => {
                        let opts: Vec<Option<i64>> = vec![None; height];
                        gpu_result(GpuColumn::from_optional_i64(&opts))
                    }
                    GpuTypeId::Uint8 => {
                        let opts: Vec<Option<u8>> = vec![None; height];
                        gpu_result(GpuColumn::from_optional_u8(&opts))
                    }
                    GpuTypeId::Uint16 => {
                        let opts: Vec<Option<u16>> = vec![None; height];
                        gpu_result(GpuColumn::from_optional_u16(&opts))
                    }
                    GpuTypeId::Uint32 => {
                        let opts: Vec<Option<u32>> = vec![None; height];
                        gpu_result(GpuColumn::from_optional_u32(&opts))
                    }
                    GpuTypeId::Uint64 => {
                        let opts: Vec<Option<u64>> = vec![None; height];
                        gpu_result(GpuColumn::from_optional_u64(&opts))
                    }
                    GpuTypeId::Float32 => {
                        let opts: Vec<Option<f32>> = vec![None; height];
                        gpu_result(GpuColumn::from_optional_f32(&opts))
                    }
                    GpuTypeId::Float64 => {
                        let opts: Vec<Option<f64>> = vec![None; height];
                        gpu_result(GpuColumn::from_optional_f64(&opts))
                    }
                    GpuTypeId::String => {
                        let opts: Vec<Option<&str>> = vec![None; height];
                        gpu_result(GpuColumn::from_optional_strings(&opts))
                    }
                    other => {
                        polars_bail!(ComputeError: "GPU engine: cannot create null column for type {:?}", other)
                    }
                };
            }
            let value = scalar.value();
            match value {
                AnyValue::Boolean(v) => {
                    let data = vec![*v; height];
                    gpu_result(GpuColumn::from_slice(&data))
                }
                AnyValue::Int8(v) => {
                    let data = vec![*v; height];
                    gpu_result(GpuColumn::from_slice(&data))
                }
                AnyValue::Int16(v) => {
                    let data = vec![*v; height];
                    gpu_result(GpuColumn::from_slice(&data))
                }
                AnyValue::Int32(v) => {
                    let data = vec![*v; height];
                    gpu_result(GpuColumn::from_slice(&data))
                }
                AnyValue::Int64(v) => {
                    let data = vec![*v; height];
                    gpu_result(GpuColumn::from_slice(&data))
                }
                AnyValue::UInt8(v) => {
                    let data = vec![*v; height];
                    gpu_result(GpuColumn::from_slice(&data))
                }
                AnyValue::UInt16(v) => {
                    let data = vec![*v; height];
                    gpu_result(GpuColumn::from_slice(&data))
                }
                AnyValue::UInt32(v) => {
                    let data = vec![*v; height];
                    gpu_result(GpuColumn::from_slice(&data))
                }
                AnyValue::UInt64(v) => {
                    let data = vec![*v; height];
                    gpu_result(GpuColumn::from_slice(&data))
                }
                AnyValue::Float32(v) => {
                    let data = vec![*v; height];
                    gpu_result(GpuColumn::from_slice(&data))
                }
                AnyValue::Float64(v) => {
                    let data = vec![*v; height];
                    gpu_result(GpuColumn::from_slice(&data))
                }
                AnyValue::String(s) => {
                    let strings: Vec<&str> = vec![s; height];
                    gpu_result(GpuColumn::from_strings(&strings))
                }
                AnyValue::StringOwned(s) => {
                    let s_ref: &str = s.as_str();
                    let strings: Vec<&str> = vec![s_ref; height];
                    gpu_result(GpuColumn::from_strings(&strings))
                }
                other => {
                    polars_bail!(ComputeError: "GPU engine: unsupported scalar AnyValue type: {:?}", other)
                }
            }
        }
        LiteralValue::Dyn(dyn_lit) => match dyn_lit {
            DynLiteralValue::Int(v) => {
                let val = i64::try_from(*v).map_err(
                    |_| polars_err!(ComputeError: "integer literal {} exceeds i64 range", v),
                )?;
                let data = vec![val; height];
                gpu_result(GpuColumn::from_slice(&data))
            }
            DynLiteralValue::Float(v) => {
                let data = vec![*v; height];
                gpu_result(GpuColumn::from_slice(&data))
            }
            DynLiteralValue::Str(s) => {
                let strings: Vec<&str> = vec![s.as_str(); height];
                gpu_result(GpuColumn::from_strings(&strings))
            }
            other => {
                polars_bail!(ComputeError: "GPU engine: unsupported dynamic literal type: {:?}", other)
            }
        },
        other => polars_bail!(ComputeError: "GPU engine: unsupported literal type: {:?}", other),
    }
}

/// Convert a Polars `LiteralValue` to a cudf `Scalar` (no GPU column allocation).
///
/// Used by the BinaryExpr scalar optimization path to avoid broadcasting
/// a literal value into a full GPU column.
fn literal_to_scalar(expr: &AExpr) -> PolarsResult<cudf::Scalar> {
    use polars_core::prelude::*;
    use polars_plan::plans::DynLiteralValue;

    match expr {
        AExpr::Literal(lit) => match lit {
            LiteralValue::Scalar(scalar) => {
                if scalar.is_null() {
                    polars_bail!(ComputeError: "GPU engine: null literal not supported in scalar path")
                }
                let value = scalar.value();
                match value {
                    AnyValue::Boolean(v) => gpu_result(cudf::Scalar::new(*v)),
                    AnyValue::Int8(v) => gpu_result(cudf::Scalar::new(*v)),
                    AnyValue::Int16(v) => gpu_result(cudf::Scalar::new(*v)),
                    AnyValue::Int32(v) => gpu_result(cudf::Scalar::new(*v)),
                    AnyValue::Int64(v) => gpu_result(cudf::Scalar::new(*v)),
                    AnyValue::UInt8(v) => gpu_result(cudf::Scalar::new(*v)),
                    AnyValue::UInt16(v) => gpu_result(cudf::Scalar::new(*v)),
                    AnyValue::UInt32(v) => gpu_result(cudf::Scalar::new(*v)),
                    AnyValue::UInt64(v) => gpu_result(cudf::Scalar::new(*v)),
                    AnyValue::Float32(v) => gpu_result(cudf::Scalar::new(*v)),
                    AnyValue::Float64(v) => gpu_result(cudf::Scalar::new(*v)),
                    AnyValue::String(_) | AnyValue::StringOwned(_) => {
                        polars_bail!(ComputeError: "GPU engine: string literal not supported in scalar path")
                    }
                    _ => {
                        polars_bail!(ComputeError: "GPU engine: scalar AnyValue type {:?} not supported for scalar optimization", value)
                    }
                }
            }
            LiteralValue::Dyn(dyn_lit) => match dyn_lit {
                DynLiteralValue::Int(v) => {
                    let val = i64::try_from(*v).map_err(
                        |_| polars_err!(ComputeError: "integer literal {} exceeds i64 range", v),
                    )?;
                    gpu_result(cudf::Scalar::new(val))
                }
                DynLiteralValue::Float(v) => gpu_result(cudf::Scalar::new(*v)),
                _ => {
                    polars_bail!(ComputeError: "GPU engine: dynamic literal type {:?} not supported for scalar optimization", dyn_lit)
                }
            },
            _ => {
                polars_bail!(ComputeError: "GPU engine: literal type {:?} not supported for scalar optimization", lit)
            }
        },
        _ => polars_bail!(ComputeError: "GPU engine: expected literal expression"),
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
        IRAggExpr::Sum(input) => {
            reduce_and_broadcast(*input, ReduceOp::Sum, expr_arena, df, height)
        }
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
        IRAggExpr::Std(input, ddof) => {
            let input_node = *input;
            let ddof_val = *ddof;
            let col = eval_expr(input_node, expr_arena, df)?;
            let output_type = GpuDataType::new(GpuTypeId::Float64);
            let scalar = gpu_result(col.reduce_std_with_ddof(ddof_val as i32, output_type))?;
            broadcast_scalar(&scalar, height)
        }
        IRAggExpr::Var(input, ddof) => {
            let input_node = *input;
            let ddof_val = *ddof;
            let col = eval_expr(input_node, expr_arena, df)?;
            let output_type = GpuDataType::new(GpuTypeId::Float64);
            let scalar = gpu_result(col.reduce_var_with_ddof(ddof_val as i32, output_type))?;
            broadcast_scalar(&scalar, height)
        }
        IRAggExpr::Count {
            input,
            include_nulls,
        } => {
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
        IRAggExpr::First(input) => {
            let col = eval_expr(*input, expr_arena, df)?;
            if col.len() == 0 || height == 0 {
                return null_column_for_type(col.data_type(), height);
            }
            // Slice the first element, then repeat to fill height
            let single_row_table = gpu_result(cudf::Table::new(vec![col]))?;
            let sliced = gpu_result(single_row_table.slice(0, 1))?;
            let repeated = gpu_result(sliced.repeat(height))?;
            let cols = gpu_result(repeated.into_columns())?;
            // SAFETY: `repeated` is a 1-column table (from single-column `sliced`),
            // so `into_columns()` returns exactly 1 element. `.next()` is always Some.
            Ok(cols.into_iter().next().unwrap())
        }
        IRAggExpr::Last(input) => {
            let col = eval_expr(*input, expr_arena, df)?;
            if col.len() == 0 || height == 0 {
                return null_column_for_type(col.data_type(), height);
            }
            let last_idx = col.len() - 1;
            let single_row_table = gpu_result(cudf::Table::new(vec![col]))?;
            let sliced = gpu_result(single_row_table.slice(last_idx, last_idx + 1))?;
            let repeated = gpu_result(sliced.repeat(height))?;
            let cols = gpu_result(repeated.into_columns())?;
            // SAFETY: `repeated` is a 1-column table (from single-column `sliced`),
            // so `into_columns()` returns exactly 1 element. `.next()` is always Some.
            Ok(cols.into_iter().next().unwrap())
        }
        other => {
            polars_bail!(ComputeError: "GPU engine: unsupported standalone aggregation: {:?}", other)
        }
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

/// Create a column of `height` null values with the given GPU data type.
fn null_column_for_type(dtype: cudf::types::DataType, height: usize) -> PolarsResult<GpuColumn> {
    let tid = dtype.id();
    match tid {
        GpuTypeId::Bool8 => {
            let opts: Vec<Option<bool>> = vec![None; height];
            gpu_result(GpuColumn::from_optional_bool(&opts))
        }
        GpuTypeId::Int8 => {
            let opts: Vec<Option<i8>> = vec![None; height];
            gpu_result(GpuColumn::from_optional_i8(&opts))
        }
        GpuTypeId::Int16 => {
            let opts: Vec<Option<i16>> = vec![None; height];
            gpu_result(GpuColumn::from_optional_i16(&opts))
        }
        GpuTypeId::Int32 => {
            let opts: Vec<Option<i32>> = vec![None; height];
            gpu_result(GpuColumn::from_optional_i32(&opts))
        }
        GpuTypeId::Int64 => {
            let opts: Vec<Option<i64>> = vec![None; height];
            gpu_result(GpuColumn::from_optional_i64(&opts))
        }
        GpuTypeId::Uint8 => {
            let opts: Vec<Option<u8>> = vec![None; height];
            gpu_result(GpuColumn::from_optional_u8(&opts))
        }
        GpuTypeId::Uint16 => {
            let opts: Vec<Option<u16>> = vec![None; height];
            gpu_result(GpuColumn::from_optional_u16(&opts))
        }
        GpuTypeId::Uint32 => {
            let opts: Vec<Option<u32>> = vec![None; height];
            gpu_result(GpuColumn::from_optional_u32(&opts))
        }
        GpuTypeId::Uint64 => {
            let opts: Vec<Option<u64>> = vec![None; height];
            gpu_result(GpuColumn::from_optional_u64(&opts))
        }
        GpuTypeId::Float32 => {
            let opts: Vec<Option<f32>> = vec![None; height];
            gpu_result(GpuColumn::from_optional_f32(&opts))
        }
        GpuTypeId::Float64 => {
            let opts: Vec<Option<f64>> = vec![None; height];
            gpu_result(GpuColumn::from_optional_f64(&opts))
        }
        GpuTypeId::String => {
            let opts: Vec<Option<&str>> = vec![None; height];
            gpu_result(GpuColumn::from_optional_strings(&opts))
        }
        // Temporal types: Date/TimestampDays and DurationDays are i32, all others are i64
        GpuTypeId::TimestampDays | GpuTypeId::DurationDays => {
            let opts: Vec<Option<i32>> = vec![None; height];
            gpu_result(GpuColumn::from_optional_i32(&opts))
        }
        GpuTypeId::TimestampSeconds
        | GpuTypeId::TimestampMilliseconds
        | GpuTypeId::TimestampMicroseconds
        | GpuTypeId::TimestampNanoseconds
        | GpuTypeId::DurationSeconds
        | GpuTypeId::DurationMilliseconds
        | GpuTypeId::DurationMicroseconds
        | GpuTypeId::DurationNanoseconds => {
            let opts: Vec<Option<i64>> = vec![None; height];
            gpu_result(GpuColumn::from_optional_i64(&opts))
        }
        other => {
            polars_bail!(ComputeError: "GPU engine: cannot create null column of type {:?}", other)
        }
    }
}

/// Broadcast a scalar value to a column of the given height.
fn broadcast_scalar(scalar: &cudf::Scalar, height: usize) -> PolarsResult<GpuColumn> {
    let dtype = scalar.data_type();
    let tid = dtype.id();

    // If the scalar is not valid (null), create a null column of the correct type
    if !scalar.is_valid() {
        return null_column_for_type(dtype, height);
    }

    match tid {
        GpuTypeId::Bool8 => {
            let v: bool = gpu_result(scalar.value())?;
            gpu_result(GpuColumn::from_slice(&vec![v; height]))
        }
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
            // GPU-native: sequence with step=0 creates a constant column without host allocation
            gpu_result(cudf::filling::sequence_i32(height, v, 0))
        }
        GpuTypeId::Int64 => {
            let v: i64 = gpu_result(scalar.value())?;
            gpu_result(cudf::filling::sequence_i64(height, v, 0))
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
            gpu_result(cudf::filling::sequence_f32(height, v, 0.0))
        }
        GpuTypeId::Float64 => {
            let v: f64 = gpu_result(scalar.value())?;
            gpu_result(cudf::filling::sequence_f64(height, v, 0.0))
        }
        // Temporal scalar broadcast is not supported — temporal columns in aggregation
        // contexts (e.g. select(col("date").min())) would need type-aware extraction.
        // Pass-through operations (filter, sort, join, groupby) don't need scalar broadcast.
        // Null temporal scalars are handled via null_column_for_type above.
        _ => polars_bail!(ComputeError: "GPU engine: cannot broadcast scalar of type {:?}", tid),
    }
}

/// Evaluate a built-in function expression.
fn eval_ir_function(
    inputs: &[polars_plan::prelude::expr_ir::ExprIR],
    function: &IRFunctionExpr,
    expr_arena: &Arena<AExpr>,
    df: &GpuDataFrame,
) -> PolarsResult<GpuColumn> {
    match function {
        IRFunctionExpr::Abs => {
            let col = eval_expr(inputs[0].node(), expr_arena, df)?;
            gpu_result(col.unary_op(UnaryOp::Abs))
        }
        IRFunctionExpr::Negate => {
            let col = eval_expr(inputs[0].node(), expr_arena, df)?;
            // Negate: multiply by scalar -1 (avoids allocating a full-length column)
            let dtype = col.data_type();
            let tid = dtype.id();
            let neg_scalar = match tid {
                GpuTypeId::Int8 => gpu_result(cudf::Scalar::new(-1i8))?,
                GpuTypeId::Int16 => gpu_result(cudf::Scalar::new(-1i16))?,
                GpuTypeId::Int32 => gpu_result(cudf::Scalar::new(-1i32))?,
                GpuTypeId::Int64 => gpu_result(cudf::Scalar::new(-1i64))?,
                GpuTypeId::Float32 => gpu_result(cudf::Scalar::new(-1.0f32))?,
                GpuTypeId::Float64 => gpu_result(cudf::Scalar::new(-1.0f64))?,
                _ => {
                    polars_bail!(ComputeError: "GPU engine: Negate not supported for type {:?}", tid)
                }
            };
            gpu_result(col.binary_op_scalar(&neg_scalar, cudf::BinaryOp::Mul, dtype))
        }
        IRFunctionExpr::Boolean(bf) => eval_boolean_function(bf, inputs, expr_arena, df),
        IRFunctionExpr::FillNull => {
            // FillNull: inputs[0] is the column, inputs[1] is the fill value
            if inputs.len() != 2 {
                polars_bail!(ComputeError: "GPU engine: FillNull expects 2 inputs");
            }
            let col = eval_expr(inputs[0].node(), expr_arena, df)?;
            let fill = eval_expr(inputs[1].node(), expr_arena, df)?;
            // Use copy_if_else: where NOT null, use original; where null, use fill
            let is_valid = gpu_result(col.is_valid())?;
            gpu_result(col.copy_if_else(&fill, &is_valid))
        }
        IRFunctionExpr::DropNulls => {
            polars_bail!(ComputeError: "GPU engine: DropNulls not supported in expression context (use Filter)")
        }
        IRFunctionExpr::NullCount => {
            let col = eval_expr(inputs[0].node(), expr_arena, df)?;
            let nc = col.null_count() as u32;
            let height = df.height();
            gpu_result(GpuColumn::from_slice(&vec![nc; height]))
        }
        _ => {
            polars_bail!(ComputeError: "GPU engine: unsupported function expression {:?}", function)
        }
    }
}

/// Evaluate a boolean function.
fn eval_boolean_function(
    bf: &IRBooleanFunction,
    inputs: &[polars_plan::prelude::expr_ir::ExprIR],
    expr_arena: &Arena<AExpr>,
    df: &GpuDataFrame,
) -> PolarsResult<GpuColumn> {
    match bf {
        IRBooleanFunction::IsNull => {
            let col = eval_expr(inputs[0].node(), expr_arena, df)?;
            gpu_result(col.is_null())
        }
        IRBooleanFunction::IsNotNull => {
            let col = eval_expr(inputs[0].node(), expr_arena, df)?;
            gpu_result(col.is_valid())
        }
        IRBooleanFunction::IsNan => {
            let col = eval_expr(inputs[0].node(), expr_arena, df)?;
            gpu_result(col.is_nan())
        }
        IRBooleanFunction::IsNotNan => {
            let col = eval_expr(inputs[0].node(), expr_arena, df)?;
            gpu_result(col.is_not_nan())
        }
        IRBooleanFunction::IsFinite => {
            // is_finite = is_valid AND NOT(is_nan) AND NOT(is_inf)
            let col = eval_expr(inputs[0].node(), expr_arena, df)?;
            let is_valid = gpu_result(col.is_valid())?;
            let not_nan = gpu_result(col.is_not_nan())?;
            let not_inf = gpu_result(col.is_not_inf())?;
            let valid_and_not_nan = gpu_result(is_valid.binary_op(
                &not_nan,
                cudf::BinaryOp::LogicalAnd,
                GpuDataType::new(GpuTypeId::Bool8),
            ))?;
            gpu_result(valid_and_not_nan.binary_op(
                &not_inf,
                cudf::BinaryOp::LogicalAnd,
                GpuDataType::new(GpuTypeId::Bool8),
            ))
        }
        IRBooleanFunction::IsInfinite => {
            let col = eval_expr(inputs[0].node(), expr_arena, df)?;
            gpu_result(col.is_inf())
        }
        IRBooleanFunction::Any { ignore_nulls } => {
            let col = eval_expr(inputs[0].node(), expr_arena, df)?;
            let height = df.height();
            if !ignore_nulls && col.null_count() > 0 {
                // Propagate null: if there are nulls and we are not ignoring them,
                // check if any non-null value is true first
                let scalar =
                    gpu_result(col.reduce(ReduceOp::Any, GpuDataType::new(GpuTypeId::Bool8)))?;
                if scalar.is_valid() {
                    let val: bool = gpu_result(scalar.value())?;
                    if val {
                        // At least one true found -> result is true
                        broadcast_scalar(&scalar, height)
                    } else {
                        // All non-null are false but there are nulls -> result is null
                        let opts: Vec<Option<bool>> = vec![None; height];
                        gpu_result(cudf::Column::from_optional_bool(&opts))
                    }
                } else {
                    // All null -> result is null
                    let opts: Vec<Option<bool>> = vec![None; height];
                    gpu_result(cudf::Column::from_optional_bool(&opts))
                }
            } else {
                let scalar =
                    gpu_result(col.reduce(ReduceOp::Any, GpuDataType::new(GpuTypeId::Bool8)))?;
                broadcast_scalar(&scalar, height)
            }
        }
        IRBooleanFunction::All { ignore_nulls } => {
            let col = eval_expr(inputs[0].node(), expr_arena, df)?;
            let height = df.height();
            if !ignore_nulls && col.null_count() > 0 {
                // Propagate null: if there are nulls and we are not ignoring them,
                // check if any non-null value is false first
                let scalar =
                    gpu_result(col.reduce(ReduceOp::All, GpuDataType::new(GpuTypeId::Bool8)))?;
                if scalar.is_valid() {
                    let val: bool = gpu_result(scalar.value())?;
                    if !val {
                        // At least one false found -> result is false
                        broadcast_scalar(&scalar, height)
                    } else {
                        // All non-null are true but there are nulls -> result is null
                        let opts: Vec<Option<bool>> = vec![None; height];
                        gpu_result(cudf::Column::from_optional_bool(&opts))
                    }
                } else {
                    // All null -> result is null
                    let opts: Vec<Option<bool>> = vec![None; height];
                    gpu_result(cudf::Column::from_optional_bool(&opts))
                }
            } else {
                let scalar =
                    gpu_result(col.reduce(ReduceOp::All, GpuDataType::new(GpuTypeId::Bool8)))?;
                broadcast_scalar(&scalar, height)
            }
        }
        IRBooleanFunction::Not => {
            let col = eval_expr(inputs[0].node(), expr_arena, df)?;
            let tid = col.data_type().id();
            if tid == GpuTypeId::Bool8 {
                gpu_result(col.unary_op(UnaryOp::Not))
            } else {
                // Integer types use bitwise invert (Polars semantics)
                gpu_result(col.unary_op(UnaryOp::BitInvert))
            }
        }
        IRBooleanFunction::IsIn { nulls_equal } => {
            if inputs.len() != 2 {
                polars_bail!(ComputeError: "GPU engine: IsIn expects 2 inputs");
            }
            let col = eval_expr(inputs[0].node(), expr_arena, df)?;
            let values = eval_expr(inputs[1].node(), expr_arena, df)?;
            // cudf::Column::contains(haystack=self, needles=arg): for each element in `col`,
            // checks if it exists in `values`. This matches Polars' `col.is_in(values)` semantics.

            // Empty values set → nothing can be "in", return all false
            if values.len() == 0 {
                return gpu_result(GpuColumn::from_optional_bool(&vec![Some(false); col.len()]));
            }

            let result = gpu_result(values.contains(&col))?;
            if !nulls_equal {
                // When nulls_equal=false (default), null in col should produce null in output
                // cudf's contains returns false for null needles, but Polars wants null
                // Fix: where col is null, set result to null
                let col_valid = gpu_result(col.is_valid())?;
                let null_col = null_column_for_type(
                    cudf::types::DataType::new(cudf::types::TypeId::Bool8),
                    col.len(),
                )?;
                // Where col is valid keep result, where col is null use null_col
                gpu_result(result.copy_if_else(&null_col, &col_valid))
            } else {
                // nulls_equal=true: null IS IN {null} should be true
                // cudf's contains returns false for null needles, fix:
                // where col is null AND values contains null → true
                let values_has_null = values.null_count() > 0;
                if values_has_null {
                    let col_is_null = gpu_result(col.is_null())?;
                    // Build a column of all true, same length as col
                    let true_col =
                        gpu_result(GpuColumn::from_optional_bool(&vec![Some(true); col.len()]))?;
                    // Where col is null, override result to true
                    gpu_result(true_col.copy_if_else(&result, &col_is_null))
                } else {
                    Ok(result)
                }
            }
        }
        _ => polars_bail!(ComputeError: "GPU engine: unsupported boolean function {:?}", bf),
    }
}
