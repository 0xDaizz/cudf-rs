//! Type and operator mapping between Polars and cudf.

use cudf::BinaryOp;
use cudf::types::{DataType as GpuDataType, TypeId as GpuTypeId};
use polars_core::prelude::DataType;
use polars_error::{PolarsResult, polars_bail};
use polars_plan::dsl::Operator;

/// Map a Polars `Operator` to a cudf `BinaryOp`.
pub fn map_operator(op: Operator) -> PolarsResult<BinaryOp> {
    match op {
        Operator::Plus => Ok(BinaryOp::Add),
        Operator::Minus => Ok(BinaryOp::Sub),
        Operator::Multiply => Ok(BinaryOp::Mul),
        Operator::RustDivide => Ok(BinaryOp::Div),
        Operator::TrueDivide => Ok(BinaryOp::TrueDiv),
        Operator::FloorDivide => Ok(BinaryOp::FloorDiv),
        Operator::Modulus => Ok(BinaryOp::Mod),
        Operator::Eq => Ok(BinaryOp::Equal),
        Operator::EqValidity => Ok(BinaryOp::NullEquals),
        Operator::NotEq => Ok(BinaryOp::NotEqual),
        Operator::NotEqValidity => Ok(BinaryOp::NullNotEquals),
        Operator::Lt => Ok(BinaryOp::Less),
        Operator::LtEq => Ok(BinaryOp::LessEqual),
        Operator::Gt => Ok(BinaryOp::Greater),
        Operator::GtEq => Ok(BinaryOp::GreaterEqual),
        Operator::And => Ok(BinaryOp::BitwiseAnd),
        Operator::Or => Ok(BinaryOp::BitwiseOr),
        Operator::Xor => Ok(BinaryOp::BitwiseXor),
        Operator::LogicalAnd => Ok(BinaryOp::LogicalAnd),
        Operator::LogicalOr => Ok(BinaryOp::LogicalOr),
    }
}

/// Returns true if the operator produces a boolean result.
pub fn is_comparison(op: Operator) -> bool {
    matches!(
        op,
        Operator::Eq
            | Operator::EqValidity
            | Operator::NotEq
            | Operator::NotEqValidity
            | Operator::Lt
            | Operator::LtEq
            | Operator::Gt
            | Operator::GtEq
            | Operator::LogicalAnd
            | Operator::LogicalOr
    )
}

/// Map a Polars `DataType` to a cudf `DataType`.
pub fn map_dtype(dtype: &DataType) -> PolarsResult<GpuDataType> {
    let type_id = match dtype {
        DataType::Boolean => GpuTypeId::Bool8,
        DataType::Int8 => GpuTypeId::Int8,
        DataType::Int16 => GpuTypeId::Int16,
        DataType::Int32 => GpuTypeId::Int32,
        DataType::Int64 => GpuTypeId::Int64,
        DataType::UInt8 => GpuTypeId::Uint8,
        DataType::UInt16 => GpuTypeId::Uint16,
        DataType::UInt32 => GpuTypeId::Uint32,
        DataType::UInt64 => GpuTypeId::Uint64,
        DataType::Float32 => GpuTypeId::Float32,
        DataType::Float64 => GpuTypeId::Float64,
        DataType::String => GpuTypeId::String,
        _ => polars_bail!(ComputeError: "GPU engine: unsupported dtype {:?}", dtype),
    };
    Ok(GpuDataType::new(type_id))
}
