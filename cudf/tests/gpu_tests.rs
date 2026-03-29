//! GPU integration tests. Only run on machines with CUDA + libcudf.
//!
//! Run with: `cargo test -p cudf --features gpu-tests`
//!
//! These tests will be skipped if the gpu-tests feature is not enabled.

#![cfg(feature = "gpu-tests")]

use cudf::*;

// ── Column roundtrip tests ───────────────────────────────────────

#[test]
fn column_i32_roundtrip() {
    let data = vec![1i32, 2, 3, 4, 5];
    let col = Column::from_slice(&data).unwrap();
    assert_eq!(col.len(), 5);
    assert_eq!(col.data_type().id(), TypeId::Int32);
    assert!(!col.has_nulls());
    let result: Vec<i32> = col.to_vec().unwrap();
    assert_eq!(data, result);
}

#[test]
fn column_f64_roundtrip() {
    let data = vec![1.1f64, 2.2, 3.3];
    let col = Column::from_slice(&data).unwrap();
    assert_eq!(col.data_type().id(), TypeId::Float64);
    let result: Vec<f64> = col.to_vec().unwrap();
    assert_eq!(data, result);
}

#[test]
fn column_all_numeric_types() {
    let _ = Column::from_slice(&[1i8, 2, 3]).unwrap();
    let _ = Column::from_slice(&[1i16, 2, 3]).unwrap();
    let _ = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let _ = Column::from_slice(&[1i64, 2, 3]).unwrap();
    let _ = Column::from_slice(&[1u8, 2, 3]).unwrap();
    let _ = Column::from_slice(&[1u16, 2, 3]).unwrap();
    let _ = Column::from_slice(&[1u32, 2, 3]).unwrap();
    let _ = Column::from_slice(&[1u64, 2, 3]).unwrap();
    let _ = Column::from_slice(&[1.0f32, 2.0, 3.0]).unwrap();
    let _ = Column::from_slice(&[1.0f64, 2.0, 3.0]).unwrap();
}

#[test]
fn column_type_mismatch_error() {
    let col = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let err = col.to_vec::<f64>().unwrap_err();
    match err {
        CudfError::TypeMismatch { .. } => {}
        _ => panic!("expected TypeMismatch, got {:?}", err),
    }
}

#[test]
fn column_empty_slice() {
    let data: Vec<i32> = vec![];
    let col = Column::from_slice(&data).unwrap();
    assert_eq!(col.len(), 0);
    assert!(col.is_empty());
    let result: Vec<i32> = col.to_vec().unwrap();
    assert!(result.is_empty());
}

#[test]
fn column_single_element() {
    let col = Column::from_slice(&[42i32]).unwrap();
    assert_eq!(col.len(), 1);
    let result: Vec<i32> = col.to_vec().unwrap();
    assert_eq!(result, vec![42]);
}

// ── Column::empty tests ─────────────────────────────────────────

#[test]
fn column_empty_numeric() {
    let col = Column::empty(DataType::new(TypeId::Int32), 10).unwrap();
    assert_eq!(col.len(), 10);
    assert!(col.is_nullable());
    assert!(col.has_nulls());
    assert_eq!(col.null_count(), 10);
}

#[test]
fn column_empty_zero_size() {
    let col = Column::empty(DataType::new(TypeId::Int32), 0).unwrap();
    assert_eq!(col.len(), 0);
    assert!(col.is_empty());
}

#[test]
fn column_empty_non_numeric_rejected() {
    let err = Column::empty(DataType::new(TypeId::String), 10).unwrap_err();
    match err {
        CudfError::InvalidArgument(_) => {}
        _ => panic!("expected InvalidArgument"),
    }
}

#[test]
fn column_empty_list_rejected() {
    let err = Column::empty(DataType::new(TypeId::List), 5).unwrap_err();
    match err {
        CudfError::InvalidArgument(_) => {}
        _ => panic!("expected InvalidArgument"),
    }
}

// ── Column display ──────────────────────────────────────────────

#[test]
fn column_display() {
    let col = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let display = format!("{}", col);
    assert!(display.contains("Int32"));
    assert!(display.contains("len=3"));
    assert!(display.contains("nulls=0"));
}

// ── Table tests ─────────────────────────────────────────────────

#[test]
fn table_basic() {
    let a = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let b = Column::from_slice(&[4.0f64, 5.0, 6.0]).unwrap();
    let table = Table::new(vec![a, b]).unwrap();
    assert_eq!(table.num_columns(), 2);
    assert_eq!(table.num_rows(), 3);
    assert!(!table.is_empty());
}

#[test]
fn table_empty_allowed() {
    let table = Table::new(vec![]).unwrap();
    assert_eq!(table.num_columns(), 0);
}

#[test]
fn table_single_column() {
    let a = Column::from_slice(&[10i32, 20]).unwrap();
    let table = Table::new(vec![a]).unwrap();
    assert_eq!(table.num_columns(), 1);
    assert_eq!(table.num_rows(), 2);
}

#[test]
fn table_column_length_mismatch() {
    let a = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let b = Column::from_slice(&[4.0f64, 5.0]).unwrap();
    let err = Table::new(vec![a, b]).unwrap_err();
    match err {
        CudfError::InvalidArgument(_) => {}
        _ => panic!("expected InvalidArgument"),
    }
}

#[test]
fn table_get_column() {
    let a = Column::from_slice(&[10i32, 20, 30]).unwrap();
    let table = Table::new(vec![a]).unwrap();
    let col = table.column(0).unwrap();
    let data: Vec<i32> = col.to_vec().unwrap();
    assert_eq!(data, vec![10, 20, 30]);
}

#[test]
fn table_column_out_of_bounds() {
    let a = Column::from_slice(&[1i32]).unwrap();
    let table = Table::new(vec![a]).unwrap();
    let err = table.column(5).unwrap_err();
    match err {
        CudfError::IndexOutOfBounds { .. } => {}
        _ => panic!("expected IndexOutOfBounds"),
    }
}

#[test]
fn table_into_columns() {
    let a = Column::from_slice(&[1i32, 2]).unwrap();
    let b = Column::from_slice(&[3i32, 4]).unwrap();
    let table = Table::new(vec![a, b]).unwrap();
    let cols = table.into_columns().unwrap();
    assert_eq!(cols.len(), 2);
    let data_a: Vec<i32> = cols[0].to_vec().unwrap();
    let data_b: Vec<i32> = cols[1].to_vec().unwrap();
    assert_eq!(data_a, vec![1, 2]);
    assert_eq!(data_b, vec![3, 4]);
}

#[test]
fn table_display() {
    let a = Column::from_slice(&[1i32, 2]).unwrap();
    let table = Table::new(vec![a]).unwrap();
    let display = format!("{}", table);
    assert!(display.contains("columns=1"));
    assert!(display.contains("rows=2"));
}

// ── Scalar tests ────────────────────────────────────────────────

#[test]
fn scalar_i32() {
    let s = Scalar::new(42i32).unwrap();
    assert!(s.is_valid());
    assert_eq!(s.data_type().id(), TypeId::Int32);
    assert_eq!(s.value::<i32>().unwrap(), 42);
}

#[test]
fn scalar_f64() {
    let s = Scalar::new(3.14f64).unwrap();
    assert_eq!(s.data_type().id(), TypeId::Float64);
    assert_eq!(s.value::<f64>().unwrap(), 3.14);
}

#[test]
fn scalar_i8() {
    let s = Scalar::new(127i8).unwrap();
    assert_eq!(s.value::<i8>().unwrap(), 127);
}

#[test]
fn scalar_u64() {
    let s = Scalar::new(u64::MAX).unwrap();
    assert_eq!(s.value::<u64>().unwrap(), u64::MAX);
}

#[test]
fn scalar_null() {
    let s = Scalar::null(DataType::new(TypeId::Int32)).unwrap();
    assert!(!s.is_valid());
    let err = s.value::<i32>().unwrap_err();
    match err {
        CudfError::InvalidArgument(_) => {}
        _ => panic!("expected error for null scalar value"),
    }
}

#[test]
fn scalar_type_mismatch() {
    let s = Scalar::new(42i32).unwrap();
    let err = s.value::<f64>().unwrap_err();
    match err {
        CudfError::TypeMismatch { .. } => {}
        _ => panic!("expected TypeMismatch"),
    }
}

#[test]
fn scalar_try_from_i32() {
    let s: Scalar = 42i32.try_into().unwrap();
    assert_eq!(s.value::<i32>().unwrap(), 42);
}

#[test]
fn scalar_try_from_f64() {
    let s: Scalar = 2.718f64.try_into().unwrap();
    assert_eq!(s.value::<f64>().unwrap(), 2.718);
}

#[test]
fn scalar_display() {
    let s = Scalar::new(42i32).unwrap();
    let display = format!("{}", s);
    assert!(display.contains("Int32"));
    assert!(display.contains("valid=true"));
}

#[test]
fn scalar_null_display() {
    let s = Scalar::null(DataType::new(TypeId::Float64)).unwrap();
    let display = format!("{}", s);
    assert!(display.contains("valid=false"));
}

// ── Sorting tests ───────────────────────────────────────────────

#[test]
fn sort_table_ascending() {
    let col = Column::from_slice(&[3i32, 1, 2]).unwrap();
    let table = Table::new(vec![col]).unwrap();
    let sorted = table
        .sort(&[SortOrder::Ascending], &[NullOrder::After])
        .unwrap();
    let result: Vec<i32> = sorted.column(0).unwrap().to_vec().unwrap();
    assert_eq!(result, vec![1, 2, 3]);
}

#[test]
fn sort_table_descending() {
    let col = Column::from_slice(&[3i32, 1, 2]).unwrap();
    let table = Table::new(vec![col]).unwrap();
    let sorted = table
        .sort(&[SortOrder::Descending], &[NullOrder::Before])
        .unwrap();
    let result: Vec<i32> = sorted.column(0).unwrap().to_vec().unwrap();
    assert_eq!(result, vec![3, 2, 1]);
}

#[test]
fn sorted_order() {
    let col = Column::from_slice(&[30i32, 10, 20]).unwrap();
    let table = Table::new(vec![col]).unwrap();
    let order = table
        .sorted_order(&[SortOrder::Ascending], &[NullOrder::After])
        .unwrap();
    assert_eq!(order.len(), 3);
    assert_eq!(order.data_type().id(), TypeId::Int32);
}

#[test]
fn is_sorted_true() {
    let col = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let table = Table::new(vec![col]).unwrap();
    let result = table
        .is_sorted(&[SortOrder::Ascending], &[NullOrder::After])
        .unwrap();
    assert!(result);
}

#[test]
fn is_sorted_false() {
    let col = Column::from_slice(&[3i32, 1, 2]).unwrap();
    let table = Table::new(vec![col]).unwrap();
    let result = table
        .is_sorted(&[SortOrder::Ascending], &[NullOrder::After])
        .unwrap();
    assert!(!result);
}

// ── Cast tests ──────────────────────────────────────────────────

#[test]
fn column_cast_i32_to_f64() {
    let col = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let casted = col.cast(DataType::new(TypeId::Float64)).unwrap();
    assert_eq!(casted.data_type().id(), TypeId::Float64);
    let data: Vec<f64> = casted.to_vec().unwrap();
    assert_eq!(data, vec![1.0, 2.0, 3.0]);
}

#[test]
fn column_cast_f64_to_i32() {
    let col = Column::from_slice(&[1.5f64, 2.7, 3.1]).unwrap();
    let casted = col.cast(DataType::new(TypeId::Int32)).unwrap();
    assert_eq!(casted.data_type().id(), TypeId::Int32);
    let data: Vec<i32> = casted.to_vec().unwrap();
    // libcudf truncates towards zero
    assert_eq!(data, vec![1, 2, 3]);
}

// ── Binary operation tests ──────────────────────────────────────

#[test]
fn binary_op_add() {
    let a = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let b = Column::from_slice(&[10i32, 20, 30]).unwrap();
    let result = a
        .binary_op(&b, BinaryOp::Add, DataType::new(TypeId::Int32))
        .unwrap();
    let data: Vec<i32> = result.to_vec().unwrap();
    assert_eq!(data, vec![11, 22, 33]);
}

#[test]
fn binary_op_sub() {
    let a = Column::from_slice(&[10i32, 20, 30]).unwrap();
    let b = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let result = a
        .binary_op(&b, BinaryOp::Sub, DataType::new(TypeId::Int32))
        .unwrap();
    let data: Vec<i32> = result.to_vec().unwrap();
    assert_eq!(data, vec![9, 18, 27]);
}

#[test]
fn binary_op_mul() {
    let a = Column::from_slice(&[2i32, 3, 4]).unwrap();
    let b = Column::from_slice(&[5i32, 6, 7]).unwrap();
    let result = a
        .binary_op(&b, BinaryOp::Mul, DataType::new(TypeId::Int32))
        .unwrap();
    let data: Vec<i32> = result.to_vec().unwrap();
    assert_eq!(data, vec![10, 18, 28]);
}

#[test]
fn binary_op_scalar_mul() {
    let col = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let scalar = Scalar::new(10i32).unwrap();
    let result = col
        .binary_op_scalar(&scalar, BinaryOp::Mul, DataType::new(TypeId::Int32))
        .unwrap();
    let data: Vec<i32> = result.to_vec().unwrap();
    assert_eq!(data, vec![10, 20, 30]);
}

// ── Unary operation tests ───────────────────────────────────────

#[test]
fn unary_abs() {
    let col = Column::from_slice(&[-1i32, -2, 3]).unwrap();
    let result = col.unary_op(UnaryOp::Abs).unwrap();
    let data: Vec<i32> = result.to_vec().unwrap();
    assert_eq!(data, vec![1, 2, 3]);
}

#[test]
fn unary_sqrt() {
    let col = Column::from_slice(&[1.0f64, 4.0, 9.0]).unwrap();
    let result = col.unary_op(UnaryOp::Sqrt).unwrap();
    let data: Vec<f64> = result.to_vec().unwrap();
    assert_eq!(data, vec![1.0, 2.0, 3.0]);
}

#[test]
fn unary_is_null_no_nulls() {
    let col = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let nulls = col.is_null().unwrap();
    assert_eq!(nulls.len(), 3);
}

#[test]
fn unary_is_valid_no_nulls() {
    let col = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let valids = col.is_valid().unwrap();
    assert_eq!(valids.len(), 3);
}

// ── Reduction tests ─────────────────────────────────────────────

#[test]
fn reduce_sum() {
    let col = Column::from_slice(&[1i32, 2, 3, 4, 5]).unwrap();
    let result = col
        .reduce(reduction::ReduceOp::Sum, DataType::new(TypeId::Int64))
        .unwrap();
    assert!(result.is_valid());
    assert_eq!(result.value::<i64>().unwrap(), 15);
}

#[test]
fn reduce_min() {
    let col = Column::from_slice(&[5i32, 2, 8, 1, 9]).unwrap();
    let result = col
        .reduce(reduction::ReduceOp::Min, DataType::new(TypeId::Int32))
        .unwrap();
    assert_eq!(result.value::<i32>().unwrap(), 1);
}

#[test]
fn reduce_max() {
    let col = Column::from_slice(&[5i32, 2, 8, 1, 9]).unwrap();
    let result = col
        .reduce(reduction::ReduceOp::Max, DataType::new(TypeId::Int32))
        .unwrap();
    assert_eq!(result.value::<i32>().unwrap(), 9);
}

// ── Scan tests ──────────────────────────────────────────────────

#[test]
fn scan_cumsum_inclusive() {
    let col = Column::from_slice(&[1i32, 2, 3, 4]).unwrap();
    let result = col.scan(reduction::ScanOp::Sum, true).unwrap();
    let data: Vec<i32> = result.to_vec().unwrap();
    assert_eq!(data, vec![1, 3, 6, 10]);
}

#[test]
fn scan_cumsum_exclusive() {
    let col = Column::from_slice(&[1i32, 2, 3, 4]).unwrap();
    let result = col.scan(reduction::ScanOp::Sum, false).unwrap();
    let data: Vec<i32> = result.to_vec().unwrap();
    assert_eq!(data, vec![0, 1, 3, 6]);
}

// ── Stream compaction tests ─────────────────────────────────────

#[test]
fn column_drop_nulls_no_nulls() {
    let col = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let result = col.drop_nulls().unwrap();
    assert_eq!(result.len(), 3);
}

#[test]
fn column_distinct_count() {
    let col = Column::from_slice(&[1i32, 2, 2, 3, 3, 3]).unwrap();
    let count = col.distinct_count().unwrap();
    assert_eq!(count, 3);
}

#[test]
fn table_unique() {
    let col = Column::from_slice(&[1i32, 2, 2, 3, 3, 3]).unwrap();
    let table = Table::new(vec![col]).unwrap();
    let unique = table
        .unique(&[0], stream_compaction::DuplicateKeepOption::First)
        .unwrap();
    assert_eq!(unique.num_rows(), 3);
}

// ── Quantile tests ──────────────────────────────────────────────

#[test]
fn column_quantile_median() {
    let col = Column::from_slice(&[1.0f64, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let q = col
        .quantile(&[0.5], quantiles::Interpolation::Linear)
        .unwrap();
    assert_eq!(q.len(), 1);
    let data: Vec<f64> = q.to_vec().unwrap();
    assert_eq!(data, vec![3.0]);
}

// ── Null mask tests ─────────────────────────────────────────────

#[test]
fn null_mask_no_nulls() {
    let col = Column::from_slice(&[1i32, 2, 3]).unwrap();
    if col.is_nullable() {
        let mask = col.null_mask_to_host().unwrap();
        // All bits should be set (valid)
        assert!(!mask.is_empty());
    }
    // If not nullable, there is no mask -- this is also correct
}

#[test]
fn null_mask_all_nulls() {
    let col = Column::empty(DataType::new(TypeId::Int32), 8).unwrap();
    let mask = col.null_mask_to_host().unwrap();
    assert_eq!(mask.len(), 1); // 8 bits = 1 byte
    assert_eq!(mask[0], 0x00); // all null
}
