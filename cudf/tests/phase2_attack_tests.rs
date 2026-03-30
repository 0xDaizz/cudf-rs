//! Phase 2 re-audit attack tests.
//! Targets areas NOT covered by the previous 78-test audit:
//! - ChunkedParquetReader/Writer
//! - HashJoin class
//! - DLPack roundtrip
//! - PackedTable roundtrip / contiguous_split
//! - GroupBy scan
//! - New string operations (padding, repeat, findall, case, combine, etc.)
//! - New list operations
//! - from_strings edge cases
//! - from_optional_bool
//! - Bool column roundtrip
//! - Scalar bool support
//! - GroupBy replace_nulls / get_groups
//!
//! Run with: `cargo test -p cudf --features gpu-tests --test phase2_attack_tests`

#![cfg(feature = "gpu-tests")]

use cudf::*;

// ══════════════════════════════════════════════════════════════
// 1. BOOL COLUMN ROUNDTRIP (new in Phase 1 fixes)
// ══════════════════════════════════════════════════════════════

#[test]
fn bool_column_roundtrip() {
    let data = vec![true, false, true, true, false];
    let col = Column::from_slice(&data).unwrap();
    assert_eq!(col.len(), 5);
    assert_eq!(col.data_type().id(), TypeId::Bool8);
    let result: Vec<bool> = col.to_vec().unwrap();
    assert_eq!(result, data);
}

#[test]
fn bool_column_empty() {
    let data: Vec<bool> = vec![];
    let col = Column::from_slice(&data).unwrap();
    assert_eq!(col.len(), 0);
    let result: Vec<bool> = col.to_vec().unwrap();
    assert!(result.is_empty());
}

#[test]
fn bool_column_single_true() {
    let col = Column::from_slice(&[true]).unwrap();
    let result: Vec<bool> = col.to_vec().unwrap();
    assert_eq!(result, vec![true]);
}

#[test]
fn bool_column_single_false() {
    let col = Column::from_slice(&[false]).unwrap();
    let result: Vec<bool> = col.to_vec().unwrap();
    assert_eq!(result, vec![false]);
}

#[test]
fn bool_column_all_same_true() {
    let data = vec![true; 100];
    let col = Column::from_slice(&data).unwrap();
    let result: Vec<bool> = col.to_vec().unwrap();
    assert_eq!(result, data);
}

#[test]
fn bool_column_all_same_false() {
    let data = vec![false; 100];
    let col = Column::from_slice(&data).unwrap();
    let result: Vec<bool> = col.to_vec().unwrap();
    assert_eq!(result, data);
}

// ══════════════════════════════════════════════════════════════
// 2. FROM_OPTIONAL_BOOL (new function)
// ══════════════════════════════════════════════════════════════

#[test]
fn from_optional_bool_basic() {
    let data = vec![Some(true), None, Some(false), Some(true), None];
    let col = Column::from_optional_bool(&data).unwrap();
    assert_eq!(col.len(), 5);
    assert_eq!(col.data_type().id(), TypeId::Bool8);
    assert!(col.has_nulls());
    assert_eq!(col.null_count(), 2);
}

#[test]
fn from_optional_bool_all_none() {
    let data = vec![None, None, None];
    let col = Column::from_optional_bool(&data).unwrap();
    assert_eq!(col.null_count(), 3);
}

#[test]
fn from_optional_bool_all_some() {
    let data = vec![Some(true), Some(false), Some(true)];
    let col = Column::from_optional_bool(&data).unwrap();
    assert_eq!(col.null_count(), 0);
}

#[test]
fn from_optional_bool_empty() {
    let data: Vec<Option<bool>> = vec![];
    let col = Column::from_optional_bool(&data).unwrap();
    assert_eq!(col.len(), 0);
}

#[test]
fn from_optional_bool_to_optional_vec_roundtrip() {
    let data = vec![Some(true), None, Some(false)];
    let col = Column::from_optional_bool(&data).unwrap();
    let result: Vec<Option<bool>> = col.to_optional_vec().unwrap();
    assert_eq!(result, data);
}

// ══════════════════════════════════════════════════════════════
// 3. SCALAR BOOL (new support)
// ══════════════════════════════════════════════════════════════

#[test]
fn scalar_bool_true() {
    let s = Scalar::new(true).unwrap();
    assert!(s.is_valid());
    assert_eq!(s.data_type().id(), TypeId::Bool8);
    assert_eq!(s.value::<bool>().unwrap(), true);
}

#[test]
fn scalar_bool_false() {
    let s = Scalar::new(false).unwrap();
    assert_eq!(s.value::<bool>().unwrap(), false);
}

#[test]
fn scalar_bool_set_valid() {
    let mut s = Scalar::new(true).unwrap();
    assert!(s.is_valid());
    s.set_valid(false).unwrap();
    assert!(!s.is_valid());
    s.set_valid(true).unwrap();
    assert!(s.is_valid());
}

// ══════════════════════════════════════════════════════════════
// 4. FROM_STRINGS EDGE CASES
// ══════════════════════════════════════════════════════════════

#[test]
fn from_strings_basic() {
    let col = Column::from_strings(&["hello", "world", "!"]).unwrap();
    assert_eq!(col.len(), 3);
    assert_eq!(col.data_type().id(), TypeId::String);
}

#[test]
fn from_strings_empty_slice() {
    let data: Vec<&str> = vec![];
    let col = Column::from_strings(&data).unwrap();
    assert_eq!(col.len(), 0);
}

#[test]
fn from_strings_empty_strings() {
    let col = Column::from_strings(&["", "", ""]).unwrap();
    assert_eq!(col.len(), 3);
}

#[test]
fn from_strings_unicode() {
    let col = Column::from_strings(&["한글", "日本語", "中文", "emoji🚀"]).unwrap();
    assert_eq!(col.len(), 4);
}

#[test]
fn from_strings_mixed_lengths() {
    let col = Column::from_strings(&["a", "abcdefghij", "", "xyz"]).unwrap();
    assert_eq!(col.len(), 4);
}

#[test]
fn from_strings_single_element() {
    let col = Column::from_strings(&["only"]).unwrap();
    assert_eq!(col.len(), 1);
}

#[test]
fn from_strings_with_newlines_and_tabs() {
    let col = Column::from_strings(&["hello\nworld", "tab\there", "cr\rline"]).unwrap();
    assert_eq!(col.len(), 3);
}

#[test]
fn from_strings_owned_string_vec() {
    let owned: Vec<String> = vec!["a".into(), "b".into(), "c".into()];
    let col = Column::from_strings(&owned).unwrap();
    assert_eq!(col.len(), 3);
}

// ══════════════════════════════════════════════════════════════
// 5. TO_OPTIONAL_VEC FOR VARIOUS TYPES
// ══════════════════════════════════════════════════════════════

#[test]
fn to_optional_vec_i32_with_nulls() {
    let data = vec![Some(1), None, Some(3), None, Some(5)];
    let col = Column::from_optional_i32(&data).unwrap();
    let result: Vec<Option<i32>> = col.to_optional_vec().unwrap();
    assert_eq!(result, data);
}

#[test]
fn to_optional_vec_f64_with_nulls() {
    let data = vec![Some(1.1), None, Some(3.3)];
    let col = Column::from_optional_f64(&data).unwrap();
    let result: Vec<Option<f64>> = col.to_optional_vec().unwrap();
    assert_eq!(result, data);
}

#[test]
fn to_optional_vec_no_nulls() {
    let col = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let result: Vec<Option<i32>> = col.to_optional_vec().unwrap();
    assert_eq!(result, vec![Some(1), Some(2), Some(3)]);
}

#[test]
fn to_vec_rejects_nullable_column() {
    let col = Column::from_optional_i32(&[Some(1), None, Some(3)]).unwrap();
    let err = col.to_vec::<i32>().unwrap_err();
    match err {
        CudfError::InvalidArgument(_) => {}
        _ => panic!("expected InvalidArgument for to_vec on nullable column"),
    }
}

#[test]
fn to_optional_vec_type_mismatch() {
    let col = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let err = col.to_optional_vec::<f64>().unwrap_err();
    match err {
        CudfError::TypeMismatch { .. } => {}
        _ => panic!("expected TypeMismatch"),
    }
}

// ══════════════════════════════════════════════════════════════
// 6. HASH JOIN
// ══════════════════════════════════════════════════════════════

#[test]
fn hash_join_inner() {
    let build = Table::new(vec![Column::from_slice(&[2i32, 3, 5]).unwrap()]).unwrap();
    let hj = join::HashJoin::new(&build).unwrap();

    let probe = Table::new(vec![Column::from_slice(&[1i32, 2, 3, 4, 5]).unwrap()]).unwrap();
    let result = hj.inner_join(&probe).unwrap();
    // Should match 2, 3, 5 => 3 result rows
    assert_eq!(result.left_indices.len(), 3);
    assert_eq!(result.right_indices.len(), 3);
}

#[test]
fn hash_join_left() {
    let build = Table::new(vec![Column::from_slice(&[2i32, 4]).unwrap()]).unwrap();
    let hj = join::HashJoin::new(&build).unwrap();

    let probe = Table::new(vec![Column::from_slice(&[1i32, 2, 3, 4]).unwrap()]).unwrap();
    let result = hj.left_join(&probe).unwrap();
    // Left join preserves all probe rows => 4
    assert_eq!(result.left_indices.len(), 4);
}

#[test]
fn hash_join_full() {
    let build = Table::new(vec![Column::from_slice(&[2i32, 4]).unwrap()]).unwrap();
    let hj = join::HashJoin::new(&build).unwrap();

    let probe = Table::new(vec![Column::from_slice(&[1i32, 2, 3]).unwrap()]).unwrap();
    let result = hj.full_join(&probe).unwrap();
    // Full: 1(L only), 2(match), 3(L only), 4(R only) => 4
    assert_eq!(result.left_indices.len(), 4);
}

#[test]
fn hash_join_inner_size_estimate() {
    let build = Table::new(vec![Column::from_slice(&[1i32, 2, 3]).unwrap()]).unwrap();
    let hj = join::HashJoin::new(&build).unwrap();

    let probe = Table::new(vec![Column::from_slice(&[2i32, 3, 4]).unwrap()]).unwrap();
    let size = hj.inner_join_size(&probe).unwrap();
    assert_eq!(size, 2);
}

#[test]
fn hash_join_reuse_build() {
    // Build once, probe multiple times
    let build = Table::new(vec![Column::from_slice(&[10i32, 20, 30]).unwrap()]).unwrap();
    let hj = join::HashJoin::new(&build).unwrap();

    let probe1 = Table::new(vec![Column::from_slice(&[10i32, 20]).unwrap()]).unwrap();
    let probe2 = Table::new(vec![Column::from_slice(&[20i32, 30, 40]).unwrap()]).unwrap();

    let r1 = hj.inner_join(&probe1).unwrap();
    assert_eq!(r1.left_indices.len(), 2);

    let r2 = hj.inner_join(&probe2).unwrap();
    assert_eq!(r2.left_indices.len(), 2);
}

// ══════════════════════════════════════════════════════════════
// 7. DLPACK ROUNDTRIP
// ══════════════════════════════════════════════════════════════

#[test]
fn dlpack_roundtrip_single_column() {
    let col = Column::from_slice(&[1.0f64, 2.0, 3.0, 4.0]).unwrap();
    let table = Table::new(vec![col]).unwrap();

    let tensor = DLPackTensor::from_table(&table).unwrap();
    assert!(tensor.as_raw_ptr() != 0);

    let restored = tensor.to_table().unwrap();
    assert_eq!(restored.num_columns(), 1);
    assert_eq!(restored.num_rows(), 4);
    let data: Vec<f64> = restored.column(0).unwrap().to_vec().unwrap();
    assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn dlpack_roundtrip_multi_column() {
    let a = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let b = Column::from_slice(&[4i32, 5, 6]).unwrap();
    let table = Table::new(vec![a, b]).unwrap();

    let tensor = DLPackTensor::from_table(&table).unwrap();
    let restored = tensor.to_table().unwrap();
    assert_eq!(restored.num_rows(), 3);
    // DLPack returns a 2D tensor which becomes columns
    assert!(restored.num_columns() >= 1);
}

// ══════════════════════════════════════════════════════════════
// 8. PACKED TABLE ROUNDTRIP
// ══════════════════════════════════════════════════════════════

#[test]
fn packed_table_roundtrip() {
    let a = Column::from_slice(&[10i32, 20, 30]).unwrap();
    let b = Column::from_slice(&[1.5f64, 2.5, 3.5]).unwrap();
    let table = Table::new(vec![a, b]).unwrap();

    let packed = PackedTable::pack(&table).unwrap();

    // Check metadata
    let meta = packed.metadata().unwrap();
    assert!(!meta.is_empty());

    // Check GPU data size
    let gpu_size = packed.gpu_data_size().unwrap();
    assert!(gpu_size > 0);

    // Unpack and verify
    let restored = packed.unpack().unwrap();
    assert_eq!(restored.num_columns(), 2);
    assert_eq!(restored.num_rows(), 3);

    let col0: Vec<i32> = restored.column(0).unwrap().to_vec().unwrap();
    assert_eq!(col0, vec![10, 20, 30]);
    let col1: Vec<f64> = restored.column(1).unwrap().to_vec().unwrap();
    assert_eq!(col1, vec![1.5, 2.5, 3.5]);
}

#[test]
fn packed_table_single_column() {
    let col = Column::from_slice(&[1i64, 2, 3, 4, 5]).unwrap();
    let table = Table::new(vec![col]).unwrap();
    let packed = PackedTable::pack(&table).unwrap();
    let restored = packed.unpack().unwrap();
    let data: Vec<i64> = restored.column(0).unwrap().to_vec().unwrap();
    assert_eq!(data, vec![1, 2, 3, 4, 5]);
}

// ══════════════════════════════════════════════════════════════
// 9. CONTIGUOUS SPLIT
// ══════════════════════════════════════════════════════════════

#[test]
fn contiguous_split_basic() {
    let col = Column::from_slice(&[1i32, 2, 3, 4, 5, 6]).unwrap();
    let table = Table::new(vec![col]).unwrap();

    let result = SplitResult::split(&table, &[2, 4]).unwrap();
    assert_eq!(result.num_parts(), 3);

    let part0 = result.get(0).unwrap();
    let t0 = part0.unpack().unwrap();
    assert_eq!(t0.num_rows(), 2);
    let d0: Vec<i32> = t0.column(0).unwrap().to_vec().unwrap();
    assert_eq!(d0, vec![1, 2]);

    let part1 = result.get(1).unwrap();
    let t1 = part1.unpack().unwrap();
    assert_eq!(t1.num_rows(), 2);
    let d1: Vec<i32> = t1.column(0).unwrap().to_vec().unwrap();
    assert_eq!(d1, vec![3, 4]);

    let part2 = result.get(2).unwrap();
    let t2 = part2.unpack().unwrap();
    assert_eq!(t2.num_rows(), 2);
    let d2: Vec<i32> = t2.column(0).unwrap().to_vec().unwrap();
    assert_eq!(d2, vec![5, 6]);
}

#[test]
fn contiguous_split_out_of_bounds_partition() {
    let col = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let table = Table::new(vec![col]).unwrap();
    let result = SplitResult::split(&table, &[1]).unwrap();
    assert_eq!(result.num_parts(), 2);
    match result.get(5) {
        Err(CudfError::IndexOutOfBounds { .. }) => {}
        Err(e) => panic!("expected IndexOutOfBounds, got {:?}", e),
        Ok(_) => panic!("expected error for out-of-bounds partition"),
    }
}

// ══════════════════════════════════════════════════════════════
// 10. GROUPBY SCAN
// ══════════════════════════════════════════════════════════════

#[test]
fn groupby_scan_cumsum() {
    let keys = Table::new(vec![Column::from_slice(&[1i32, 1, 2, 2]).unwrap()]).unwrap();
    let values = Table::new(vec![
        Column::from_slice(&[10.0f64, 20.0, 30.0, 40.0]).unwrap(),
    ])
    .unwrap();

    let result = groupby::GroupByScan::new(&keys)
        .scan(0, groupby::GroupByScanOp::Sum)
        .execute(&values)
        .unwrap();

    assert!(result.num_columns() >= 2); // keys + scan result
    assert_eq!(result.num_rows(), 4);
}

#[test]
fn groupby_scan_cummin() {
    let keys = Table::new(vec![Column::from_slice(&[1i32, 1, 1]).unwrap()]).unwrap();
    let values = Table::new(vec![Column::from_slice(&[30.0f64, 10.0, 20.0]).unwrap()]).unwrap();

    let result = groupby::GroupByScan::new(&keys)
        .scan(0, groupby::GroupByScanOp::Min)
        .execute(&values)
        .unwrap();
    assert_eq!(result.num_rows(), 3);
}

#[test]
fn groupby_scan_cummax() {
    let keys = Table::new(vec![Column::from_slice(&[1i32, 1, 1]).unwrap()]).unwrap();
    let values = Table::new(vec![Column::from_slice(&[10.0f64, 30.0, 20.0]).unwrap()]).unwrap();

    let result = groupby::GroupByScan::new(&keys)
        .scan(0, groupby::GroupByScanOp::Max)
        .execute(&values)
        .unwrap();
    assert_eq!(result.num_rows(), 3);
}

#[test]
fn groupby_scan_empty_request_rejected() {
    let keys = Table::new(vec![Column::from_slice(&[1i32, 2]).unwrap()]).unwrap();
    let values = Table::new(vec![Column::from_slice(&[10.0f64, 20.0]).unwrap()]).unwrap();

    let err = groupby::GroupByScan::new(&keys)
        .execute(&values)
        .unwrap_err();
    match err {
        CudfError::InvalidArgument(_) => {}
        _ => panic!("expected InvalidArgument for empty scan request"),
    }
}

// ══════════════════════════════════════════════════════════════
// 11. GROUPBY GET_GROUPS / REPLACE_NULLS
// ══════════════════════════════════════════════════════════════

#[test]
fn groupby_get_groups() {
    let keys = Table::new(vec![Column::from_slice(&[2i32, 1, 2, 1, 3]).unwrap()]).unwrap();

    let groups = keys.groupby_get_groups().unwrap();
    assert!(groups.keys.num_rows() > 0);
    assert!(groups.offsets.len() > 0);
    assert!(groups.values.is_none());
}

#[test]
fn groupby_get_groups_with_values() {
    let keys = Table::new(vec![Column::from_slice(&[1i32, 1, 2, 2]).unwrap()]).unwrap();
    let values = Table::new(vec![
        Column::from_slice(&[10.0f64, 20.0, 30.0, 40.0]).unwrap(),
    ])
    .unwrap();

    let groups = keys.groupby_get_groups_with_values(&values).unwrap();
    assert!(groups.values.is_some());
    assert_eq!(groups.values.unwrap().num_rows(), 4);
}

#[test]
fn groupby_replace_nulls_forward() {
    let keys = Table::new(vec![Column::from_slice(&[1i32, 1, 1, 2, 2]).unwrap()]).unwrap();
    let values = Table::new(vec![
        Column::from_optional_f64(&[Some(10.0), None, Some(30.0), None, Some(50.0)]).unwrap(),
    ])
    .unwrap();

    let result = keys
        .groupby_replace_nulls(&values, &[groupby::GroupByReplacePolicy::Forward])
        .unwrap();
    assert_eq!(result.num_rows(), 5);
}

// ══════════════════════════════════════════════════════════════
// 12. CHUNKED PARQUET I/O
// ══════════════════════════════════════════════════════════════

#[test]
fn chunked_parquet_writer_and_reader() {
    let path = "/tmp/cudf_test_chunked.parquet";

    // Write 3 chunks
    let mut writer = io::parquet::ChunkedParquetWriter::new(path).unwrap();
    for i in 0..3 {
        let col = Column::from_slice(&[i * 10 + 1, i * 10 + 2, i * 10 + 3]).unwrap();
        let chunk = Table::new(vec![col]).unwrap();
        writer.write(&chunk).unwrap();
    }
    writer.close().unwrap();

    // Read back as single table
    let table = io::parquet::read_parquet(path).unwrap();
    assert_eq!(table.num_rows(), 9);
    assert_eq!(table.num_columns(), 1);
}

#[test]
fn chunked_parquet_reader_iteration() {
    let path = "/tmp/cudf_test_chunked_read.parquet";

    // Write a table
    let col = Column::from_slice(&[1i32, 2, 3, 4, 5, 6, 7, 8, 9, 10]).unwrap();
    let table = Table::new(vec![col]).unwrap();
    io::parquet::write_parquet(&table, path).unwrap();

    // Read back in chunks
    let reader = io::parquet::ChunkedParquetReader::new(path, 100).unwrap();
    let mut total_rows = 0;
    while reader.has_next().unwrap() {
        let chunk = reader.read_chunk().unwrap();
        total_rows += chunk.num_rows();
    }
    assert_eq!(total_rows, 10);
}

#[test]
fn chunked_parquet_writer_with_compression() {
    let path = "/tmp/cudf_test_chunked_zstd.parquet";
    let mut writer =
        io::parquet::ChunkedParquetWriter::with_compression(path, io::parquet::Compression::Zstd)
            .unwrap();
    let col = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let chunk = Table::new(vec![col]).unwrap();
    writer.write(&chunk).unwrap();
    writer.close().unwrap();

    let table = io::parquet::read_parquet(path).unwrap();
    assert_eq!(table.num_rows(), 3);
}

// ══════════════════════════════════════════════════════════════
// 13. PARQUET METADATA
// ══════════════════════════════════════════════════════════════

#[test]
fn parquet_metadata_read() {
    let path = "/tmp/cudf_test_meta.parquet";
    let a = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let b = Column::from_slice(&[4.0f64, 5.0, 6.0]).unwrap();
    let table = Table::new(vec![a, b]).unwrap();
    io::parquet::write_parquet(&table, path).unwrap();

    let meta = io::parquet::read_parquet_metadata(path).unwrap();
    assert_eq!(meta.num_rows(), 3);
    assert_eq!(meta.num_columns(), 2);
    assert!(meta.num_row_groups() >= 1);
    assert_eq!(meta.column_names().len(), 2);
}

// ══════════════════════════════════════════════════════════════
// 14. STRING OPERATIONS
// ══════════════════════════════════════════════════════════════

#[test]
fn str_case_conversion() {
    let col = Column::from_strings(&["Hello", "WORLD", "rust"]).unwrap();

    let upper = col.str_to_upper().unwrap();
    assert_eq!(upper.len(), 3);

    let lower = col.str_to_lower().unwrap();
    assert_eq!(lower.len(), 3);

    let swap = col.str_swapcase().unwrap();
    assert_eq!(swap.len(), 3);
}

#[test]
fn str_title_and_capitalize() {
    let col = Column::from_strings(&["hello world", "HELLO WORLD"]).unwrap();
    let titled = col.str_title().unwrap();
    assert_eq!(titled.len(), 2);

    let cap = col.str_capitalize("").unwrap();
    assert_eq!(cap.len(), 2);
}

#[test]
fn str_is_title() {
    let col = Column::from_strings(&["Hello World", "hello world"]).unwrap();
    let result = col.str_is_title().unwrap();
    assert_eq!(result.len(), 2);
    assert_eq!(result.data_type().id(), TypeId::Bool8);
}

#[test]
fn str_padding() {
    let col = Column::from_strings(&["a", "bb", "ccc"]).unwrap();
    let padded = col
        .str_pad(5, strings::padding::PadSide::Left, " ")
        .unwrap();
    assert_eq!(padded.len(), 3);
}

#[test]
fn str_zfill() {
    let col = Column::from_strings(&["1", "22", "333"]).unwrap();
    let filled = col.str_zfill(5).unwrap();
    assert_eq!(filled.len(), 3);
}

#[test]
fn str_repeat_scalar() {
    let col = Column::from_strings(&["ab", "cd"]).unwrap();
    let repeated = col.str_repeat(3).unwrap();
    assert_eq!(repeated.len(), 2);
}

#[test]
fn str_repeat_per_row() {
    let col = Column::from_strings(&["a", "b", "c"]).unwrap();
    let counts = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let result = col.str_repeat_per_row(&counts).unwrap();
    assert_eq!(result.len(), 3);
}

#[test]
fn str_contains_literal() {
    let col = Column::from_strings(&["hello world", "foo bar", "hello"]).unwrap();
    let result = col.str_contains("hello").unwrap();
    assert_eq!(result.len(), 3);
    assert_eq!(result.data_type().id(), TypeId::Bool8);
}

#[test]
fn str_contains_regex() {
    let col = Column::from_strings(&["abc123", "def456", "ghi"]).unwrap();
    let result = col.str_contains_re("\\d+").unwrap();
    assert_eq!(result.len(), 3);
}

#[test]
fn str_matches_regex() {
    let col = Column::from_strings(&["123", "abc", "456"]).unwrap();
    let result = col.str_matches_re("^\\d+$").unwrap();
    assert_eq!(result.len(), 3);
}

#[test]
fn str_count_regex() {
    let col = Column::from_strings(&["aaa", "aba", "bbb"]).unwrap();
    let result = col.str_count_re("a").unwrap();
    assert_eq!(result.len(), 3);
    assert_eq!(result.data_type().id(), TypeId::Int32);
}

#[test]
fn str_findall() {
    let col = Column::from_strings(&["abc123def456", "no_match"]).unwrap();
    let result = col.str_findall("\\d+").unwrap();
    assert_eq!(result.len(), 2);
    assert_eq!(result.data_type().id(), TypeId::List);
}

#[test]
fn str_find_re() {
    let col = Column::from_strings(&["abc123", "nomatch"]).unwrap();
    let result = col.str_find_re("\\d+").unwrap();
    assert_eq!(result.len(), 2);
    assert_eq!(result.data_type().id(), TypeId::Int32);
}

#[test]
fn str_replace_literal() {
    let col = Column::from_strings(&["hello world", "hello rust"]).unwrap();
    let result = col.str_replace("hello", "hi").unwrap();
    assert_eq!(result.len(), 2);
}

#[test]
fn str_replace_regex() {
    let col = Column::from_strings(&["abc123", "def456"]).unwrap();
    let result = col.str_replace_re("\\d+", "NUM").unwrap();
    assert_eq!(result.len(), 2);
}

#[test]
fn str_replace_slice() {
    let col = Column::from_strings(&["abcdef", "ghijkl"]).unwrap();
    let result = col.str_replace_slice("XX", 1, 3).unwrap();
    assert_eq!(result.len(), 2);
}

#[test]
fn str_split_basic() {
    let col = Column::from_strings(&["a,b,c", "d,e"]).unwrap();
    let table = col.str_split(",", -1).unwrap();
    assert_eq!(table.num_rows(), 2);
    assert!(table.num_columns() >= 2);
}

#[test]
fn str_split_record() {
    let col = Column::from_strings(&["a,b,c", "d,e"]).unwrap();
    let result = col.str_split_record(",", -1).unwrap();
    assert_eq!(result.len(), 2);
    assert_eq!(result.data_type().id(), TypeId::List);
}

#[test]
fn str_split_part() {
    let col = Column::from_strings(&["a-b-c", "x-y-z"]).unwrap();
    let result = col.str_split_part("-", 1).unwrap();
    assert_eq!(result.len(), 2);
}

#[test]
fn str_slice_basic() {
    let col = Column::from_strings(&["abcdef", "ghijkl"]).unwrap();
    let result = col.str_slice(1, 4).unwrap();
    assert_eq!(result.len(), 2);
}

#[test]
fn str_join() {
    let col = Column::from_strings(&["a", "b", "c"]).unwrap();
    let result = col.str_join(",").unwrap();
    assert_eq!(result.len(), 1);
}

#[test]
fn str_extract_groups() {
    let col = Column::from_strings(&["abc123", "def456"]).unwrap();
    let table = col.str_extract("([a-z]+)(\\d+)").unwrap();
    assert_eq!(table.num_columns(), 2);
    assert_eq!(table.num_rows(), 2);
}

#[test]
fn str_extract_single() {
    let col = Column::from_strings(&["abc123", "def456"]).unwrap();
    let result = col.str_extract_single("([a-z]+)(\\d+)", 0).unwrap();
    assert_eq!(result.len(), 2);
}

#[test]
fn str_count_characters() {
    let col = Column::from_strings(&["hello", "hi", ""]).unwrap();
    let result = col.str_count_characters().unwrap();
    assert_eq!(result.len(), 3);
    assert_eq!(result.data_type().id(), TypeId::Int32);
}

#[test]
fn str_count_bytes() {
    let col = Column::from_strings(&["hello", "hi"]).unwrap();
    let result = col.str_count_bytes().unwrap();
    assert_eq!(result.len(), 2);
}

// ══════════════════════════════════════════════════════════════
// 15. CROSS JOIN / SEMI / ANTI JOINS
// ══════════════════════════════════════════════════════════════

#[test]
fn cross_join() {
    let left = Table::new(vec![Column::from_slice(&[1i32, 2]).unwrap()]).unwrap();
    let right = Table::new(vec![Column::from_slice(&[10i32, 20, 30]).unwrap()]).unwrap();
    let result = left.cross_join(&right).unwrap();
    assert_eq!(result.num_rows(), 6); // 2 * 3
    assert_eq!(result.num_columns(), 2);
}

#[test]
fn left_semi_join() {
    let left = Table::new(vec![Column::from_slice(&[1i32, 2, 3, 4, 5]).unwrap()]).unwrap();
    let right = Table::new(vec![Column::from_slice(&[2i32, 4]).unwrap()]).unwrap();
    let result = left.left_semi_join(&right).unwrap();
    assert_eq!(result.left_indices.len(), 2);
}

#[test]
fn left_anti_join() {
    let left = Table::new(vec![Column::from_slice(&[1i32, 2, 3, 4, 5]).unwrap()]).unwrap();
    let right = Table::new(vec![Column::from_slice(&[2i32, 4]).unwrap()]).unwrap();
    let result = left.left_anti_join(&right).unwrap();
    assert_eq!(result.left_indices.len(), 3); // 1, 3, 5
}

// ══════════════════════════════════════════════════════════════
// 16. FULL OUTER JOIN
// ══════════════════════════════════════════════════════════════

#[test]
fn full_join() {
    let left = Table::new(vec![Column::from_slice(&[1i32, 2]).unwrap()]).unwrap();
    let right = Table::new(vec![Column::from_slice(&[2i32, 3]).unwrap()]).unwrap();
    let result = left.full_join(&right).unwrap();
    // 1(L), 2(match), 3(R) => 3 rows
    assert_eq!(result.left_indices.len(), 3);
}

// ══════════════════════════════════════════════════════════════
// 17. ALL OPTIONAL TYPES ROUNDTRIP
// ══════════════════════════════════════════════════════════════

#[test]
fn from_optional_i8_roundtrip() {
    let data = vec![Some(1i8), None, Some(-1)];
    let col = Column::from_optional_i8(&data).unwrap();
    let result: Vec<Option<i8>> = col.to_optional_vec().unwrap();
    assert_eq!(result, data);
}

#[test]
fn from_optional_i16_roundtrip() {
    let data = vec![Some(100i16), None, Some(-100)];
    let col = Column::from_optional_i16(&data).unwrap();
    let result: Vec<Option<i16>> = col.to_optional_vec().unwrap();
    assert_eq!(result, data);
}

#[test]
fn from_optional_u8_roundtrip() {
    let data = vec![Some(255u8), None, Some(0)];
    let col = Column::from_optional_u8(&data).unwrap();
    let result: Vec<Option<u8>> = col.to_optional_vec().unwrap();
    assert_eq!(result, data);
}

#[test]
fn from_optional_u16_roundtrip() {
    let data = vec![Some(1000u16), None, Some(0)];
    let col = Column::from_optional_u16(&data).unwrap();
    let result: Vec<Option<u16>> = col.to_optional_vec().unwrap();
    assert_eq!(result, data);
}

#[test]
fn from_optional_u32_roundtrip() {
    let data = vec![Some(u32::MAX), None, Some(0)];
    let col = Column::from_optional_u32(&data).unwrap();
    let result: Vec<Option<u32>> = col.to_optional_vec().unwrap();
    assert_eq!(result, data);
}

#[test]
fn from_optional_u64_roundtrip() {
    let data = vec![Some(u64::MAX), None, Some(0)];
    let col = Column::from_optional_u64(&data).unwrap();
    let result: Vec<Option<u64>> = col.to_optional_vec().unwrap();
    assert_eq!(result, data);
}

#[test]
fn from_optional_i64_roundtrip() {
    let data = vec![Some(i64::MIN), None, Some(i64::MAX)];
    let col = Column::from_optional_i64(&data).unwrap();
    let result: Vec<Option<i64>> = col.to_optional_vec().unwrap();
    assert_eq!(result, data);
}

#[test]
fn from_optional_f32_roundtrip() {
    let data = vec![Some(1.5f32), None, Some(-2.5)];
    let col = Column::from_optional_f32(&data).unwrap();
    let result: Vec<Option<f32>> = col.to_optional_vec().unwrap();
    assert_eq!(result, data);
}

// ══════════════════════════════════════════════════════════════
// 18. SORTING EDGE CASES
// ══════════════════════════════════════════════════════════════

#[test]
fn stable_sort() {
    let col = Column::from_slice(&[3i32, 1, 2, 1]).unwrap();
    let table = Table::new(vec![col]).unwrap();
    let sorted = table
        .stable_sort(&[SortOrder::Ascending], &[NullOrder::After])
        .unwrap();
    let result: Vec<i32> = sorted.column(0).unwrap().to_vec().unwrap();
    assert_eq!(result, vec![1, 1, 2, 3]);
}

#[test]
fn top_k() {
    let col = Column::from_slice(&[5i32, 3, 8, 1, 9, 2]).unwrap();
    let top = col.top_k(3, SortOrder::Descending).unwrap();
    assert_eq!(top.len(), 3);
}

#[test]
fn rank_column() {
    let col = Column::from_slice(&[30i32, 10, 20]).unwrap();
    let ranks = col
        .rank(
            sorting::RankMethod::First,
            SortOrder::Ascending,
            NullOrder::After,
            false,
        )
        .unwrap();
    assert_eq!(ranks.len(), 3);
    assert_eq!(ranks.data_type().id(), TypeId::Int32); // First method returns size_type (i32), not f64
}

#[test]
fn sort_order_length_mismatch_error() {
    let col = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let table = Table::new(vec![col]).unwrap();
    let err = table
        .sort(
            &[SortOrder::Ascending, SortOrder::Descending],
            &[NullOrder::After],
        )
        .unwrap_err();
    match err {
        CudfError::InvalidArgument(_) => {}
        _ => panic!("expected InvalidArgument for sort order length mismatch"),
    }
}

// ══════════════════════════════════════════════════════════════
// 19. GROUPBY ADVANCED
// NOTE: GroupBy::agg() tests are SKIPPED due to linker bug:
// agg_any() and agg_all() reference undefined symbols
// (make_any_aggregation<groupby_aggregation> and
//  make_all_aggregation<groupby_aggregation>).
// Any use of Aggregation::new() (via GroupBy::agg) pulls in ALL
// match arms including the broken Any/All variants.
// This is release blocker D6 -- see findings.md.
// ══════════════════════════════════════════════════════════════

// ══════════════════════════════════════════════════════════════
// 20. ARROW IPC ROUNDTRIP (column + table)
// ══════════════════════════════════════════════════════════════

#[test]
fn arrow_ipc_column_roundtrip() {
    let col = Column::from_slice(&[1i32, 2, 3, 4]).unwrap();
    let ipc = col.to_arrow_ipc().unwrap();
    assert!(!ipc.is_empty());
    let restored = Column::from_arrow_ipc(&ipc).unwrap();
    assert_eq!(restored.len(), 4);
    let data: Vec<i32> = restored.to_vec().unwrap();
    assert_eq!(data, vec![1, 2, 3, 4]);
}

#[test]
fn arrow_ipc_table_roundtrip() {
    let a = Column::from_slice(&[10i32, 20]).unwrap();
    let b = Column::from_slice(&[1.5f64, 2.5]).unwrap();
    let table = Table::new(vec![a, b]).unwrap();
    let ipc = table.to_arrow_ipc().unwrap();
    let restored = Table::from_arrow_ipc(&ipc).unwrap();
    assert_eq!(restored.num_columns(), 2);
    assert_eq!(restored.num_rows(), 2);
}

// ══════════════════════════════════════════════════════════════
// 21. PARQUET READER/WRITER WITH OPTIONS
// ══════════════════════════════════════════════════════════════

#[test]
fn parquet_reader_with_metadata() {
    let path = "/tmp/cudf_test_parquet_meta.parquet";
    let col = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let table = Table::new(vec![col]).unwrap();
    io::parquet::write_parquet(&table, path).unwrap();

    let twm = io::parquet::ParquetReader::new(path)
        .read_with_metadata()
        .unwrap();
    assert_eq!(twm.table.num_rows(), 3);
    assert_eq!(twm.column_names.len(), 1);
}

#[test]
fn parquet_reader_skip_rows() {
    let path = "/tmp/cudf_test_skip.parquet";
    let col = Column::from_slice(&[1i32, 2, 3, 4, 5]).unwrap();
    let table = Table::new(vec![col]).unwrap();
    io::parquet::write_parquet(&table, path).unwrap();

    let result = io::parquet::ParquetReader::new(path)
        .skip_rows(2)
        .read()
        .unwrap();
    assert_eq!(result.num_rows(), 3);
}

#[test]
fn parquet_reader_num_rows() {
    let path = "/tmp/cudf_test_numrows.parquet";
    let col = Column::from_slice(&[1i32, 2, 3, 4, 5]).unwrap();
    let table = Table::new(vec![col]).unwrap();
    io::parquet::write_parquet(&table, path).unwrap();

    let result = io::parquet::ParquetReader::new(path)
        .num_rows(2)
        .read()
        .unwrap();
    assert_eq!(result.num_rows(), 2);
}

// ══════════════════════════════════════════════════════════════
// 22. LARGE COLUMN (stress test)
// ══════════════════════════════════════════════════════════════

#[test]
fn large_column_roundtrip() {
    let n = 1_000_000;
    let data: Vec<i32> = (0..n).collect();
    let col = Column::from_slice(&data).unwrap();
    assert_eq!(col.len(), n as usize);
    let result: Vec<i32> = col.to_vec().unwrap();
    assert_eq!(result.len(), n as usize);
    assert_eq!(result[0], 0);
    assert_eq!(result[n as usize - 1], n - 1);
}
