//! Pure Rust unit tests for cudf types, enums, and builder patterns.
//!
//! These tests exercise logic that does NOT require a GPU or libcudf linkage
//! at runtime -- type conversions, enum discriminants, builder construction,
//! Display impls, and classification helpers.

// ── 1. TypeId tests ──────────────────────────────────────────────

#[cfg(test)]
mod type_id_tests {
    use cudf::TypeId;

    #[test]
    fn from_raw_valid() {
        assert_eq!(TypeId::from_raw(0), Some(TypeId::Empty));
        assert_eq!(TypeId::from_raw(1), Some(TypeId::Int8));
        assert_eq!(TypeId::from_raw(2), Some(TypeId::Int16));
        assert_eq!(TypeId::from_raw(3), Some(TypeId::Int32));
        assert_eq!(TypeId::from_raw(4), Some(TypeId::Int64));
        assert_eq!(TypeId::from_raw(5), Some(TypeId::Uint8));
        assert_eq!(TypeId::from_raw(6), Some(TypeId::Uint16));
        assert_eq!(TypeId::from_raw(7), Some(TypeId::Uint32));
        assert_eq!(TypeId::from_raw(8), Some(TypeId::Uint64));
        assert_eq!(TypeId::from_raw(9), Some(TypeId::Float32));
        assert_eq!(TypeId::from_raw(10), Some(TypeId::Float64));
        assert_eq!(TypeId::from_raw(11), Some(TypeId::Bool8));
        assert_eq!(TypeId::from_raw(23), Some(TypeId::String));
        assert_eq!(TypeId::from_raw(24), Some(TypeId::List));
        assert_eq!(TypeId::from_raw(28), Some(TypeId::Struct));
    }

    #[test]
    fn from_raw_invalid() {
        assert_eq!(TypeId::from_raw(-1), None);
        assert_eq!(TypeId::from_raw(-100), None);
        assert_eq!(TypeId::from_raw(29), None);
        assert_eq!(TypeId::from_raw(100), None);
        assert_eq!(TypeId::from_raw(i32::MAX), None);
        assert_eq!(TypeId::from_raw(i32::MIN), None);
    }

    #[test]
    fn all_variants_from_raw() {
        for i in 0..=28 {
            assert!(
                TypeId::from_raw(i).is_some(),
                "from_raw({}) should be Some",
                i
            );
        }
    }

    #[test]
    fn size_in_bytes_1() {
        assert_eq!(TypeId::Int8.size_in_bytes(), 1);
        assert_eq!(TypeId::Uint8.size_in_bytes(), 1);
        assert_eq!(TypeId::Bool8.size_in_bytes(), 1);
    }

    #[test]
    fn size_in_bytes_2() {
        assert_eq!(TypeId::Int16.size_in_bytes(), 2);
        assert_eq!(TypeId::Uint16.size_in_bytes(), 2);
    }

    #[test]
    fn size_in_bytes_4() {
        assert_eq!(TypeId::Int32.size_in_bytes(), 4);
        assert_eq!(TypeId::Uint32.size_in_bytes(), 4);
        assert_eq!(TypeId::Float32.size_in_bytes(), 4);
        assert_eq!(TypeId::Decimal32.size_in_bytes(), 4);
        assert_eq!(TypeId::Dictionary32.size_in_bytes(), 4);
        assert_eq!(TypeId::TimestampDays.size_in_bytes(), 4);
        assert_eq!(TypeId::DurationDays.size_in_bytes(), 4);
    }

    #[test]
    fn size_in_bytes_8() {
        assert_eq!(TypeId::Int64.size_in_bytes(), 8);
        assert_eq!(TypeId::Uint64.size_in_bytes(), 8);
        assert_eq!(TypeId::Float64.size_in_bytes(), 8);
        assert_eq!(TypeId::Decimal64.size_in_bytes(), 8);
        assert_eq!(TypeId::TimestampSeconds.size_in_bytes(), 8);
        assert_eq!(TypeId::TimestampMilliseconds.size_in_bytes(), 8);
        assert_eq!(TypeId::TimestampMicroseconds.size_in_bytes(), 8);
        assert_eq!(TypeId::TimestampNanoseconds.size_in_bytes(), 8);
        assert_eq!(TypeId::DurationSeconds.size_in_bytes(), 8);
        assert_eq!(TypeId::DurationMilliseconds.size_in_bytes(), 8);
        assert_eq!(TypeId::DurationMicroseconds.size_in_bytes(), 8);
        assert_eq!(TypeId::DurationNanoseconds.size_in_bytes(), 8);
    }

    #[test]
    fn size_in_bytes_16() {
        assert_eq!(TypeId::Decimal128.size_in_bytes(), 16);
    }

    #[test]
    fn size_in_bytes_variable_width() {
        assert_eq!(TypeId::Empty.size_in_bytes(), 0);
        assert_eq!(TypeId::String.size_in_bytes(), 0);
        assert_eq!(TypeId::List.size_in_bytes(), 0);
        assert_eq!(TypeId::Struct.size_in_bytes(), 0);
    }

    #[test]
    fn is_numeric() {
        let numeric = [
            TypeId::Int8, TypeId::Int16, TypeId::Int32, TypeId::Int64,
            TypeId::Uint8, TypeId::Uint16, TypeId::Uint32, TypeId::Uint64,
            TypeId::Float32, TypeId::Float64,
        ];
        for t in &numeric {
            assert!(t.is_numeric(), "{:?} should be numeric", t);
        }

        let not_numeric = [
            TypeId::Empty, TypeId::Bool8, TypeId::String, TypeId::List,
            TypeId::Struct, TypeId::TimestampDays, TypeId::Decimal64,
        ];
        for t in &not_numeric {
            assert!(!t.is_numeric(), "{:?} should NOT be numeric", t);
        }
    }

    #[test]
    fn is_integer() {
        let integers = [
            TypeId::Int8, TypeId::Int16, TypeId::Int32, TypeId::Int64,
            TypeId::Uint8, TypeId::Uint16, TypeId::Uint32, TypeId::Uint64,
        ];
        for t in &integers {
            assert!(t.is_integer(), "{:?} should be integer", t);
        }
        assert!(!TypeId::Float32.is_integer());
        assert!(!TypeId::Float64.is_integer());
        assert!(!TypeId::String.is_integer());
    }

    #[test]
    fn is_floating() {
        assert!(TypeId::Float32.is_floating());
        assert!(TypeId::Float64.is_floating());
        assert!(!TypeId::Int32.is_floating());
        assert!(!TypeId::Decimal64.is_floating());
    }

    #[test]
    fn is_temporal() {
        let temporal = [
            TypeId::TimestampDays, TypeId::TimestampSeconds,
            TypeId::TimestampMilliseconds, TypeId::TimestampMicroseconds,
            TypeId::TimestampNanoseconds, TypeId::DurationDays,
            TypeId::DurationSeconds, TypeId::DurationMilliseconds,
            TypeId::DurationMicroseconds, TypeId::DurationNanoseconds,
        ];
        for t in &temporal {
            assert!(t.is_temporal(), "{:?} should be temporal", t);
        }
        assert!(!TypeId::Int32.is_temporal());
        assert!(!TypeId::Float64.is_temporal());
        assert!(!TypeId::String.is_temporal());
    }

    #[test]
    fn is_nested() {
        assert!(TypeId::List.is_nested());
        assert!(TypeId::Struct.is_nested());
        assert!(!TypeId::Int32.is_nested());
        assert!(!TypeId::String.is_nested());
        assert!(!TypeId::Dictionary32.is_nested());
    }

    #[test]
    fn is_fixed_width() {
        assert!(TypeId::Int32.is_fixed_width());
        assert!(TypeId::Float64.is_fixed_width());
        assert!(TypeId::Bool8.is_fixed_width());
        assert!(TypeId::Decimal128.is_fixed_width());
        assert!(TypeId::TimestampNanoseconds.is_fixed_width());
        // Variable-width types
        assert!(!TypeId::Empty.is_fixed_width());
        assert!(!TypeId::String.is_fixed_width());
        assert!(!TypeId::List.is_fixed_width());
        assert!(!TypeId::Struct.is_fixed_width());
    }

    #[test]
    fn display() {
        assert_eq!(format!("{}", TypeId::Int32), "Int32");
        assert_eq!(format!("{}", TypeId::Float64), "Float64");
        assert_eq!(format!("{}", TypeId::String), "String");
        assert_eq!(format!("{}", TypeId::Empty), "Empty");
        assert_eq!(format!("{}", TypeId::Struct), "Struct");
    }

    #[test]
    fn debug() {
        assert_eq!(format!("{:?}", TypeId::Int32), "Int32");
    }

    #[test]
    fn clone_copy_eq_hash() {
        let a = TypeId::Int32;
        let b = a; // Copy
        let c = a.clone(); // Clone
        assert_eq!(a, b);
        assert_eq!(a, c);

        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(TypeId::Int32);
        set.insert(TypeId::Int32);
        assert_eq!(set.len(), 1);
        set.insert(TypeId::Float64);
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn discriminant_values() {
        assert_eq!(TypeId::Empty as i32, 0);
        assert_eq!(TypeId::Int8 as i32, 1);
        assert_eq!(TypeId::Int16 as i32, 2);
        assert_eq!(TypeId::Int32 as i32, 3);
        assert_eq!(TypeId::Int64 as i32, 4);
        assert_eq!(TypeId::Uint8 as i32, 5);
        assert_eq!(TypeId::Uint16 as i32, 6);
        assert_eq!(TypeId::Uint32 as i32, 7);
        assert_eq!(TypeId::Uint64 as i32, 8);
        assert_eq!(TypeId::Float32 as i32, 9);
        assert_eq!(TypeId::Float64 as i32, 10);
        assert_eq!(TypeId::Bool8 as i32, 11);
        assert_eq!(TypeId::TimestampDays as i32, 12);
        assert_eq!(TypeId::String as i32, 23);
        assert_eq!(TypeId::List as i32, 24);
        assert_eq!(TypeId::Decimal32 as i32, 25);
        assert_eq!(TypeId::Decimal64 as i32, 26);
        assert_eq!(TypeId::Decimal128 as i32, 27);
        assert_eq!(TypeId::Struct as i32, 28);
    }
}

// ── 2. DataType tests ────────────────────────────────────────────

#[cfg(test)]
mod data_type_tests {
    use cudf::{DataType, TypeId};

    #[test]
    fn new_basic() {
        let dt = DataType::new(TypeId::Int32);
        assert_eq!(dt.id(), TypeId::Int32);
        assert_eq!(dt.scale(), 0);
    }

    #[test]
    fn new_all_types() {
        for i in 0..=28 {
            let tid = TypeId::from_raw(i).unwrap();
            let dt = DataType::new(tid);
            assert_eq!(dt.id(), tid);
            assert_eq!(dt.scale(), 0);
        }
    }

    #[test]
    fn decimal32() {
        let dt = DataType::decimal(TypeId::Decimal32, -5);
        assert_eq!(dt.id(), TypeId::Decimal32);
        assert_eq!(dt.scale(), -5);
    }

    #[test]
    fn decimal64() {
        let dt = DataType::decimal(TypeId::Decimal64, -3);
        assert_eq!(dt.id(), TypeId::Decimal64);
        assert_eq!(dt.scale(), -3);
    }

    #[test]
    fn decimal128() {
        let dt = DataType::decimal(TypeId::Decimal128, 2);
        assert_eq!(dt.id(), TypeId::Decimal128);
        assert_eq!(dt.scale(), 2);
    }

    #[test]
    fn decimal_zero_scale() {
        let dt = DataType::decimal(TypeId::Decimal64, 0);
        assert_eq!(dt.scale(), 0);
    }

    #[test]
    #[should_panic(expected = "decimal() requires a decimal TypeId")]
    fn decimal_non_decimal_int32() {
        DataType::decimal(TypeId::Int32, 0);
    }

    #[test]
    #[should_panic(expected = "decimal() requires a decimal TypeId")]
    fn decimal_non_decimal_string() {
        DataType::decimal(TypeId::String, 0);
    }

    #[test]
    #[should_panic(expected = "decimal() requires a decimal TypeId")]
    fn decimal_non_decimal_float() {
        DataType::decimal(TypeId::Float64, -2);
    }

    #[test]
    fn from_type_id() {
        let dt: DataType = TypeId::Float64.into();
        assert_eq!(dt.id(), TypeId::Float64);
        assert_eq!(dt.scale(), 0);
    }

    #[test]
    fn from_type_id_via_from() {
        let dt = DataType::from(TypeId::Int8);
        assert_eq!(dt.id(), TypeId::Int8);
    }

    #[test]
    fn display_non_decimal() {
        assert_eq!(format!("{}", DataType::new(TypeId::Int32)), "Int32");
        assert_eq!(format!("{}", DataType::new(TypeId::Float64)), "Float64");
        assert_eq!(format!("{}", DataType::new(TypeId::String)), "String");
    }

    #[test]
    fn display_decimal_with_scale() {
        assert_eq!(
            format!("{}", DataType::decimal(TypeId::Decimal64, -3)),
            "Decimal64(scale=-3)"
        );
        assert_eq!(
            format!("{}", DataType::decimal(TypeId::Decimal128, 5)),
            "Decimal128(scale=5)"
        );
    }

    #[test]
    fn display_decimal_zero_scale() {
        // scale == 0 uses the non-decimal display path
        assert_eq!(
            format!("{}", DataType::decimal(TypeId::Decimal64, 0)),
            "Decimal64"
        );
    }

    #[test]
    fn eq() {
        assert_eq!(DataType::new(TypeId::Int32), DataType::new(TypeId::Int32));
        assert_ne!(DataType::new(TypeId::Int32), DataType::new(TypeId::Int64));
        assert_ne!(DataType::new(TypeId::Int32), DataType::new(TypeId::Float32));
    }

    #[test]
    fn eq_decimal_different_scale() {
        let a = DataType::decimal(TypeId::Decimal64, -3);
        let b = DataType::decimal(TypeId::Decimal64, -5);
        assert_ne!(a, b);
    }

    #[test]
    fn eq_decimal_same() {
        let a = DataType::decimal(TypeId::Decimal64, -3);
        let b = DataType::decimal(TypeId::Decimal64, -3);
        assert_eq!(a, b);
    }

    #[test]
    fn clone_copy() {
        let dt = DataType::new(TypeId::Int32);
        let dt2 = dt; // Copy
        let dt3 = dt.clone(); // Clone
        assert_eq!(dt, dt2);
        assert_eq!(dt, dt3);
    }

    #[test]
    fn hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(DataType::new(TypeId::Int32));
        set.insert(DataType::new(TypeId::Int32));
        assert_eq!(set.len(), 1);
        set.insert(DataType::new(TypeId::Float64));
        assert_eq!(set.len(), 2);
    }
}

// ── 3. CudfType trait tests ──────────────────────────────────────

#[cfg(test)]
mod cudf_type_tests {
    use cudf::{CudfType, TypeId};

    #[test]
    fn i8_type_id() { assert_eq!(i8::TYPE_ID, TypeId::Int8); }
    #[test]
    fn i16_type_id() { assert_eq!(i16::TYPE_ID, TypeId::Int16); }
    #[test]
    fn i32_type_id() { assert_eq!(i32::TYPE_ID, TypeId::Int32); }
    #[test]
    fn i64_type_id() { assert_eq!(i64::TYPE_ID, TypeId::Int64); }
    #[test]
    fn u8_type_id() { assert_eq!(u8::TYPE_ID, TypeId::Uint8); }
    #[test]
    fn u16_type_id() { assert_eq!(u16::TYPE_ID, TypeId::Uint16); }
    #[test]
    fn u32_type_id() { assert_eq!(u32::TYPE_ID, TypeId::Uint32); }
    #[test]
    fn u64_type_id() { assert_eq!(u64::TYPE_ID, TypeId::Uint64); }
    #[test]
    fn f32_type_id() { assert_eq!(f32::TYPE_ID, TypeId::Float32); }
    #[test]
    fn f64_type_id() { assert_eq!(f64::TYPE_ID, TypeId::Float64); }

    #[test]
    fn all_numeric_types_are_numeric() {
        assert!(i8::TYPE_ID.is_numeric());
        assert!(i16::TYPE_ID.is_numeric());
        assert!(i32::TYPE_ID.is_numeric());
        assert!(i64::TYPE_ID.is_numeric());
        assert!(u8::TYPE_ID.is_numeric());
        assert!(u16::TYPE_ID.is_numeric());
        assert!(u32::TYPE_ID.is_numeric());
        assert!(u64::TYPE_ID.is_numeric());
        assert!(f32::TYPE_ID.is_numeric());
        assert!(f64::TYPE_ID.is_numeric());
    }

    #[test]
    fn integer_types_are_integer() {
        assert!(i8::TYPE_ID.is_integer());
        assert!(i16::TYPE_ID.is_integer());
        assert!(i32::TYPE_ID.is_integer());
        assert!(i64::TYPE_ID.is_integer());
        assert!(u8::TYPE_ID.is_integer());
        assert!(u16::TYPE_ID.is_integer());
        assert!(u32::TYPE_ID.is_integer());
        assert!(u64::TYPE_ID.is_integer());
        assert!(!f32::TYPE_ID.is_integer());
        assert!(!f64::TYPE_ID.is_integer());
    }

    #[test]
    fn float_types_are_floating() {
        assert!(f32::TYPE_ID.is_floating());
        assert!(f64::TYPE_ID.is_floating());
        assert!(!i32::TYPE_ID.is_floating());
    }

    #[test]
    fn all_cudf_types_are_fixed_width() {
        assert!(i8::TYPE_ID.is_fixed_width());
        assert!(i16::TYPE_ID.is_fixed_width());
        assert!(i32::TYPE_ID.is_fixed_width());
        assert!(i64::TYPE_ID.is_fixed_width());
        assert!(u8::TYPE_ID.is_fixed_width());
        assert!(u16::TYPE_ID.is_fixed_width());
        assert!(u32::TYPE_ID.is_fixed_width());
        assert!(u64::TYPE_ID.is_fixed_width());
        assert!(f32::TYPE_ID.is_fixed_width());
        assert!(f64::TYPE_ID.is_fixed_width());
    }
}

// ── 4. Enum discriminant tests ───────────────────────────────────

#[cfg(test)]
mod enum_tests {
    use cudf::sorting::{SortOrder, NullOrder, RankMethod, NullHandling};
    use cudf::unary::UnaryOp;
    use cudf::binaryop::BinaryOp;
    use cudf::stream_compaction::DuplicateKeepOption;
    use cudf::reduction::{ReduceOp, ScanOp};
    use cudf::quantiles::Interpolation;
    use cudf::rolling::RollingAgg;
    use cudf::io::parquet::Compression;

    // -- SortOrder --
    #[test]
    fn sort_order_ascending() { assert_eq!(SortOrder::Ascending as i32, 0); }
    #[test]
    fn sort_order_descending() { assert_eq!(SortOrder::Descending as i32, 1); }

    // -- NullOrder --
    #[test]
    fn null_order_after() { assert_eq!(NullOrder::After as i32, 0); }
    #[test]
    fn null_order_before() { assert_eq!(NullOrder::Before as i32, 1); }

    // -- RankMethod --
    #[test]
    fn rank_method_values() {
        assert_eq!(RankMethod::First as i32, 0);
        assert_eq!(RankMethod::Average as i32, 1);
        assert_eq!(RankMethod::Min as i32, 2);
        assert_eq!(RankMethod::Max as i32, 3);
        assert_eq!(RankMethod::Dense as i32, 4);
    }

    // -- NullHandling (sorting) --
    #[test]
    fn null_handling_sorting_values() {
        assert_eq!(NullHandling::Include as i32, 0);
        assert_eq!(NullHandling::Exclude as i32, 1);
    }

    // -- UnaryOp --
    #[test]
    fn unary_op_first() { assert_eq!(UnaryOp::Sin as i32, 0); }
    #[test]
    fn unary_op_last() { assert_eq!(UnaryOp::Negate as i32, 23); }
    #[test]
    fn unary_op_all_values() {
        assert_eq!(UnaryOp::Sin as i32, 0);
        assert_eq!(UnaryOp::Cos as i32, 1);
        assert_eq!(UnaryOp::Tan as i32, 2);
        assert_eq!(UnaryOp::Arcsin as i32, 3);
        assert_eq!(UnaryOp::Arccos as i32, 4);
        assert_eq!(UnaryOp::Arctan as i32, 5);
        assert_eq!(UnaryOp::Sinh as i32, 6);
        assert_eq!(UnaryOp::Cosh as i32, 7);
        assert_eq!(UnaryOp::Tanh as i32, 8);
        assert_eq!(UnaryOp::Arcsinh as i32, 9);
        assert_eq!(UnaryOp::Arccosh as i32, 10);
        assert_eq!(UnaryOp::Arctanh as i32, 11);
        assert_eq!(UnaryOp::Exp as i32, 12);
        assert_eq!(UnaryOp::Log as i32, 13);
        assert_eq!(UnaryOp::Sqrt as i32, 14);
        assert_eq!(UnaryOp::Cbrt as i32, 15);
        assert_eq!(UnaryOp::Ceil as i32, 16);
        assert_eq!(UnaryOp::Floor as i32, 17);
        assert_eq!(UnaryOp::Abs as i32, 18);
        assert_eq!(UnaryOp::Rint as i32, 19);
        assert_eq!(UnaryOp::BitCount as i32, 20);
        assert_eq!(UnaryOp::BitInvert as i32, 21);
        assert_eq!(UnaryOp::Not as i32, 22);
        assert_eq!(UnaryOp::Negate as i32, 23);
    }

    // -- BinaryOp --
    #[test]
    fn binary_op_first() { assert_eq!(BinaryOp::Add as i32, 0); }
    #[test]
    fn binary_op_last() { assert_eq!(BinaryOp::InvalidBinary as i32, 34); }
    #[test]
    fn binary_op_key_values() {
        assert_eq!(BinaryOp::Add as i32, 0);
        assert_eq!(BinaryOp::Sub as i32, 1);
        assert_eq!(BinaryOp::Mul as i32, 2);
        assert_eq!(BinaryOp::Div as i32, 3);
        assert_eq!(BinaryOp::Mod as i32, 6);
        assert_eq!(BinaryOp::Pow as i32, 9);
        assert_eq!(BinaryOp::Equal as i32, 21);
        assert_eq!(BinaryOp::NotEqual as i32, 22);
        assert_eq!(BinaryOp::Less as i32, 23);
        assert_eq!(BinaryOp::Greater as i32, 24);
        assert_eq!(BinaryOp::NullEquals as i32, 27);
        assert_eq!(BinaryOp::NullNotEquals as i32, 28);
        assert_eq!(BinaryOp::GenericBinary as i32, 31);
    }

    // -- DuplicateKeepOption --
    #[test]
    fn duplicate_keep_any() { assert_eq!(DuplicateKeepOption::Any as i32, 0); }
    #[test]
    fn duplicate_keep_first() { assert_eq!(DuplicateKeepOption::First as i32, 1); }
    #[test]
    fn duplicate_keep_last() { assert_eq!(DuplicateKeepOption::Last as i32, 2); }
    #[test]
    fn duplicate_keep_none() { assert_eq!(DuplicateKeepOption::None as i32, 3); }

    // -- ReduceOp --
    #[test]
    fn reduce_op_values() {
        assert_eq!(ReduceOp::Sum as i32, 0);
        assert_eq!(ReduceOp::Product as i32, 1);
        assert_eq!(ReduceOp::Min as i32, 2);
        assert_eq!(ReduceOp::Max as i32, 3);
        assert_eq!(ReduceOp::SumOfSquares as i32, 4);
        assert_eq!(ReduceOp::Mean as i32, 5);
        assert_eq!(ReduceOp::Variance as i32, 6);
        assert_eq!(ReduceOp::Std as i32, 7);
        assert_eq!(ReduceOp::Any as i32, 8);
        assert_eq!(ReduceOp::All as i32, 9);
        assert_eq!(ReduceOp::Median as i32, 10);
    }

    // -- ScanOp --
    #[test]
    fn scan_op_values() {
        assert_eq!(ScanOp::Sum as i32, 0);
        assert_eq!(ScanOp::Product as i32, 1);
        assert_eq!(ScanOp::Min as i32, 2);
        assert_eq!(ScanOp::Max as i32, 3);
    }

    // -- Interpolation --
    #[test]
    fn interpolation_values() {
        assert_eq!(Interpolation::Linear as i32, 0);
        assert_eq!(Interpolation::Lower as i32, 1);
        assert_eq!(Interpolation::Higher as i32, 2);
        assert_eq!(Interpolation::Midpoint as i32, 3);
        assert_eq!(Interpolation::Nearest as i32, 4);
    }

    // -- RollingAgg --
    #[test]
    fn rolling_agg_values() {
        assert_eq!(RollingAgg::Sum as i32, 0);
        assert_eq!(RollingAgg::Min as i32, 1);
        assert_eq!(RollingAgg::Max as i32, 2);
        assert_eq!(RollingAgg::Count as i32, 3);
        assert_eq!(RollingAgg::Mean as i32, 4);
        assert_eq!(RollingAgg::CollectList as i32, 5);
        assert_eq!(RollingAgg::RowNumber as i32, 6);
        assert_eq!(RollingAgg::Lead as i32, 7);
        assert_eq!(RollingAgg::Lag as i32, 8);
    }

    // -- Compression --
    #[test]
    fn compression_values() {
        assert_eq!(Compression::None as i32, 0);
        assert_eq!(Compression::Auto as i32, 1);
        assert_eq!(Compression::Snappy as i32, 2);
        assert_eq!(Compression::Gzip as i32, 3);
        assert_eq!(Compression::Bzip2 as i32, 4);
        assert_eq!(Compression::Brotli as i32, 5);
        assert_eq!(Compression::Zip as i32, 6);
        assert_eq!(Compression::Xz as i32, 7);
        assert_eq!(Compression::Zlib as i32, 8);
        assert_eq!(Compression::Lz4 as i32, 9);
        assert_eq!(Compression::Lzo as i32, 10);
        assert_eq!(Compression::Zstd as i32, 11);
    }
}

// ── 5. IO Builder pattern tests ──────────────────────────────────

#[cfg(test)]
mod io_builder_tests {
    use cudf::io::parquet::{ParquetReader, Compression};
    use cudf::io::csv::CsvReader;
    use cudf::io::json::JsonReader;

    #[test]
    fn parquet_reader_builder_basic() {
        let _reader = ParquetReader::new("/tmp/test.parquet");
    }

    #[test]
    fn parquet_reader_builder_with_columns() {
        let _reader = ParquetReader::new("/tmp/test.parquet")
            .columns(vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn parquet_reader_builder_with_skip_rows() {
        let _reader = ParquetReader::new("/tmp/test.parquet")
            .skip_rows(10);
    }

    #[test]
    fn parquet_reader_builder_with_num_rows() {
        let _reader = ParquetReader::new("/tmp/test.parquet")
            .num_rows(100);
    }

    #[test]
    fn parquet_reader_builder_chained() {
        let _reader = ParquetReader::new("/tmp/test.parquet")
            .columns(vec!["a".to_string(), "b".to_string()])
            .skip_rows(10)
            .num_rows(100);
    }

    #[test]
    fn parquet_reader_from_string() {
        let path = String::from("/tmp/test.parquet");
        let _reader = ParquetReader::new(path);
    }

    #[test]
    fn csv_reader_builder_basic() {
        let _reader = CsvReader::new("/tmp/test.csv");
    }

    #[test]
    fn csv_reader_builder_tab_delimiter() {
        let _reader = CsvReader::new("/tmp/test.csv")
            .delimiter(b'\t');
    }

    #[test]
    fn csv_reader_builder_no_header() {
        let _reader = CsvReader::new("/tmp/test.csv")
            .no_header();
    }

    #[test]
    fn csv_reader_builder_chained() {
        let _reader = CsvReader::new("/tmp/test.csv")
            .delimiter(b'\t')
            .no_header()
            .skip_rows(5)
            .num_rows(50);
    }

    #[test]
    fn json_reader_builder_basic() {
        let _reader = JsonReader::new("/tmp/test.json");
    }

    #[test]
    fn json_reader_builder_lines() {
        let _reader = JsonReader::new("/tmp/test.json")
            .lines(true);
    }

    #[test]
    fn json_reader_builder_no_lines() {
        let _reader = JsonReader::new("/tmp/test.json")
            .lines(false);
    }

    // Verify Compression enum is constructible and Debug-printable
    #[test]
    fn compression_debug() {
        let c = Compression::Snappy;
        let dbg = format!("{:?}", c);
        assert!(dbg.contains("Snappy"));
    }

    #[test]
    fn compression_clone_eq() {
        let a = Compression::Zstd;
        let b = a;
        assert_eq!(a, b);
    }
}

// ── 6. AggregationKind tests ─────────────────────────────────────

#[cfg(test)]
mod aggregation_tests {
    use cudf::AggregationKind;

    #[test]
    fn simple_kinds_constructible() {
        let _ = AggregationKind::Sum;
        let _ = AggregationKind::Product;
        let _ = AggregationKind::Min;
        let _ = AggregationKind::Max;
        let _ = AggregationKind::Count;
        let _ = AggregationKind::Any;
        let _ = AggregationKind::All;
        let _ = AggregationKind::SumOfSquares;
        let _ = AggregationKind::Mean;
        let _ = AggregationKind::Median;
        let _ = AggregationKind::Nunique;
        let _ = AggregationKind::CollectList;
        let _ = AggregationKind::CollectSet;
        let _ = AggregationKind::Argmax;
        let _ = AggregationKind::Argmin;
        let _ = AggregationKind::RowNumber;
    }

    #[test]
    fn parameterized_kinds_constructible() {
        let _ = AggregationKind::Variance { ddof: 1 };
        let _ = AggregationKind::Variance { ddof: 0 };
        let _ = AggregationKind::Std { ddof: 1 };
        let _ = AggregationKind::Std { ddof: 0 };
        let _ = AggregationKind::NthElement { n: 0 };
        let _ = AggregationKind::NthElement { n: 5 };
        let _ = AggregationKind::Quantile { q: 0.0 };
        let _ = AggregationKind::Quantile { q: 0.5 };
        let _ = AggregationKind::Quantile { q: 1.0 };
        let _ = AggregationKind::Lag { offset: 1 };
        let _ = AggregationKind::Lag { offset: 10 };
        let _ = AggregationKind::Lead { offset: 1 };
        let _ = AggregationKind::Lead { offset: 10 };
    }

    #[test]
    fn all_kinds_in_vec() {
        let kinds = vec![
            AggregationKind::Sum,
            AggregationKind::Product,
            AggregationKind::Min,
            AggregationKind::Max,
            AggregationKind::Count,
            AggregationKind::Any,
            AggregationKind::All,
            AggregationKind::SumOfSquares,
            AggregationKind::Mean,
            AggregationKind::Median,
            AggregationKind::Variance { ddof: 1 },
            AggregationKind::Std { ddof: 1 },
            AggregationKind::Nunique,
            AggregationKind::NthElement { n: 0 },
            AggregationKind::CollectList,
            AggregationKind::CollectSet,
            AggregationKind::Argmax,
            AggregationKind::Argmin,
            AggregationKind::RowNumber,
            AggregationKind::Quantile { q: 0.5 },
            AggregationKind::Lag { offset: 1 },
            AggregationKind::Lead { offset: 1 },
        ];
        assert_eq!(kinds.len(), 22);
    }

    #[test]
    fn aggregation_kind_eq() {
        assert_eq!(AggregationKind::Sum, AggregationKind::Sum);
        assert_ne!(AggregationKind::Sum, AggregationKind::Min);
        assert_eq!(
            AggregationKind::Variance { ddof: 1 },
            AggregationKind::Variance { ddof: 1 }
        );
        assert_ne!(
            AggregationKind::Variance { ddof: 0 },
            AggregationKind::Variance { ddof: 1 }
        );
    }

    #[test]
    fn aggregation_kind_debug() {
        let dbg = format!("{:?}", AggregationKind::Sum);
        assert_eq!(dbg, "Sum");
        let dbg = format!("{:?}", AggregationKind::Variance { ddof: 1 });
        assert!(dbg.contains("Variance"));
        assert!(dbg.contains("1"));
    }

    #[test]
    fn aggregation_kind_clone() {
        let a = AggregationKind::Quantile { q: 0.75 };
        let b = a.clone();
        assert_eq!(a, b);
    }
}

// ── 7. Error type tests ─────────────────────────────────────────

#[cfg(test)]
mod error_tests {
    use cudf::CudfError;

    #[test]
    fn invalid_argument_display() {
        let err = CudfError::InvalidArgument("bad input".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("bad input"));
        assert!(msg.contains("invalid argument"));
    }

    #[test]
    fn type_mismatch_display() {
        let err = CudfError::TypeMismatch {
            expected: "Int32".to_string(),
            actual: "Float64".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("Int32"));
        assert!(msg.contains("Float64"));
        assert!(msg.contains("type mismatch"));
    }

    #[test]
    fn index_out_of_bounds_display() {
        let err = CudfError::IndexOutOfBounds { index: 5, size: 3 };
        let msg = format!("{}", err);
        assert!(msg.contains("5"));
        assert!(msg.contains("3"));
        assert!(msg.contains("index out of bounds"));
    }

    #[test]
    fn cxx_error_display() {
        let err = CudfError::Cxx("something broke".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("something broke"));
    }

    #[test]
    fn cuda_error_display() {
        let err = CudfError::Cuda("out of memory".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("out of memory"));
        assert!(msg.contains("CUDA"));
    }

    #[test]
    fn io_error_from() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file missing");
        let cudf_err: CudfError = io_err.into();
        let msg = format!("{}", cudf_err);
        assert!(msg.contains("file missing"));
    }

    #[test]
    fn error_is_debug() {
        let err = CudfError::InvalidArgument("test".to_string());
        let dbg = format!("{:?}", err);
        assert!(dbg.contains("InvalidArgument"));
    }
}

// ── 8. Re-export availability tests ─────────────────────────────

#[cfg(test)]
mod reexport_tests {
    // Verify that key types are accessible from the crate root
    #[test]
    fn root_reexports() {
        let _: cudf::TypeId = cudf::TypeId::Int32;
        let _: cudf::DataType = cudf::DataType::new(cudf::TypeId::Int32);
        let _: cudf::SortOrder = cudf::SortOrder::Ascending;
        let _: cudf::NullOrder = cudf::NullOrder::After;
        let _: cudf::UnaryOp = cudf::UnaryOp::Abs;
        let _: cudf::BinaryOp = cudf::BinaryOp::Add;
        let _: cudf::DuplicateKeepOption = cudf::DuplicateKeepOption::First;
        let _: cudf::ReduceOp = cudf::ReduceOp::Sum;
        let _: cudf::ScanOp = cudf::ScanOp::Sum;
        let _: cudf::Interpolation = cudf::Interpolation::Linear;
        let _: cudf::RollingAgg = cudf::RollingAgg::Sum;
        let _: cudf::AggregationKind = cudf::AggregationKind::Sum;
    }
}
