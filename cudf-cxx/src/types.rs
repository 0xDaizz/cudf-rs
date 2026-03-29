//! Bridge definitions for libcudf core types.
//!
//! Maps `cudf::type_id` and `cudf::data_type` to Rust-accessible types
//! via cxx shared enums and opaque C++ types.

#[cxx::bridge(namespace = "cudf_shims")]
pub mod ffi {
    /// Mirrors `cudf::type_id` — identifies the element type stored in a column.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    #[repr(i32)]
    enum TypeId {
        Empty = 0,
        Int8 = 1,
        Int16 = 2,
        Int32 = 3,
        Int64 = 4,
        Uint8 = 5,
        Uint16 = 6,
        Uint32 = 7,
        Uint64 = 8,
        Float32 = 9,
        Float64 = 10,
        Bool8 = 11,
        TimestampDays = 12,
        TimestampSeconds = 13,
        TimestampMilliseconds = 14,
        TimestampMicroseconds = 15,
        TimestampNanoseconds = 16,
        DurationDays = 17,
        DurationSeconds = 18,
        DurationMilliseconds = 19,
        DurationMicroseconds = 20,
        DurationNanoseconds = 21,
        Dictionary32 = 22,
        String = 23,
        List = 24,
        Decimal32 = 25,
        Decimal64 = 26,
        Decimal128 = 27,
        Struct = 28,
    }

    unsafe extern "C++" {
        include!("types_shim.h");

        /// Opaque handle to `cudf::data_type`.
        type DataType;

        /// Create a DataType from a type ID.
        fn make_data_type(id: TypeId) -> UniquePtr<DataType>;

        /// Create a DataType with scale (for decimal types).
        fn make_data_type_with_scale(id: TypeId, scale: i32) -> UniquePtr<DataType>;

        /// Get the type ID from a DataType.
        fn data_type_id(dt: &DataType) -> TypeId;

        /// Get the scale from a DataType (meaningful for decimal types).
        fn data_type_scale(dt: &DataType) -> i32;
    }
}
