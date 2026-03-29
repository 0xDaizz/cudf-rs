//! Bridge definitions for libcudf core types.
//!
//! Maps `cudf::type_id` and `cudf::data_type` to Rust-accessible types
//! via i32 type IDs and opaque C++ types.

#[cxx::bridge(namespace = "cudf_shims")]
pub mod ffi {
    unsafe extern "C++" {
        include!("types_shim.h");

        /// Opaque handle to `cudf::data_type`.
        type DataType;

        /// Create a DataType from a type ID (as i32, matching cudf::type_id).
        fn make_data_type(id: i32) -> UniquePtr<DataType>;

        /// Create a DataType with scale (for decimal types).
        fn make_data_type_with_scale(id: i32, scale: i32) -> UniquePtr<DataType>;

        /// Get the type ID from a DataType (as i32).
        fn data_type_id(dt: &DataType) -> i32;

        /// Get the scale from a DataType (meaningful for decimal types).
        fn data_type_scale(dt: &DataType) -> i32;
    }
}
