//! Bridge definitions for libcudf string conversion operations.

#[cxx::bridge(namespace = "cudf_shims")]
pub mod ffi {
    unsafe extern "C++" {
        include!("strings/convert_shim.h");
        include!("column_shim.h");

        type OwnedColumn = crate::column::ffi::OwnedColumn;

        /// Convert string column to integer column of the specified type.
        /// `type_id` corresponds to `cudf::type_id` (e.g., INT32, INT64).
        fn str_to_integers(col: &OwnedColumn, type_id: i32) -> Result<UniquePtr<OwnedColumn>>;

        /// Convert integer column to string column.
        fn str_from_integers(col: &OwnedColumn) -> Result<UniquePtr<OwnedColumn>>;

        /// Convert string column to float column of the specified type.
        /// `type_id` corresponds to `cudf::type_id` (e.g., FLOAT32, FLOAT64).
        fn str_to_floats(col: &OwnedColumn, type_id: i32) -> Result<UniquePtr<OwnedColumn>>;

        /// Convert float column to string column.
        fn str_from_floats(col: &OwnedColumn) -> Result<UniquePtr<OwnedColumn>>;
    }
}
