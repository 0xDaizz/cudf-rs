//! Bridge definitions for libcudf JSONPath operations.

#[cxx::bridge(namespace = "cudf_shims")]
pub mod ffi {
    unsafe extern "C++" {
        include!("json_shim.h");
        include!("column_shim.h");

        type OwnedColumn = crate::column::ffi::OwnedColumn;

        /// Apply a JSONPath query to each string in the column.
        fn get_json_object(
            col: &OwnedColumn,
            json_path: &str,
            allow_single_quotes: bool,
            strip_quotes: bool,
            missing_fields_as_nulls: bool,
        ) -> Result<UniquePtr<OwnedColumn>>;
    }
}
