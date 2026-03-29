//! Bridge definitions for libcudf string combine (concatenation) operations.

#[cxx::bridge(namespace = "cudf_shims")]
pub mod ffi {
    unsafe extern "C++" {
        include!("strings/combine_shim.h");
        include!("column_shim.h");

        type OwnedColumn = crate::column::ffi::OwnedColumn;

        /// Concatenate all strings in the column into a single string,
        /// separated by `separator`. Returns a single-element string column.
        fn str_join(col: &OwnedColumn, separator: &str) -> Result<UniquePtr<OwnedColumn>>;

        /// Join list elements within each row using a scalar separator.
        fn str_join_list_elements(
            col: &OwnedColumn,
            separator: &str,
        ) -> Result<UniquePtr<OwnedColumn>>;
    }
}
