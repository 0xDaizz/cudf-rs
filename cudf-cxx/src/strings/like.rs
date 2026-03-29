//! Bridge definitions for libcudf SQL LIKE pattern matching.

#[cxx::bridge(namespace = "cudf_shims")]
pub mod ffi {
    unsafe extern "C++" {
        include!("strings/like_shim.h");
        include!("column_shim.h");

        type OwnedColumn = crate::column::ffi::OwnedColumn;

        /// SQL LIKE pattern matching.
        /// `%` matches zero or more characters, `_` matches any single character.
        fn str_like(
            col: &OwnedColumn,
            pattern: &str,
            escape_char: &str,
        ) -> Result<UniquePtr<OwnedColumn>>;
    }
}
