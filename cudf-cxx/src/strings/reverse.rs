//! Bridge definitions for libcudf string reverse operations.

#[cxx::bridge(namespace = "cudf_shims")]
pub mod ffi {
    unsafe extern "C++" {
        include!("strings/reverse_shim.h");
        include!("column_shim.h");

        type OwnedColumn = crate::column::ffi::OwnedColumn;

        /// Reverse each string character-by-character.
        fn str_reverse(col: &OwnedColumn) -> Result<UniquePtr<OwnedColumn>>;
    }
}
