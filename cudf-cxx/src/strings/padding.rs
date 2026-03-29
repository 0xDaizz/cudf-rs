//! Bridge definitions for libcudf string padding operations.

#[cxx::bridge(namespace = "cudf_shims")]
pub mod ffi {
    unsafe extern "C++" {
        include!("strings/padding_shim.h");
        include!("column_shim.h");

        type OwnedColumn = crate::column::ffi::OwnedColumn;

        /// Pad each string to at least `width` characters.
        /// `side`: 0=LEFT, 1=RIGHT, 2=BOTH.
        fn str_pad(
            col: &OwnedColumn,
            width: i32,
            side: i32,
            fill_char: &str,
        ) -> Result<UniquePtr<OwnedColumn>>;

        /// Pad each string with leading zeros to at least `width` characters.
        fn str_zfill(col: &OwnedColumn, width: i32) -> Result<UniquePtr<OwnedColumn>>;
    }
}
