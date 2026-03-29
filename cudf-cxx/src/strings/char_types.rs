//! Bridge definitions for libcudf string character type operations.

#[cxx::bridge(namespace = "cudf_shims")]
pub mod ffi {
    unsafe extern "C++" {
        include!("strings/char_types_shim.h");
        include!("column_shim.h");

        type OwnedColumn = crate::column::ffi::OwnedColumn;

        /// Check if all characters of each string are of the given type(s).
        /// `types` and `verify_types` are bitmask values of string_character_types.
        fn str_all_characters_of_type(
            col: &OwnedColumn,
            types: u32,
            verify_types: u32,
        ) -> Result<UniquePtr<OwnedColumn>>;
    }
}
