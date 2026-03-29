//! Bridge definitions for libcudf dictionary encoding operations.
//!
//! Provides GPU-accelerated dictionary encoding and decoding of columns.

#[cxx::bridge(namespace = "cudf_shims")]
pub mod ffi {
    unsafe extern "C++" {
        include!("dictionary_shim.h");
        include!("column_shim.h");

        type OwnedColumn = crate::column::ffi::OwnedColumn;

        // ── Encode / Decode ───────────────────────────────────────

        /// Dictionary-encode a column. Returns a DICTIONARY type column
        /// with keys (unique sorted values) and indices.
        fn dictionary_encode(col: &OwnedColumn) -> Result<UniquePtr<OwnedColumn>>;

        /// Decode a dictionary column back to its original representation.
        fn dictionary_decode(col: &OwnedColumn) -> Result<UniquePtr<OwnedColumn>>;
    }
}
