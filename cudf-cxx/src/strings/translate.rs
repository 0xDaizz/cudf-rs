//! Bridge definitions for libcudf string translate operations.

#[cxx::bridge(namespace = "cudf_shims")]
pub mod ffi {
    unsafe extern "C++" {
        include!("strings/translate_shim.h");
        include!("column_shim.h");

        type OwnedColumn = crate::column::ffi::OwnedColumn;

        /// Translate characters using parallel arrays of source/target code points.
        fn str_translate(
            col: &OwnedColumn,
            src_chars: &[u32],
            dst_chars: &[u32],
        ) -> Result<UniquePtr<OwnedColumn>>;

        /// Filter characters by keeping or removing specified ranges.
        /// `range_pairs` contains consecutive (lo, hi) pairs of code points.
        fn str_filter_characters(
            col: &OwnedColumn,
            range_pairs: &[u32],
            keep: bool,
            replacement: &str,
        ) -> Result<UniquePtr<OwnedColumn>>;
    }
}
