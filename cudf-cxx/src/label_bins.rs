//! Bridge definitions for libcudf label_bins operation.

#[cxx::bridge(namespace = "cudf_shims")]
pub mod ffi {
    unsafe extern "C++" {
        include!("label_bins_shim.h");
        include!("column_shim.h");

        type OwnedColumn = crate::column::ffi::OwnedColumn;

        /// Label elements based on membership in the specified bins.
        fn label_bins(
            input: &OwnedColumn,
            left_edges: &OwnedColumn,
            left_inclusive: bool,
            right_edges: &OwnedColumn,
            right_inclusive: bool,
        ) -> Result<UniquePtr<OwnedColumn>>;
    }
}
