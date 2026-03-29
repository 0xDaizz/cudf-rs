//! Bridge definitions for libcudf replace operations.
//!
//! Provides GPU-accelerated null/NaN replacement and value clamping.

#[cxx::bridge(namespace = "cudf_shims")]
pub mod ffi {
    unsafe extern "C++" {
        include!("replace_shim.h");
        include!("column_shim.h");
        include!("scalar_shim.h");

        type OwnedColumn = crate::column::ffi::OwnedColumn;
        type OwnedScalar = crate::scalar::ffi::OwnedScalar;

        // ── Replace nulls ─────────────────────────────────────────

        /// Replace null values in `col` with corresponding values from `replacement`.
        fn replace_nulls_column(
            col: &OwnedColumn,
            replacement: &OwnedColumn,
        ) -> Result<UniquePtr<OwnedColumn>>;

        /// Replace null values in `col` with a scalar.
        fn replace_nulls_scalar(
            col: &OwnedColumn,
            replacement: &OwnedScalar,
        ) -> Result<UniquePtr<OwnedColumn>>;

        // ── Replace NaNs ──────────────────────────────────────────

        /// Replace NaN values in `col` with a scalar.
        fn replace_nans_scalar(
            col: &OwnedColumn,
            replacement: &OwnedScalar,
        ) -> Result<UniquePtr<OwnedColumn>>;

        /// Replace NaN values in `col` with corresponding values from `replacement`.
        fn replace_nans_column(
            col: &OwnedColumn,
            replacement: &OwnedColumn,
        ) -> Result<UniquePtr<OwnedColumn>>;

        // ── Clamp ─────────────────────────────────────────────────

        /// Clamp values in `col` to the range [lo, hi].
        /// Values below `lo` become `lo`, values above `hi` become `hi`.
        fn clamp(
            col: &OwnedColumn,
            lo: &OwnedScalar,
            hi: &OwnedScalar,
        ) -> Result<UniquePtr<OwnedColumn>>;

        // ── Normalize ─────────────────────────────────────────────

        /// Normalize -NaN to +NaN and -0.0 to +0.0.
        fn normalize_nans_and_zeros(col: &OwnedColumn) -> Result<UniquePtr<OwnedColumn>>;

        /// Replace nulls using a policy (PRECEDING=0, FOLLOWING=1).
        fn replace_nulls_policy(col: &OwnedColumn, policy: i32) -> Result<UniquePtr<OwnedColumn>>;

        /// Find and replace all occurrences of old_values with new_values.
        fn find_and_replace_all(
            col: &OwnedColumn,
            old_values: &OwnedColumn,
            new_values: &OwnedColumn,
        ) -> Result<UniquePtr<OwnedColumn>>;

        /// Normalize NaN and zeros in-place.
        fn normalize_nans_and_zeros_inplace(col: Pin<&mut OwnedColumn>) -> Result<()>;

        /// Clamp with replacement values.
        fn clamp_with_replace(
            col: &OwnedColumn,
            lo: &OwnedScalar,
            lo_replace: &OwnedScalar,
            hi: &OwnedScalar,
            hi_replace: &OwnedScalar,
        ) -> Result<UniquePtr<OwnedColumn>>;
    }
}
