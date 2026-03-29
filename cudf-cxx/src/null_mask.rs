//! Bridge definitions for libcudf null mask operations.
//!
//! Provides utilities for creating, inspecting, and manipulating
//! null bitmasks on GPU columns.

#[cxx::bridge(namespace = "cudf_shims")]
pub mod ffi {
    unsafe extern "C++" {
        include!("null_mask_shim.h");
        include!("column_shim.h");

        type OwnedColumn = crate::column::ffi::OwnedColumn;

        // ── Device Buffer ──────────────────────────────────────────

        /// Opaque owning handle for a device buffer (null mask).
        type OwnedDeviceBuffer;

        /// Size of the buffer in bytes.
        fn size_bytes(self: &OwnedDeviceBuffer) -> i32;

        // ── Existing (Phase 1) ─────────────────────────────────────

        /// Count the number of valid (non-null) elements in a column.
        fn valid_count(col: &OwnedColumn) -> i32;

        /// Return a copy of the column with its null mask removed
        /// (all elements marked valid).
        fn set_all_valid(col: &OwnedColumn) -> Result<UniquePtr<OwnedColumn>>;

        // ── New (Phase 2) ──────────────────────────────────────────

        /// Create a null mask device buffer.
        /// `state`: 0=UNALLOCATED, 1=UNINITIALIZED, 2=ALL_VALID, 3=ALL_NULL.
        fn create_null_mask(size: i32, state: i32) -> Result<UniquePtr<OwnedDeviceBuffer>>;

        /// Count null values in a column.
        fn null_count_column(col: &OwnedColumn) -> i32;

        /// Compute the number of bytes needed for a bitmask of given size.
        fn bitmask_allocation_size(number_of_bits: i32) -> i32;

        /// Copy a column's null mask to host.
        fn copy_null_mask_to_host(col: &OwnedColumn, out: &mut [u8]) -> Result<()>;

        /// Create a new column with a null mask set from host-side bytes.
        fn set_null_mask_from_host(
            col: &OwnedColumn,
            mask: &[u8],
            null_count: i32,
        ) -> Result<UniquePtr<OwnedColumn>>;
    }
}
