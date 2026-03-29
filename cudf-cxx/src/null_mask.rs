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

        /// Set a range of bits in a column's null mask.
        fn set_null_mask_range(
            col: &OwnedColumn,
            begin_bit: i32,
            end_bit: i32,
            valid: bool,
        ) -> Result<UniquePtr<OwnedColumn>>;

        /// Copy a column's bitmask to host bytes.
        fn copy_bitmask_to_host(col: &OwnedColumn) -> Vec<u8>;

        // ── Bitmask Builder ──────────────────────────────────────

        /// Builder for collecting column views for bitmask operations.
        type BitmaskBuilder;

        fn bitmask_builder_new() -> UniquePtr<BitmaskBuilder>;

        fn add_column(self: Pin<&mut BitmaskBuilder>, col: &OwnedColumn);

        fn num_columns(self: &BitmaskBuilder) -> i32;

        /// Result of a bitmask AND/OR operation.
        type BitmaskResult;

        fn get_mask(self: &BitmaskResult) -> Vec<u8>;
        fn get_null_count(self: &BitmaskResult) -> i32;

        /// Bitwise AND of null masks from multiple columns.
        fn bitmask_and(builder: &BitmaskBuilder) -> Result<UniquePtr<BitmaskResult>>;

        /// Bitwise OR of null masks from multiple columns.
        fn bitmask_or(builder: &BitmaskBuilder) -> Result<UniquePtr<BitmaskResult>>;
    }
}
