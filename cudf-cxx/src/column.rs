//! Bridge definitions for libcudf column types.
//!
//! Provides `OwnedColumn` (wrapping `std::unique_ptr<cudf::column>`) and
//! operations for creating, inspecting, and transferring column data
//! between host and device.

#[cxx::bridge(namespace = "cudf_shims")]
pub mod ffi {
    unsafe extern "C++" {
        include!("column_shim.h");

        /// Opaque owning handle wrapping `std::unique_ptr<cudf::column>`.
        /// Dropping this on the Rust side invokes the C++ destructor,
        /// freeing GPU memory.
        type OwnedColumn;

        // ── Accessors ──────────────────────────────────────────────

        /// Number of elements in this column.
        fn size(self: &OwnedColumn) -> i32;

        /// Type ID of the column's element type.
        fn type_id(self: &OwnedColumn) -> i32;

        /// Scale (for decimal types; 0 otherwise).
        fn type_scale(self: &OwnedColumn) -> i32;

        /// Number of null elements. Returns 0 if not nullable.
        fn null_count(self: &OwnedColumn) -> i32;

        /// Whether this column can contain nulls (has a validity bitmask).
        fn is_nullable(self: &OwnedColumn) -> bool;

        /// Whether this column actually contains any null values.
        fn has_nulls(self: &OwnedColumn) -> bool;

        /// Number of child columns (for nested types like LIST, STRUCT).
        fn num_children(self: &OwnedColumn) -> i32;

        // ── Construction ───────────────────────────────────────────

        /// Create a column from a host `i8` slice. Copies data to GPU.
        fn column_from_i8(data: &[i8]) -> Result<UniquePtr<OwnedColumn>>;

        /// Create a column from a host `i16` slice.
        fn column_from_i16(data: &[i16]) -> Result<UniquePtr<OwnedColumn>>;

        /// Create a column from a host `i32` slice.
        fn column_from_i32(data: &[i32]) -> Result<UniquePtr<OwnedColumn>>;

        /// Create a column from a host `i64` slice.
        fn column_from_i64(data: &[i64]) -> Result<UniquePtr<OwnedColumn>>;

        /// Create a column from a host `u8` slice.
        fn column_from_u8(data: &[u8]) -> Result<UniquePtr<OwnedColumn>>;

        /// Create a column from a host `u16` slice.
        fn column_from_u16(data: &[u16]) -> Result<UniquePtr<OwnedColumn>>;

        /// Create a column from a host `u32` slice.
        fn column_from_u32(data: &[u32]) -> Result<UniquePtr<OwnedColumn>>;

        /// Create a column from a host `u64` slice.
        fn column_from_u64(data: &[u64]) -> Result<UniquePtr<OwnedColumn>>;

        /// Create a column from a host `f32` slice.
        fn column_from_f32(data: &[f32]) -> Result<UniquePtr<OwnedColumn>>;

        /// Create a column from a host `f64` slice.
        fn column_from_f64(data: &[f64]) -> Result<UniquePtr<OwnedColumn>>;

        /// Create a column from a host `bool` slice.
        fn column_from_bool(data: &[bool]) -> Result<UniquePtr<OwnedColumn>>;

        /// Create an empty column of the given type and size (all null).
        fn column_empty(type_id: i32, size: i32) -> Result<UniquePtr<OwnedColumn>>;

        /// Create a string column from host string data.
        fn column_from_strings(data: &[String]) -> Result<UniquePtr<OwnedColumn>>;

        /// Create a nullable i32 column from host data and validity mask.
        fn column_from_i32_nullable(
            data: &[i32],
            validity: &[bool],
        ) -> Result<UniquePtr<OwnedColumn>>;

        /// Create a nullable i64 column from host data and validity mask.
        fn column_from_i64_nullable(
            data: &[i64],
            validity: &[bool],
        ) -> Result<UniquePtr<OwnedColumn>>;

        /// Create a nullable f32 column from host data and validity mask.
        fn column_from_f32_nullable(
            data: &[f32],
            validity: &[bool],
        ) -> Result<UniquePtr<OwnedColumn>>;

        /// Create a nullable f64 column from host data and validity mask.
        fn column_from_f64_nullable(
            data: &[f64],
            validity: &[bool],
        ) -> Result<UniquePtr<OwnedColumn>>;

        // ── Data Transfer ──────────────────────────────────────────

        /// Copy column data to host as i32. Panics if type mismatch.
        fn column_to_i32(col: &OwnedColumn, out: &mut [i32]) -> Result<()>;

        /// Copy column data to host as i64.
        fn column_to_i64(col: &OwnedColumn, out: &mut [i64]) -> Result<()>;

        /// Copy column data to host as f32.
        fn column_to_f32(col: &OwnedColumn, out: &mut [f32]) -> Result<()>;

        /// Copy column data to host as f64.
        fn column_to_f64(col: &OwnedColumn, out: &mut [f64]) -> Result<()>;

        /// Copy column data to host as i8.
        fn column_to_i8(col: &OwnedColumn, out: &mut [i8]) -> Result<()>;

        /// Copy column data to host as i16.
        fn column_to_i16(col: &OwnedColumn, out: &mut [i16]) -> Result<()>;

        /// Copy column data to host as u8.
        fn column_to_u8(col: &OwnedColumn, out: &mut [u8]) -> Result<()>;

        /// Copy column data to host as u16.
        fn column_to_u16(col: &OwnedColumn, out: &mut [u16]) -> Result<()>;

        /// Copy column data to host as u32.
        fn column_to_u32(col: &OwnedColumn, out: &mut [u32]) -> Result<()>;

        /// Copy column data to host as u64.
        fn column_to_u64(col: &OwnedColumn, out: &mut [u64]) -> Result<()>;

        /// Copy the null bitmask to host. Each bit indicates validity.
        fn column_null_mask(col: &OwnedColumn, out: &mut [u8]) -> Result<()>;
    }
}
