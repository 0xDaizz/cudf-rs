//! Bridge definitions for libcudf scalar types.
//!
//! Provides `OwnedScalar` (wrapping `std::unique_ptr<cudf::scalar>`) and
//! operations for creating, inspecting, and transferring scalar data
//! between host and device.

#[cxx::bridge(namespace = "cudf_shims")]
pub mod ffi {
    unsafe extern "C++" {
        include!("scalar_shim.h");

        /// Opaque owning handle wrapping `std::unique_ptr<cudf::scalar>`.
        /// Dropping this on the Rust side invokes the C++ destructor,
        /// freeing GPU memory.
        type OwnedScalar;

        // ── Accessors ──────────────────────────────────────────────

        /// Type ID of the scalar's element type.
        fn type_id(self: &OwnedScalar) -> i32;

        /// Whether this scalar holds a valid (non-null) value.
        fn is_valid(self: &OwnedScalar) -> bool;

        // ── Construction ───────────────────────────────────────────

        /// Create a numeric scalar of the given type. Initially invalid (null).
        fn make_numeric_scalar(type_id: i32) -> Result<UniquePtr<OwnedScalar>>;

        // ── Setters ────────────────────────────────────────────────

        /// Set the value of an INT32 scalar.
        fn scalar_set_i32(s: Pin<&mut OwnedScalar>, value: i32) -> Result<()>;

        /// Set the value of an INT64 scalar.
        fn scalar_set_i64(s: Pin<&mut OwnedScalar>, value: i64) -> Result<()>;

        /// Set the value of a FLOAT32 scalar.
        fn scalar_set_f32(s: Pin<&mut OwnedScalar>, value: f32) -> Result<()>;

        /// Set the value of a FLOAT64 scalar.
        fn scalar_set_f64(s: Pin<&mut OwnedScalar>, value: f64) -> Result<()>;

        /// Set the value of an INT8 scalar.
        fn scalar_set_i8(s: Pin<&mut OwnedScalar>, value: i8) -> Result<()>;

        /// Set the value of an INT16 scalar.
        fn scalar_set_i16(s: Pin<&mut OwnedScalar>, value: i16) -> Result<()>;

        /// Set the value of a UINT8 scalar.
        fn scalar_set_u8(s: Pin<&mut OwnedScalar>, value: u8) -> Result<()>;

        /// Set the value of a UINT16 scalar.
        fn scalar_set_u16(s: Pin<&mut OwnedScalar>, value: u16) -> Result<()>;

        /// Set the value of a UINT32 scalar.
        fn scalar_set_u32(s: Pin<&mut OwnedScalar>, value: u32) -> Result<()>;

        /// Set the value of a UINT64 scalar.
        fn scalar_set_u64(s: Pin<&mut OwnedScalar>, value: u64) -> Result<()>;

        /// Set the validity flag of this scalar.
        fn scalar_set_valid(s: Pin<&mut OwnedScalar>, valid: bool) -> Result<()>;

        // ── Getters ────────────────────────────────────────────────

        /// Get the value of an INT32 scalar.
        fn scalar_get_i32(s: &OwnedScalar) -> Result<i32>;

        /// Get the value of an INT64 scalar.
        fn scalar_get_i64(s: &OwnedScalar) -> Result<i64>;

        /// Get the value of a FLOAT32 scalar.
        fn scalar_get_f32(s: &OwnedScalar) -> Result<f32>;

        /// Get the value of a FLOAT64 scalar.
        fn scalar_get_f64(s: &OwnedScalar) -> Result<f64>;

        /// Get the value of an INT8 scalar.
        fn scalar_get_i8(s: &OwnedScalar) -> Result<i8>;

        /// Get the value of an INT16 scalar.
        fn scalar_get_i16(s: &OwnedScalar) -> Result<i16>;

        /// Get the value of a UINT8 scalar.
        fn scalar_get_u8(s: &OwnedScalar) -> Result<u8>;

        /// Get the value of a UINT16 scalar.
        fn scalar_get_u16(s: &OwnedScalar) -> Result<u16>;

        /// Get the value of a UINT32 scalar.
        fn scalar_get_u32(s: &OwnedScalar) -> Result<u32>;

        /// Get the value of a UINT64 scalar.
        fn scalar_get_u64(s: &OwnedScalar) -> Result<u64>;
    }
}
