//! Bridge definitions for libcudf string conversion operations.

#[cxx::bridge(namespace = "cudf_shims")]
pub mod ffi {
    unsafe extern "C++" {
        include!("strings/convert_shim.h");
        include!("column_shim.h");

        type OwnedColumn = crate::column::ffi::OwnedColumn;

        /// Convert string column to integer column of the specified type.
        /// `type_id` corresponds to `cudf::type_id` (e.g., INT32, INT64).
        fn str_to_integers(col: &OwnedColumn, type_id: i32) -> Result<UniquePtr<OwnedColumn>>;

        /// Convert integer column to string column.
        fn str_from_integers(col: &OwnedColumn) -> Result<UniquePtr<OwnedColumn>>;

        /// Convert string column to float column of the specified type.
        /// `type_id` corresponds to `cudf::type_id` (e.g., FLOAT32, FLOAT64).
        fn str_to_floats(col: &OwnedColumn, type_id: i32) -> Result<UniquePtr<OwnedColumn>>;

        /// Convert float column to string column.
        fn str_from_floats(col: &OwnedColumn) -> Result<UniquePtr<OwnedColumn>>;

        // ── Booleans ──────────────────────────────────────────────

        /// Convert string column to boolean column using `true_str` as the true value.
        fn str_to_booleans(col: &OwnedColumn, true_str: &str) -> Result<UniquePtr<OwnedColumn>>;

        /// Convert boolean column to string column.
        fn str_from_booleans(
            col: &OwnedColumn,
            true_str: &str,
            false_str: &str,
        ) -> Result<UniquePtr<OwnedColumn>>;

        // ── Timestamps ────────────────────────────────────────────

        /// Convert string column to timestamp column.
        fn str_to_timestamps(
            col: &OwnedColumn,
            format: &str,
            type_id: i32,
        ) -> Result<UniquePtr<OwnedColumn>>;

        /// Convert timestamp column to string column.
        fn str_from_timestamps(col: &OwnedColumn, format: &str) -> Result<UniquePtr<OwnedColumn>>;

        // ── Durations ─────────────────────────────────────────────

        /// Convert string column to duration column.
        fn str_to_durations(
            col: &OwnedColumn,
            format: &str,
            type_id: i32,
        ) -> Result<UniquePtr<OwnedColumn>>;

        /// Convert duration column to string column.
        fn str_from_durations(col: &OwnedColumn, format: &str) -> Result<UniquePtr<OwnedColumn>>;

        // ── Fixed Point ───────────────────────────────────────────

        /// Convert string column to fixed-point (decimal) column.
        fn str_to_fixed_point(
            col: &OwnedColumn,
            type_id: i32,
            scale: i32,
        ) -> Result<UniquePtr<OwnedColumn>>;

        /// Convert fixed-point column to string column.
        fn str_from_fixed_point(col: &OwnedColumn) -> Result<UniquePtr<OwnedColumn>>;

        // ── Type Checks ───────────────────────────────────────────

        /// Check if each string is a valid integer representation.
        fn str_is_integer(col: &OwnedColumn) -> Result<UniquePtr<OwnedColumn>>;

        /// Check if each string is a valid float representation.
        fn str_is_float(col: &OwnedColumn) -> Result<UniquePtr<OwnedColumn>>;

        // ── Hex ───────────────────────────────────────────────────

        /// Convert hex string column to integer column.
        fn str_hex_to_integers(col: &OwnedColumn, type_id: i32) -> Result<UniquePtr<OwnedColumn>>;

        /// Convert integer column to hex string column.
        fn str_integers_to_hex(col: &OwnedColumn) -> Result<UniquePtr<OwnedColumn>>;

        // ── IPv4 ──────────────────────────────────────────────────

        /// Convert IPv4 string column to integer column.
        fn str_ipv4_to_integers(col: &OwnedColumn) -> Result<UniquePtr<OwnedColumn>>;

        /// Convert integer column to IPv4 string column.
        fn str_integers_to_ipv4(col: &OwnedColumn) -> Result<UniquePtr<OwnedColumn>>;

        // ── URL ───────────────────────────────────────────────────

        /// URL-encode each string.
        fn str_url_encode(col: &OwnedColumn) -> Result<UniquePtr<OwnedColumn>>;

        /// URL-decode each string.
        fn str_url_decode(col: &OwnedColumn) -> Result<UniquePtr<OwnedColumn>>;
    }
}
