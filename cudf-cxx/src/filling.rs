//! Bridge definitions for libcudf filling operations.
//!
//! Provides GPU-accelerated fill, repeat, and sequence generation
//! for columns and tables.

#[cxx::bridge(namespace = "cudf_shims")]
pub mod ffi {
    unsafe extern "C++" {
        include!("filling_shim.h");
        include!("column_shim.h");
        include!("table_shim.h");
        include!("scalar_shim.h");

        type OwnedColumn = crate::column::ffi::OwnedColumn;
        type OwnedTable = crate::table::ffi::OwnedTable;
        type OwnedScalar = crate::scalar::ffi::OwnedScalar;

        // ── Fill ───────────────────────────────────────────────────

        /// Fill a column in-place from `begin` to `end` with a scalar value.
        fn fill_in_place(
            destination: Pin<&mut OwnedColumn>,
            begin: i32,
            end: i32,
            value: &OwnedScalar,
        ) -> Result<()>;

        /// Fill a column out-of-place, returning a new column.
        fn fill(
            input: &OwnedColumn,
            begin: i32,
            end: i32,
            value: &OwnedScalar,
        ) -> Result<UniquePtr<OwnedColumn>>;

        // ── Repeat ─────────────────────────────────────────────────

        /// Repeat rows of a table `count` times.
        fn repeat_table(table: &OwnedTable, count: i32) -> Result<UniquePtr<OwnedTable>>;

        /// Repeat rows of a table, each row repeated by the corresponding count.
        fn repeat_table_variable(
            table: &OwnedTable,
            counts: &OwnedColumn,
        ) -> Result<UniquePtr<OwnedTable>>;

        // ── Sequence ───────────────────────────────────────────────

        /// Generate an i32 sequence: init, init+step, init+2*step, ...
        fn sequence_i32(size: i32, init: i32, step: i32) -> Result<UniquePtr<OwnedColumn>>;

        /// Generate an i64 sequence.
        fn sequence_i64(size: i32, init: i64, step: i64) -> Result<UniquePtr<OwnedColumn>>;

        /// Generate an f32 sequence.
        fn sequence_f32(size: i32, init: f32, step: f32) -> Result<UniquePtr<OwnedColumn>>;

        /// Generate an f64 sequence.
        fn sequence_f64(size: i32, init: f64, step: f64) -> Result<UniquePtr<OwnedColumn>>;

        /// Generate a sequence of months starting from a timestamp scalar.
        fn calendrical_month_sequence(
            size: i32,
            init: &OwnedScalar,
            months: i32,
        ) -> Result<UniquePtr<OwnedColumn>>;
    }
}
