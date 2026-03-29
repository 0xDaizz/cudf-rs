//! Bridge definitions for libcudf rolling window operations.
//!
//! Provides GPU-accelerated fixed-size rolling window aggregation
//! for columns.

#[cxx::bridge(namespace = "cudf_shims")]
pub mod ffi {
    unsafe extern "C++" {
        include!("column_shim.h");
        include!("table_shim.h");
        include!("rolling_shim.h");

        type OwnedColumn = crate::column::ffi::OwnedColumn;
        type OwnedTable = crate::table::ffi::OwnedTable;

        /// Fixed-size rolling window aggregation.
        /// agg_kind: 0=sum, 1=min, 2=max, 3=count, 4=mean,
        ///           5=collect_list, 6=row_number, 7=lead, 8=lag
        fn rolling_window(
            col: &OwnedColumn,
            preceding: i32,
            following: i32,
            min_periods: i32,
            agg_kind: i32,
        ) -> Result<UniquePtr<OwnedColumn>>;

        /// Grouped rolling window aggregation.
        /// The input must be pre-sorted by group_keys.
        fn grouped_rolling_window(
            group_keys: &OwnedTable,
            input: &OwnedColumn,
            preceding: i32,
            following: i32,
            min_periods: i32,
            agg_kind: i32,
        ) -> Result<UniquePtr<OwnedColumn>>;

        /// Variable-size rolling window aggregation.
        /// `preceding_col` and `following_col` specify per-row window sizes.
        fn rolling_window_variable(
            col: &OwnedColumn,
            preceding_col: &OwnedColumn,
            following_col: &OwnedColumn,
            min_periods: i32,
            agg_kind: i32,
        ) -> Result<UniquePtr<OwnedColumn>>;

        /// Check if a rolling aggregation is valid for a given source data type.
        fn is_valid_rolling_aggregation(source_type_id: i32, agg_kind: i32) -> Result<bool>;
    }
}
