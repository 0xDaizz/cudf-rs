//! Bridge definitions for libcudf groupby operations.
//!
//! Uses a `GroupByBuilder` opaque C++ type that accumulates aggregation
//! requests and executes them against a values table.

#[cxx::bridge(namespace = "cudf_shims")]
pub mod ffi {
    unsafe extern "C++" {
        include!("groupby_shim.h");
        include!("table_shim.h");
        include!("column_shim.h");
        include!("aggregation_shim.h");

        type OwnedTable = crate::table::ffi::OwnedTable;
        type OwnedColumn = crate::column::ffi::OwnedColumn;
        type OwnedAggregation = crate::aggregation::ffi::OwnedAggregation;

        /// Opaque groupby builder that holds key columns and accumulated requests.
        type GroupByBuilder;

        /// Create a groupby builder with the given key columns.
        fn groupby_new(keys: &OwnedTable) -> UniquePtr<GroupByBuilder>;

        /// Add an aggregation request: aggregate column `col_idx` from the
        /// values table using the given aggregation.
        fn groupby_add_request(
            builder: Pin<&mut GroupByBuilder>,
            col_idx: i32,
            agg: UniquePtr<OwnedAggregation>,
        );

        /// Execute the groupby and return a combined table with key columns
        /// first, followed by aggregation result columns.
        fn groupby_execute(
            builder: Pin<&mut GroupByBuilder>,
            values: &OwnedTable,
        ) -> Result<UniquePtr<OwnedTable>>;

        /// Execute the groupby and return only the key columns result.
        fn groupby_execute_keys(
            builder: Pin<&mut GroupByBuilder>,
            values: &OwnedTable,
        ) -> Result<UniquePtr<OwnedTable>>;

        /// Execute the groupby and return only the values (aggregation results).
        fn groupby_execute_values(
            builder: Pin<&mut GroupByBuilder>,
            values: &OwnedTable,
        ) -> Result<UniquePtr<OwnedTable>>;

        // ── GroupBy Scan ──────────────────────────────────────────

        /// Opaque groupby scan builder.
        type GroupByScanBuilder;

        /// Create a groupby scan builder.
        fn groupby_scan_new(keys: &OwnedTable) -> UniquePtr<GroupByScanBuilder>;

        /// Add a scan aggregation request.
        /// agg_kind: 0=sum, 2=min, 3=max, 11=count, 12=rank
        fn groupby_scan_add_request(
            builder: Pin<&mut GroupByScanBuilder>,
            col_idx: i32,
            agg_kind: i32,
        );

        /// Execute the scan, returning keys + scan results.
        fn groupby_scan_execute(
            builder: Pin<&mut GroupByScanBuilder>,
            values: &OwnedTable,
        ) -> Result<UniquePtr<OwnedTable>>;

        // ── GroupBy Get Groups ────────────────────────────────────

        /// Opaque result of get_groups.
        type GroupByGroupsResult;

        /// Get grouped keys and offsets (no values).
        fn groupby_get_groups(keys: &OwnedTable) -> Result<UniquePtr<GroupByGroupsResult>>;

        /// Get grouped keys, offsets, and values.
        fn groupby_get_groups_with_values(
            keys: &OwnedTable,
            values: &OwnedTable,
        ) -> Result<UniquePtr<GroupByGroupsResult>>;

        /// Take keys from groups result.
        fn groupby_groups_take_keys(
            result: Pin<&mut GroupByGroupsResult>,
        ) -> Result<UniquePtr<OwnedTable>>;

        /// Take offsets from groups result.
        fn groupby_groups_take_offsets(
            result: Pin<&mut GroupByGroupsResult>,
        ) -> Result<UniquePtr<OwnedColumn>>;

        /// Take values from groups result.
        fn groupby_groups_take_values(
            result: Pin<&mut GroupByGroupsResult>,
        ) -> Result<UniquePtr<OwnedTable>>;

        // ── GroupBy Replace Nulls ─────────────────────────────────

        /// Replace nulls within groups.
        /// policies: 0=FORWARD, 1=BACKWARD (one per value column).
        fn groupby_replace_nulls(
            keys: &OwnedTable,
            values: &OwnedTable,
            policies: &[i32],
        ) -> Result<UniquePtr<OwnedTable>>;
    }
}
