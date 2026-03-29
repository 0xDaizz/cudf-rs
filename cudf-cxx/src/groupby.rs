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
    }
}
