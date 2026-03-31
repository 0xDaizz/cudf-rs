//! Bridge definitions for libcudf reduction operations.
//!
//! Provides GPU-accelerated reduce, scan, and segmented reduce functions
//! for columns.

#[cxx::bridge(namespace = "cudf_shims")]
pub mod ffi {
    unsafe extern "C++" {
        include!("column_shim.h");
        include!("scalar_shim.h");
        include!("reduction_shim.h");

        type OwnedColumn = crate::column::ffi::OwnedColumn;
        type OwnedScalar = crate::scalar::ffi::OwnedScalar;

        /// Reduce a column to a single scalar value.
        /// agg_kind: 0=sum, 1=product, 2=min, 3=max, 4=sum_of_squares,
        ///           5=mean, 6=variance, 7=std, 8=any, 9=all, 10=median
        fn reduce(
            col: &OwnedColumn,
            agg_kind: i32,
            output_type_id: i32,
        ) -> Result<UniquePtr<OwnedScalar>>;

        /// Prefix scan (cumulative operation).
        /// agg_kind: 0=sum, 1=product, 2=min, 3=max
        /// inclusive: true for inclusive scan, false for exclusive
        fn scan(
            col: &OwnedColumn,
            agg_kind: i32,
            inclusive: bool,
        ) -> Result<UniquePtr<OwnedColumn>>;

        /// Segmented reduce: reduce within segments defined by offsets.
        fn segmented_reduce(
            col: &OwnedColumn,
            offsets: &OwnedColumn,
            agg_kind: i32,
            output_type_id: i32,
            include_nulls: bool,
        ) -> Result<UniquePtr<OwnedColumn>>;

        // ── MinMax ────────────────────────────────────────────────

        /// Opaque result holding min and max scalars.
        type MinMaxResult;

        /// Compute min and max of a column simultaneously.
        fn minmax(col: &OwnedColumn) -> Result<UniquePtr<MinMaxResult>>;

        /// Move min scalar out of MinMaxResult.
        fn minmax_take_min(result: Pin<&mut MinMaxResult>) -> Result<UniquePtr<OwnedScalar>>;

        /// Move max scalar out of MinMaxResult.
        fn minmax_take_max(result: Pin<&mut MinMaxResult>) -> Result<UniquePtr<OwnedScalar>>;

        /// Reduce with an initial value.
        fn reduce_with_init(
            col: &OwnedColumn,
            agg_kind: i32,
            output_type_id: i32,
            init: &OwnedScalar,
        ) -> Result<UniquePtr<OwnedScalar>>;

        /// Check if a reduction aggregation is valid for a given source data type.
        fn is_valid_reduction_aggregation(source_type_id: i32, agg_kind: i32) -> Result<bool>;

        /// Reduce: variance with custom ddof.
        fn reduce_var_with_ddof(
            col: &OwnedColumn,
            ddof: i32,
            output_type_id: i32,
        ) -> Result<UniquePtr<OwnedScalar>>;

        /// Reduce: standard deviation with custom ddof.
        fn reduce_std_with_ddof(
            col: &OwnedColumn,
            ddof: i32,
            output_type_id: i32,
        ) -> Result<UniquePtr<OwnedScalar>>;
    }
}
