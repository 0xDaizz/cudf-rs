#[cxx::bridge(namespace = "cudf_shims")]
pub mod ffi {
    unsafe extern "C++" {
        include!("io/parquet_shim.h");
        include!("table_shim.h");
        type OwnedTable = crate::table::ffi::OwnedTable;
        type OwnedTableWithMetadata = crate::table::ffi::OwnedTableWithMetadata;

        fn read_parquet(
            filepath: &str,
            columns: &[String],
            skip_rows: i64,
            num_rows: i64,
        ) -> Result<UniquePtr<OwnedTable>>;

        fn read_parquet_with_metadata(
            filepath: &str,
            columns: &[String],
            skip_rows: i64,
            num_rows: i64,
        ) -> Result<UniquePtr<OwnedTableWithMetadata>>;

        fn write_parquet(table: &OwnedTable, filepath: &str, compression: i32) -> Result<()>;
    }
}
