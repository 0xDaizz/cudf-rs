#[cxx::bridge(namespace = "cudf_shims")]
pub mod ffi {
    unsafe extern "C++" {
        include!("io/csv_shim.h");
        include!("table_shim.h");
        type OwnedTable = crate::table::ffi::OwnedTable;
        type OwnedTableWithMetadata = crate::table::ffi::OwnedTableWithMetadata;

        fn read_csv(
            filepath: &str,
            delimiter: u8,
            header_row: i32,
            skip_rows: i64,
            num_rows: i64,
        ) -> Result<UniquePtr<OwnedTable>>;

        fn read_csv_with_metadata(
            filepath: &str,
            delimiter: u8,
            header_row: i32,
            skip_rows: i64,
            num_rows: i64,
        ) -> Result<UniquePtr<OwnedTableWithMetadata>>;

        fn write_csv(
            table: &OwnedTable,
            filepath: &str,
            delimiter: u8,
            include_header: bool,
        ) -> Result<()>;
    }
}
