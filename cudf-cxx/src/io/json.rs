#[cxx::bridge(namespace = "cudf_shims")]
pub mod ffi {
    unsafe extern "C++" {
        include!("io/json_shim.h");
        include!("table_shim.h");
        type OwnedTable = crate::table::ffi::OwnedTable;
        type OwnedTableWithMetadata = crate::table::ffi::OwnedTableWithMetadata;

        fn read_json(filepath: &str, lines: bool) -> Result<UniquePtr<OwnedTable>>;
        fn read_json_with_metadata(
            filepath: &str,
            lines: bool,
        ) -> Result<UniquePtr<OwnedTableWithMetadata>>;
        fn write_json(table: &OwnedTable, filepath: &str, lines: bool) -> Result<()>;
    }
}
