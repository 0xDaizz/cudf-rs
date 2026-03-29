#[cxx::bridge(namespace = "cudf_shims")]
pub mod ffi {
    unsafe extern "C++" {
        include!("io/avro_shim.h");
        include!("table_shim.h");
        type OwnedTable = crate::table::ffi::OwnedTable;
        type OwnedTableWithMetadata = crate::table::ffi::OwnedTableWithMetadata;

        fn read_avro(filepath: &str, columns: &[String]) -> Result<UniquePtr<OwnedTable>>;
        fn read_avro_with_metadata(
            filepath: &str,
            columns: &[String],
        ) -> Result<UniquePtr<OwnedTableWithMetadata>>;
    }
}
