#[cxx::bridge(namespace = "cudf_shims")]
pub mod ffi {
    unsafe extern "C++" {
        include!("io/avro_shim.h");
        include!("table_shim.h");
        type OwnedTable = crate::table::ffi::OwnedTable;

        fn read_avro(filepath: &str, columns: &[String]) -> Result<UniquePtr<OwnedTable>>;
    }
}
