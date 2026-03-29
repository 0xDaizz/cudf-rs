#[cxx::bridge(namespace = "cudf_shims")]
pub mod ffi {
    unsafe extern "C++" {
        include!("io/json_shim.h");
        include!("table_shim.h");
        type OwnedTable = crate::table::ffi::OwnedTable;

        fn read_json(filepath: &str, lines: bool) -> Result<UniquePtr<OwnedTable>>;
        fn write_json(table: &OwnedTable, filepath: &str, lines: bool) -> Result<()>;
    }
}
