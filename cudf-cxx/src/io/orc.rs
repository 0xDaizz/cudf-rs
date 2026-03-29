#[cxx::bridge(namespace = "cudf_shims")]
pub mod ffi {
    unsafe extern "C++" {
        include!("io/orc_shim.h");
        include!("table_shim.h");
        type OwnedTable = crate::table::ffi::OwnedTable;

        fn read_orc(
            filepath: &str,
            columns: &[String],
            skip_rows: i64,
            num_rows: i64,
        ) -> Result<UniquePtr<OwnedTable>>;
        fn write_orc(table: &OwnedTable, filepath: &str, compression: i32) -> Result<()>;
    }
}
