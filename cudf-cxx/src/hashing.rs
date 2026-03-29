//! Bridge definitions for libcudf hashing operations.
//!
//! Provides GPU-accelerated hashing of table rows using various algorithms.

#[cxx::bridge(namespace = "cudf_shims")]
pub mod ffi {
    unsafe extern "C++" {
        include!("hashing_shim.h");
        include!("table_shim.h");
        include!("column_shim.h");
        type OwnedTable = crate::table::ffi::OwnedTable;
        type OwnedColumn = crate::column::ffi::OwnedColumn;

        /// Hash each row using MurmurHash3 (32-bit).
        fn hash_murmur3(table: &OwnedTable, seed: u32) -> Result<UniquePtr<OwnedColumn>>;

        /// Hash each row using xxHash64.
        fn hash_xxhash64(table: &OwnedTable, seed: u64) -> Result<UniquePtr<OwnedColumn>>;

        /// Hash each row using MD5 (returns a string column).
        fn hash_md5(table: &OwnedTable) -> Result<UniquePtr<OwnedColumn>>;

        /// Hash each row using SHA-256 (returns a string column).
        fn hash_sha256(table: &OwnedTable) -> Result<UniquePtr<OwnedColumn>>;

        /// Hash each row using SHA-1 (returns a string column).
        fn hash_sha1(table: &OwnedTable) -> Result<UniquePtr<OwnedColumn>>;

        /// Hash each row using SHA-224 (returns a string column).
        fn hash_sha224(table: &OwnedTable) -> Result<UniquePtr<OwnedColumn>>;

        /// Hash each row using SHA-384 (returns a string column).
        fn hash_sha384(table: &OwnedTable) -> Result<UniquePtr<OwnedColumn>>;

        /// Hash each row using SHA-512 (returns a string column).
        fn hash_sha512(table: &OwnedTable) -> Result<UniquePtr<OwnedColumn>>;

        /// Hash each row using xxHash32.
        fn hash_xxhash32(table: &OwnedTable, seed: u32) -> Result<UniquePtr<OwnedColumn>>;

        /// Hash each row using MurmurHash3 x64 128-bit. Returns a 2-column table.
        fn hash_murmurhash3_x64_128(table: &OwnedTable, seed: u64)
        -> Result<UniquePtr<OwnedTable>>;
    }
}
