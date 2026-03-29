//! Bridge definitions for libcudf dictionary encoding operations.
//!
//! Provides GPU-accelerated dictionary encoding and decoding of columns.

#[cxx::bridge(namespace = "cudf_shims")]
pub mod ffi {
    unsafe extern "C++" {
        include!("dictionary_shim.h");
        include!("column_shim.h");
        include!("scalar_shim.h");
        include!("table_shim.h");

        type OwnedColumn = crate::column::ffi::OwnedColumn;
        type OwnedScalar = crate::scalar::ffi::OwnedScalar;
        type OwnedTable = crate::table::ffi::OwnedTable;

        // ── Encode / Decode ───────────────────────────────────────

        /// Dictionary-encode a column. Returns a DICTIONARY type column
        /// with keys (unique sorted values) and indices.
        fn dictionary_encode(col: &OwnedColumn) -> Result<UniquePtr<OwnedColumn>>;

        /// Decode a dictionary column back to its original representation.
        fn dictionary_decode(col: &OwnedColumn) -> Result<UniquePtr<OwnedColumn>>;

        // ── Search ────────────────────────────────────────────────

        /// Get the index of a key in a dictionary column. Returns a scalar.
        fn dictionary_get_index(
            col: &OwnedColumn,
            key: &OwnedScalar,
        ) -> Result<UniquePtr<OwnedScalar>>;

        // ── Key Management ────────────────────────────────────────

        /// Add new keys to a dictionary column.
        fn dictionary_add_keys(
            col: &OwnedColumn,
            new_keys: &OwnedColumn,
        ) -> Result<UniquePtr<OwnedColumn>>;

        /// Remove specified keys from a dictionary column.
        fn dictionary_remove_keys(
            col: &OwnedColumn,
            keys_to_remove: &OwnedColumn,
        ) -> Result<UniquePtr<OwnedColumn>>;

        /// Remove unused keys from a dictionary column.
        fn dictionary_remove_unused_keys(col: &OwnedColumn) -> Result<UniquePtr<OwnedColumn>>;

        /// Replace all keys in a dictionary column with new keys.
        fn dictionary_set_keys(
            col: &OwnedColumn,
            new_keys: &OwnedColumn,
        ) -> Result<UniquePtr<OwnedColumn>>;

        // ── Match Dictionaries ────────────────────────────────────

        /// Opaque builder for collecting dictionary columns.
        type DictionaryMatchBuilder;

        fn dictionary_match_builder_new() -> UniquePtr<DictionaryMatchBuilder>;

        fn add_column(self: Pin<&mut DictionaryMatchBuilder>, col: UniquePtr<OwnedColumn>);

        fn num_columns(self: &DictionaryMatchBuilder) -> i32;

        /// Match dictionaries across multiple columns.
        fn dictionary_match_dictionaries(
            builder: UniquePtr<DictionaryMatchBuilder>,
        ) -> Result<UniquePtr<OwnedTable>>;
    }
}
