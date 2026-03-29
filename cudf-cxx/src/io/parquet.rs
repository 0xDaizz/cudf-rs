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

        // ── Chunked Parquet Reader ────────────────────────────────

        /// Opaque chunked parquet reader.
        type OwnedChunkedParquetReader;

        /// Create a chunked parquet reader.
        fn chunked_parquet_reader_create(
            filepath: &str,
            chunk_read_limit: i64,
        ) -> Result<UniquePtr<OwnedChunkedParquetReader>>;

        /// Check if there is more data to read.
        fn chunked_parquet_reader_has_next(reader: &OwnedChunkedParquetReader) -> Result<bool>;

        /// Read the next chunk.
        fn chunked_parquet_reader_read_chunk(
            reader: &OwnedChunkedParquetReader,
        ) -> Result<UniquePtr<OwnedTable>>;

        // ── Chunked Parquet Writer ────────────────────────────────

        /// Opaque chunked parquet writer.
        type OwnedChunkedParquetWriter;

        /// Create a chunked parquet writer.
        fn chunked_parquet_writer_create(
            filepath: &str,
            compression: i32,
        ) -> Result<UniquePtr<OwnedChunkedParquetWriter>>;

        /// Write a table chunk.
        fn chunked_parquet_writer_write(
            writer: Pin<&mut OwnedChunkedParquetWriter>,
            table: &OwnedTable,
        ) -> Result<()>;

        /// Finalize and close the writer.
        fn chunked_parquet_writer_close(writer: Pin<&mut OwnedChunkedParquetWriter>) -> Result<()>;

        // ── Parquet Metadata ─────────────────────────────────────

        /// Opaque wrapper for Parquet file metadata.
        type OwnedParquetMetadata;

        /// Read metadata from a Parquet file without reading the data.
        fn read_parquet_metadata(filepath: &str) -> Result<UniquePtr<OwnedParquetMetadata>>;

        /// Get number of rows.
        fn get_num_rows(meta: &OwnedParquetMetadata) -> i64;

        /// Get number of row groups.
        fn get_num_row_groups(meta: &OwnedParquetMetadata) -> i32;

        /// Get number of columns.
        fn get_num_columns(meta: &OwnedParquetMetadata) -> i32;

        /// Get column name by index.
        fn get_column_name(meta: &OwnedParquetMetadata, index: i32) -> String;
    }
}
