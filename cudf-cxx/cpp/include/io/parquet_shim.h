#pragma once
#include <cudf/io/parquet.hpp>
#include <memory>
#include "rust/cxx.h"
#include "table_shim.h"

namespace cudf_shims {
std::unique_ptr<OwnedTable> read_parquet(rust::Str filepath, rust::Slice<const rust::String> columns, int64_t skip_rows, int64_t num_rows);
std::unique_ptr<OwnedTableWithMetadata> read_parquet_with_metadata(rust::Str filepath, rust::Slice<const rust::String> columns, int64_t skip_rows, int64_t num_rows);
void write_parquet(const OwnedTable& table, rust::Str filepath, int32_t compression);

// ── Chunked Parquet Reader ────────────────────────────────────

/// Opaque wrapper around cudf::io::chunked_parquet_reader.
struct OwnedChunkedParquetReader {
    std::unique_ptr<cudf::io::chunked_parquet_reader> inner;

    explicit OwnedChunkedParquetReader(
        std::unique_ptr<cudf::io::chunked_parquet_reader> r)
        : inner(std::move(r)) {}
};

/// Create a chunked parquet reader for the given file.
std::unique_ptr<OwnedChunkedParquetReader> chunked_parquet_reader_create(
    rust::Str filepath, int64_t chunk_read_limit);

/// Check if there is more data to read.
bool chunked_parquet_reader_has_next(const OwnedChunkedParquetReader& reader);

/// Read the next chunk, returning a table.
std::unique_ptr<OwnedTable> chunked_parquet_reader_read_chunk(
    const OwnedChunkedParquetReader& reader);

// ── Chunked Parquet Writer ────────────────────────────────────

/// Opaque wrapper around cudf::io::chunked_parquet_writer.
struct OwnedChunkedParquetWriter {
    std::unique_ptr<cudf::io::chunked_parquet_writer> inner;

    explicit OwnedChunkedParquetWriter(
        std::unique_ptr<cudf::io::chunked_parquet_writer> w)
        : inner(std::move(w)) {}
};

/// Create a chunked parquet writer for the given file.
std::unique_ptr<OwnedChunkedParquetWriter> chunked_parquet_writer_create(
    rust::Str filepath, int32_t compression);

/// Write a table chunk.
void chunked_parquet_writer_write(
    OwnedChunkedParquetWriter& writer, const OwnedTable& table);

/// Finalize and close the chunked writer.
void chunked_parquet_writer_close(OwnedChunkedParquetWriter& writer);

// ── Parquet Metadata ─────────────────────────────────────────

/// Holds basic metadata from a Parquet file.
struct OwnedParquetMetadata {
    int64_t num_rows_val;
    int32_t num_row_groups_val;
    std::vector<std::string> column_names;
};

/// Read metadata from a Parquet file without reading the data.
std::unique_ptr<OwnedParquetMetadata> read_parquet_metadata(rust::Str filepath);

/// Accessor: get number of rows.
inline int64_t get_num_rows(const OwnedParquetMetadata& meta) { return meta.num_rows_val; }

/// Accessor: get number of row groups.
inline int32_t get_num_row_groups(const OwnedParquetMetadata& meta) { return meta.num_row_groups_val; }

/// Accessor: get number of columns.
inline int32_t get_num_columns(const OwnedParquetMetadata& meta) {
    return static_cast<int32_t>(meta.column_names.size());
}

/// Accessor: get column name by index.
inline rust::String get_column_name(const OwnedParquetMetadata& meta, int32_t index) {
    auto& name = meta.column_names.at(index);
    return rust::String(name.data(), name.size());
}

} // namespace cudf_shims
