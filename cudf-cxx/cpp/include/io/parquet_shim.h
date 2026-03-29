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

} // namespace cudf_shims
