#include "io/parquet_shim.h"
#include <cudf/io/parquet.hpp>
#include <string>
#include <vector>

namespace cudf_shims {

std::unique_ptr<OwnedTable> read_parquet(rust::Str filepath, rust::Slice<const rust::String> columns, int64_t skip_rows, int64_t num_rows) {
    std::string path(filepath.data(), filepath.size());
    auto source = cudf::io::source_info(path);
    auto builder = cudf::io::parquet_reader_options::builder(source);
    if (columns.size() > 0) {
        std::vector<std::string> cols;
        for (const auto& c : columns) cols.emplace_back(c.data(), c.size());
        builder.columns(cols);
    }
    if (skip_rows >= 0) builder.skip_rows(skip_rows);
    if (num_rows >= 0) builder.num_rows(num_rows);
    auto result = cudf::io::read_parquet(builder.build());
    return std::make_unique<OwnedTable>(std::move(result.tbl));
}

void write_parquet(const OwnedTable& table, rust::Str filepath, int32_t compression) {
    std::string path(filepath.data(), filepath.size());
    auto sink = cudf::io::sink_info(path);
    auto builder = cudf::io::parquet_writer_options::builder(sink, table.view());
    builder.compression(static_cast<cudf::io::compression_type>(compression));
    cudf::io::write_parquet(builder.build());
}

std::unique_ptr<OwnedTableWithMetadata> read_parquet_with_metadata(rust::Str filepath, rust::Slice<const rust::String> columns, int64_t skip_rows, int64_t num_rows) {
    std::string path(filepath.data(), filepath.size());
    auto source = cudf::io::source_info(path);
    auto builder = cudf::io::parquet_reader_options::builder(source);
    if (columns.size() > 0) {
        std::vector<std::string> cols;
        for (const auto& c : columns) cols.emplace_back(c.data(), c.size());
        builder.columns(cols);
    }
    if (skip_rows >= 0) builder.skip_rows(skip_rows);
    if (num_rows >= 0) builder.num_rows(num_rows);
    auto result = cudf::io::read_parquet(builder.build());

    std::vector<std::string> names;
    for (const auto& info : result.metadata.schema_info) {
        names.push_back(info.name);
    }

    return std::make_unique<OwnedTableWithMetadata>(std::move(result.tbl), std::move(names));
}

} // namespace cudf_shims
