#include "io/avro_shim.h"
#include <cudf/io/avro.hpp>
#include <string>
#include <vector>

namespace cudf_shims {

std::unique_ptr<OwnedTable> read_avro(rust::Str filepath, rust::Slice<const rust::String> columns) {
    std::string path(filepath.data(), filepath.size());
    auto source = cudf::io::source_info(path);
    auto builder = cudf::io::avro_reader_options::builder(source);
    if (columns.size() > 0) {
        std::vector<std::string> cols;
        for (const auto& c : columns) cols.emplace_back(c.data(), c.size());
        builder.columns(cols);
    }
    auto result = cudf::io::read_avro(builder.build());
    return std::make_unique<OwnedTable>(std::move(result.tbl));
}

std::unique_ptr<OwnedTableWithMetadata> read_avro_with_metadata(rust::Str filepath, rust::Slice<const rust::String> columns) {
    std::string path(filepath.data(), filepath.size());
    auto source = cudf::io::source_info(path);
    auto builder = cudf::io::avro_reader_options::builder(source);
    if (columns.size() > 0) {
        std::vector<std::string> cols;
        for (const auto& c : columns) cols.emplace_back(c.data(), c.size());
        builder.columns(cols);
    }
    auto result = cudf::io::read_avro(builder.build());

    std::vector<std::string> names;
    for (const auto& info : result.metadata.schema_info) {
        names.push_back(info.name);
    }

    return std::make_unique<OwnedTableWithMetadata>(std::move(result.tbl), std::move(names));
}

} // namespace cudf_shims
