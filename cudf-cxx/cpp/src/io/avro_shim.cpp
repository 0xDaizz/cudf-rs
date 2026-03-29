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

} // namespace cudf_shims
