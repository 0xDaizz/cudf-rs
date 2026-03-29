#include "io/json_shim.h"
#include <cudf/io/json.hpp>
#include <string>

namespace cudf_shims {

std::unique_ptr<OwnedTable> read_json(rust::Str filepath, bool lines) {
    std::string path(filepath.data(), filepath.size());
    auto source = cudf::io::source_info(path);
    auto builder = cudf::io::json_reader_options::builder(source);
    builder.lines(lines);
    auto result = cudf::io::read_json(builder.build());
    return std::make_unique<OwnedTable>(std::move(result.tbl));
}

void write_json(const OwnedTable& table, rust::Str filepath, bool lines) {
    std::string path(filepath.data(), filepath.size());
    auto sink = cudf::io::sink_info(path);
    auto builder = cudf::io::json_writer_options::builder(sink, table.view());
    builder.lines(lines);
    cudf::io::write_json(builder.build());
}

} // namespace cudf_shims
