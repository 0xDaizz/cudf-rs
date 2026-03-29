#include "io/csv_shim.h"
#include <cudf/io/csv.hpp>
#include <string>

namespace cudf_shims {

std::unique_ptr<OwnedTable> read_csv(rust::Str filepath, uint8_t delimiter, int32_t header_row, int64_t skip_rows, int64_t num_rows) {
    std::string path(filepath.data(), filepath.size());
    auto source = cudf::io::source_info(path);
    auto builder = cudf::io::csv_reader_options::builder(source);
    builder.delimiter(static_cast<char>(delimiter));
    if (header_row >= 0) builder.header(header_row);
    else builder.header(-1);
    if (skip_rows >= 0) builder.skiprows(skip_rows);
    if (num_rows >= 0) builder.nrows(num_rows);
    auto result = cudf::io::read_csv(builder.build());
    return std::make_unique<OwnedTable>(std::move(result.tbl));
}

void write_csv(const OwnedTable& table, rust::Str filepath, uint8_t delimiter, bool include_header) {
    std::string path(filepath.data(), filepath.size());
    auto sink = cudf::io::sink_info(path);
    auto builder = cudf::io::csv_writer_options::builder(sink, table.view());
    builder.inter_column_delimiter(static_cast<char>(delimiter));
    builder.include_header(include_header);
    cudf::io::write_csv(builder.build());
}

} // namespace cudf_shims
