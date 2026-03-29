#pragma once
#include <cudf/io/csv.hpp>
#include <memory>
#include "rust/cxx.h"
#include "table_shim.h"

namespace cudf_shims {
std::unique_ptr<OwnedTable> read_csv(rust::Str filepath, uint8_t delimiter, int32_t header_row, int64_t skip_rows, int64_t num_rows);
void write_csv(const OwnedTable& table, rust::Str filepath, uint8_t delimiter, bool include_header);
} // namespace cudf_shims
