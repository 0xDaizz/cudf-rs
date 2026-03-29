#pragma once
#include <cudf/io/orc.hpp>
#include <memory>
#include "rust/cxx.h"
#include "table_shim.h"

namespace cudf_shims {
std::unique_ptr<OwnedTable> read_orc(rust::Str filepath, rust::Slice<const rust::String> columns, int64_t skip_rows, int64_t num_rows);
void write_orc(const OwnedTable& table, rust::Str filepath, int32_t compression);
} // namespace cudf_shims
