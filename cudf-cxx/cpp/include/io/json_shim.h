#pragma once
#include <cudf/io/json.hpp>
#include <memory>
#include "rust/cxx.h"
#include "table_shim.h"

namespace cudf_shims {
std::unique_ptr<OwnedTable> read_json(rust::Str filepath, bool lines);
void write_json(const OwnedTable& table, rust::Str filepath, bool lines);
} // namespace cudf_shims
