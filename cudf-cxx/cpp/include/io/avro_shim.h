#pragma once
#include <cudf/io/avro.hpp>
#include <memory>
#include "rust/cxx.h"
#include "table_shim.h"

namespace cudf_shims {
std::unique_ptr<OwnedTable> read_avro(rust::Str filepath, rust::Slice<const rust::String> columns);
} // namespace cudf_shims
