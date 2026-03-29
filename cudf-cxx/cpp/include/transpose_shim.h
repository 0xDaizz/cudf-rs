#pragma once

#include <cudf/transpose.hpp>
#include <cudf/table/table.hpp>
#include <memory>
#include "rust/cxx.h"
#include "table_shim.h"

namespace cudf_shims {

/// Transpose a table (swap rows and columns).
std::unique_ptr<OwnedTable> transpose_table(const OwnedTable& table);

} // namespace cudf_shims
