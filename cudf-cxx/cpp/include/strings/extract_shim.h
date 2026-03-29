#pragma once

#include <cudf/strings/extract.hpp>
#include <memory>
#include "rust/cxx.h"
#include "column_shim.h"
#include "table_shim.h"

namespace cudf_shims {

std::unique_ptr<OwnedTable> str_extract(
    const OwnedColumn& col, rust::Str pattern);

} // namespace cudf_shims
