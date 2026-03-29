#pragma once

#include <cudf/strings/reverse.hpp>
#include <memory>
#include "rust/cxx.h"
#include "column_shim.h"

namespace cudf_shims {

std::unique_ptr<OwnedColumn> str_reverse(const OwnedColumn& col);

} // namespace cudf_shims
