#pragma once

#include <cudf/strings/wrap.hpp>
#include <memory>
#include "rust/cxx.h"
#include "column_shim.h"

namespace cudf_shims {

std::unique_ptr<OwnedColumn> str_wrap(const OwnedColumn& col, int32_t width);

} // namespace cudf_shims
