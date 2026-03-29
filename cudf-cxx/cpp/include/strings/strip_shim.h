#pragma once

#include <cudf/strings/strip.hpp>
#include <memory>
#include "rust/cxx.h"
#include "column_shim.h"

namespace cudf_shims {

std::unique_ptr<OwnedColumn> str_strip(const OwnedColumn& col);
std::unique_ptr<OwnedColumn> str_lstrip(const OwnedColumn& col);
std::unique_ptr<OwnedColumn> str_rstrip(const OwnedColumn& col);

} // namespace cudf_shims
