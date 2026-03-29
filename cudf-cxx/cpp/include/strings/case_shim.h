#pragma once

#include <cudf/strings/case.hpp>
#include <memory>
#include "rust/cxx.h"
#include "column_shim.h"

namespace cudf_shims {

std::unique_ptr<OwnedColumn> str_to_upper(const OwnedColumn& col);
std::unique_ptr<OwnedColumn> str_to_lower(const OwnedColumn& col);
std::unique_ptr<OwnedColumn> str_swapcase(const OwnedColumn& col);
std::unique_ptr<OwnedColumn> str_capitalize(const OwnedColumn& col);
std::unique_ptr<OwnedColumn> str_title(const OwnedColumn& col);

} // namespace cudf_shims
