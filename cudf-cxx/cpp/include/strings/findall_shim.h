#pragma once

#include <cudf/strings/findall.hpp>
#include <memory>
#include "rust/cxx.h"
#include "column_shim.h"

namespace cudf_shims {

std::unique_ptr<OwnedColumn> str_findall(
    const OwnedColumn& col, rust::Str pattern);
std::unique_ptr<OwnedColumn> str_find_re(
    const OwnedColumn& col, rust::Str pattern);

} // namespace cudf_shims
