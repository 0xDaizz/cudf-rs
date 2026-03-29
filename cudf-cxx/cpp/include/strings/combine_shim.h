#pragma once

#include <cudf/strings/combine.hpp>
#include <memory>
#include "rust/cxx.h"
#include "column_shim.h"

namespace cudf_shims {

std::unique_ptr<OwnedColumn> str_join(
    const OwnedColumn& col, rust::Str separator);

} // namespace cudf_shims
