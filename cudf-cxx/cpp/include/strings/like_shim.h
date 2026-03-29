#pragma once

#include <cudf/strings/contains.hpp>
#include <memory>
#include "rust/cxx.h"
#include "column_shim.h"

namespace cudf_shims {

std::unique_ptr<OwnedColumn> str_like(
    const OwnedColumn& col, rust::Str pattern, rust::Str escape_char);

} // namespace cudf_shims
