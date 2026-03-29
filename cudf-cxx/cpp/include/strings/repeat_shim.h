#pragma once

#include <cudf/strings/repeat_strings.hpp>
#include <memory>
#include "rust/cxx.h"
#include "column_shim.h"

namespace cudf_shims {

std::unique_ptr<OwnedColumn> str_repeat(
    const OwnedColumn& col, int32_t count);

} // namespace cudf_shims
