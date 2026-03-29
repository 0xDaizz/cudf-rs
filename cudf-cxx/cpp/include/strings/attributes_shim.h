#pragma once

#include <cudf/strings/attributes.hpp>
#include <memory>
#include "rust/cxx.h"
#include "column_shim.h"

namespace cudf_shims {

std::unique_ptr<OwnedColumn> str_count_characters(const OwnedColumn& col);
std::unique_ptr<OwnedColumn> str_count_bytes(const OwnedColumn& col);
std::unique_ptr<OwnedColumn> str_code_points(const OwnedColumn& col);

} // namespace cudf_shims
