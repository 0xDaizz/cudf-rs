#pragma once

#include <cudf/strings/padding.hpp>
#include <cudf/strings/side_type.hpp>
#include <memory>
#include "rust/cxx.h"
#include "column_shim.h"

namespace cudf_shims {

std::unique_ptr<OwnedColumn> str_pad(
    const OwnedColumn& col, int32_t width, int32_t side, rust::Str fill_char);
std::unique_ptr<OwnedColumn> str_zfill(
    const OwnedColumn& col, int32_t width);

} // namespace cudf_shims
