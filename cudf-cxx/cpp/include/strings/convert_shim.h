#pragma once

#include <cudf/strings/convert/convert_integers.hpp>
#include <cudf/strings/convert/convert_floats.hpp>
#include <memory>
#include "rust/cxx.h"
#include "column_shim.h"

namespace cudf_shims {

std::unique_ptr<OwnedColumn> str_to_integers(
    const OwnedColumn& col, int32_t type_id);
std::unique_ptr<OwnedColumn> str_from_integers(const OwnedColumn& col);
std::unique_ptr<OwnedColumn> str_to_floats(
    const OwnedColumn& col, int32_t type_id);
std::unique_ptr<OwnedColumn> str_from_floats(const OwnedColumn& col);

} // namespace cudf_shims
