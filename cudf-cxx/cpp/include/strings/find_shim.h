#pragma once

#include <cudf/strings/find.hpp>
#include <memory>
#include "rust/cxx.h"
#include "column_shim.h"

namespace cudf_shims {

std::unique_ptr<OwnedColumn> str_find(
    const OwnedColumn& col, rust::Str target, int32_t start);
std::unique_ptr<OwnedColumn> str_rfind(
    const OwnedColumn& col, rust::Str target);
std::unique_ptr<OwnedColumn> str_starts_with(
    const OwnedColumn& col, rust::Str target);
std::unique_ptr<OwnedColumn> str_ends_with(
    const OwnedColumn& col, rust::Str target);

} // namespace cudf_shims
