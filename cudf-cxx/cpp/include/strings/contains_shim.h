#pragma once

#include <cudf/strings/contains.hpp>
#include <memory>
#include "rust/cxx.h"
#include "column_shim.h"

namespace cudf_shims {

std::unique_ptr<OwnedColumn> str_contains(
    const OwnedColumn& col, rust::Str target);
std::unique_ptr<OwnedColumn> str_contains_re(
    const OwnedColumn& col, rust::Str pattern);
std::unique_ptr<OwnedColumn> str_matches_re(
    const OwnedColumn& col, rust::Str pattern);
std::unique_ptr<OwnedColumn> str_count_re(
    const OwnedColumn& col, rust::Str pattern);

} // namespace cudf_shims
