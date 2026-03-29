#pragma once

#include <cudf/strings/replace.hpp>
#include <cudf/strings/replace_re.hpp>
#include <memory>
#include "rust/cxx.h"
#include "column_shim.h"

namespace cudf_shims {

std::unique_ptr<OwnedColumn> str_replace(
    const OwnedColumn& col, rust::Str target, rust::Str replacement);
std::unique_ptr<OwnedColumn> str_replace_re(
    const OwnedColumn& col, rust::Str pattern, rust::Str replacement);

} // namespace cudf_shims
