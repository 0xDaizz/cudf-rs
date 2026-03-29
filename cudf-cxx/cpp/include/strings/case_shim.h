#pragma once

#include <cudf/strings/case.hpp>
#include <cudf/strings/capitalize.hpp>
#include <memory>
#include "rust/cxx.h"
#include "column_shim.h"

namespace cudf_shims {

std::unique_ptr<OwnedColumn> str_to_upper(const OwnedColumn& col);
std::unique_ptr<OwnedColumn> str_to_lower(const OwnedColumn& col);
std::unique_ptr<OwnedColumn> str_swapcase(const OwnedColumn& col);

/// Capitalize first character of each string (or each word if delimiters given).
std::unique_ptr<OwnedColumn> str_capitalize(const OwnedColumn& col, rust::Str delimiters);

/// Title-case each word in each string.
std::unique_ptr<OwnedColumn> str_title(const OwnedColumn& col);

/// Check if each string is in title-case.
std::unique_ptr<OwnedColumn> str_is_title(const OwnedColumn& col);

} // namespace cudf_shims
