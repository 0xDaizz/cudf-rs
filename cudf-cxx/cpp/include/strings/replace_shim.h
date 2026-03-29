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

/// Replace characters in the [start, stop) range with `replacement`.
std::unique_ptr<OwnedColumn> str_replace_slice(
    const OwnedColumn& col, rust::Str replacement, int32_t start, int32_t stop);

/// Replace multiple target strings with corresponding replacements.
std::unique_ptr<OwnedColumn> str_replace_multiple(
    const OwnedColumn& col, const OwnedColumn& targets, const OwnedColumn& replacements);

/// Replace regex matches using back-references in the replacement template.
std::unique_ptr<OwnedColumn> str_replace_with_backrefs(
    const OwnedColumn& col, rust::Str pattern, rust::Str replacement);

/// Replace multiple regex patterns with corresponding replacements from a column.
std::unique_ptr<OwnedColumn> str_replace_re_multiple(
    const OwnedColumn& col,
    rust::Slice<const rust::String> patterns,
    const OwnedColumn& replacements);

} // namespace cudf_shims
