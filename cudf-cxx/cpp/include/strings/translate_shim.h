#pragma once

#include <cudf/strings/translate.hpp>
#include <memory>
#include "rust/cxx.h"
#include "column_shim.h"

namespace cudf_shims {

/// Translate characters using parallel arrays of source/target code points.
/// A target of 0 means remove the character.
std::unique_ptr<OwnedColumn> str_translate(
    const OwnedColumn& col,
    rust::Slice<const uint32_t> src_chars,
    rust::Slice<const uint32_t> dst_chars);

/// Filter characters by keeping or removing specified ranges.
/// Each range is a pair (lo, hi) stored as consecutive u32 values.
/// `keep`: true = KEEP matching, false = REMOVE matching.
std::unique_ptr<OwnedColumn> str_filter_characters(
    const OwnedColumn& col,
    rust::Slice<const uint32_t> range_pairs,
    bool keep,
    rust::Str replacement);

} // namespace cudf_shims
