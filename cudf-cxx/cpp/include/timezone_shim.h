#pragma once

#include <cudf/timezone.hpp>
#include <memory>
#include "rust/cxx.h"
#include "table_shim.h"

namespace cudf_shims {

/// Create a timezone transition table for converting ORC timestamps to UTC.
std::unique_ptr<OwnedTable> make_timezone_transition_table(
    rust::Str timezone_name);

} // namespace cudf_shims
