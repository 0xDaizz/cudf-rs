#pragma once

#include <cudf/strings/char_types/char_types.hpp>
#include <cudf/strings/char_types/char_types_enum.hpp>
#include <memory>
#include "rust/cxx.h"
#include "column_shim.h"

namespace cudf_shims {

/// Check if all characters of each string match the given character type(s).
/// `types` and `verify_types` are cudf::strings::string_character_types values.
std::unique_ptr<OwnedColumn> str_all_characters_of_type(
    const OwnedColumn& col, uint32_t types, uint32_t verify_types);

} // namespace cudf_shims
