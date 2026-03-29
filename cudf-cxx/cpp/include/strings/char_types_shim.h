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

/// Filter characters of the given types, replacing removed characters with `replacement`.
/// `types_to_remove` = character types to remove.
/// `types_to_keep` = character types to keep (NONE = 0 for default).
std::unique_ptr<OwnedColumn> str_filter_characters_of_type(
    const OwnedColumn& col, uint32_t types_to_remove, rust::Str replacement, uint32_t types_to_keep);

} // namespace cudf_shims
