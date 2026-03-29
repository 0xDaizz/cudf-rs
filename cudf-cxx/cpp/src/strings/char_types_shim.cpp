#include "strings/char_types_shim.h"
#include <cudf/strings/char_types/char_types.hpp>
#include <cudf/strings/char_types/char_types_enum.hpp>
#include <cudf/utilities/default_stream.hpp>

namespace cudf_shims {

std::unique_ptr<OwnedColumn> str_all_characters_of_type(
    const OwnedColumn& col, uint32_t types, uint32_t verify_types)
{
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    auto t = static_cast<cudf::strings::string_character_types>(types);
    auto v = static_cast<cudf::strings::string_character_types>(verify_types);
    auto result = cudf::strings::all_characters_of_type(col.view(), t, v, stream, mr);
    return std::make_unique<OwnedColumn>(std::move(result));
}

} // namespace cudf_shims
