#include "strings/like_shim.h"
#include <cudf/strings/contains.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <string>

namespace cudf_shims {

std::unique_ptr<OwnedColumn> str_like(
    const OwnedColumn& col, rust::Str pattern, rust::Str escape_char)
{
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    std::string pat(pattern.data(), pattern.size());
    std::string esc(escape_char.data(), escape_char.size());
    auto result = cudf::strings::like(col.view(), pat, esc, stream, mr);
    return std::make_unique<OwnedColumn>(std::move(result));
}

} // namespace cudf_shims
