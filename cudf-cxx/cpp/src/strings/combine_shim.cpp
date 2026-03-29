#include "strings/combine_shim.h"
#include <cudf/strings/combine.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <string>

namespace cudf_shims {

std::unique_ptr<OwnedColumn> str_join(
    const OwnedColumn& col, rust::Str separator)
{
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    std::string sep(separator.data(), separator.size());
    cudf::string_scalar scalar_sep(sep, true, stream);
    cudf::string_scalar scalar_narep("", false, stream);
    auto result = cudf::strings::join_strings(
        col.view(), scalar_sep, scalar_narep, stream, mr);
    return std::make_unique<OwnedColumn>(std::move(result));
}

} // namespace cudf_shims
