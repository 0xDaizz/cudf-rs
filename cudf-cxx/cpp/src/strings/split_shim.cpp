#include "strings/split_shim.h"
#include <cudf/strings/split/split.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <string>

namespace cudf_shims {

std::unique_ptr<OwnedTable> str_split(
    const OwnedColumn& col, rust::Str delimiter, int32_t maxsplit)
{
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    std::string delim(delimiter.data(), delimiter.size());
    cudf::string_scalar scalar_delim(delim, true, stream);
    auto result = cudf::strings::split(col.view(), scalar_delim, maxsplit, stream, mr);
    return std::make_unique<OwnedTable>(std::move(result));
}

std::unique_ptr<OwnedTable> str_rsplit(
    const OwnedColumn& col, rust::Str delimiter, int32_t maxsplit)
{
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    std::string delim(delimiter.data(), delimiter.size());
    cudf::string_scalar scalar_delim(delim, true, stream);
    auto result = cudf::strings::rsplit(col.view(), scalar_delim, maxsplit, stream, mr);
    return std::make_unique<OwnedTable>(std::move(result));
}

} // namespace cudf_shims
