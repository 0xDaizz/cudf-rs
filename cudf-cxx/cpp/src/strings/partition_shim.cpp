#include "strings/partition_shim.h"
#include <cudf/strings/split/partition.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <string>

namespace cudf_shims {

std::unique_ptr<OwnedTable> str_partition(
    const OwnedColumn& col, rust::Str delimiter)
{
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    std::string delim(delimiter.data(), delimiter.size());
    cudf::string_scalar scalar_delim(delim, true, stream);
    auto result = cudf::strings::partition(col.view(), scalar_delim, stream, mr);
    return std::make_unique<OwnedTable>(std::move(result));
}

std::unique_ptr<OwnedTable> str_rpartition(
    const OwnedColumn& col, rust::Str delimiter)
{
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    std::string delim(delimiter.data(), delimiter.size());
    cudf::string_scalar scalar_delim(delim, true, stream);
    auto result = cudf::strings::rpartition(col.view(), scalar_delim, stream, mr);
    return std::make_unique<OwnedTable>(std::move(result));
}

} // namespace cudf_shims
