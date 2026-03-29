#include "strings/convert_shim.h"
#include <cudf/strings/convert/convert_integers.hpp>
#include <cudf/strings/convert/convert_floats.hpp>
#include <cudf/utilities/default_stream.hpp>

namespace cudf_shims {

std::unique_ptr<OwnedColumn> str_to_integers(
    const OwnedColumn& col, int32_t type_id)
{
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    auto tid = static_cast<cudf::type_id>(type_id);
    auto result = cudf::strings::to_integers(
        col.view(), cudf::data_type{tid}, stream, mr);
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> str_from_integers(const OwnedColumn& col) {
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    auto result = cudf::strings::from_integers(col.view(), stream, mr);
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> str_to_floats(
    const OwnedColumn& col, int32_t type_id)
{
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    auto tid = static_cast<cudf::type_id>(type_id);
    auto result = cudf::strings::to_floats(
        col.view(), cudf::data_type{tid}, stream, mr);
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> str_from_floats(const OwnedColumn& col) {
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    auto result = cudf::strings::from_floats(col.view(), stream, mr);
    return std::make_unique<OwnedColumn>(std::move(result));
}

} // namespace cudf_shims
