#include "strings/attributes_shim.h"
#include <cudf/strings/attributes.hpp>
#include <cudf/utilities/default_stream.hpp>

namespace cudf_shims {

std::unique_ptr<OwnedColumn> str_count_characters(const OwnedColumn& col) {
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    auto result = cudf::strings::count_characters(col.view(), stream, mr);
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> str_count_bytes(const OwnedColumn& col) {
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    auto result = cudf::strings::count_bytes(col.view(), stream, mr);
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> str_code_points(const OwnedColumn& col) {
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    auto result = cudf::strings::code_points(col.view(), stream, mr);
    return std::make_unique<OwnedColumn>(std::move(result));
}

} // namespace cudf_shims
