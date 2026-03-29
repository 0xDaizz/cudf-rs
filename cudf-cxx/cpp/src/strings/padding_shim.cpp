#include "strings/padding_shim.h"
#include <cudf/strings/padding.hpp>
#include <cudf/strings/side_type.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <string>

namespace cudf_shims {

std::unique_ptr<OwnedColumn> str_pad(
    const OwnedColumn& col, int32_t width, int32_t side, rust::Str fill_char)
{
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    auto side_type = static_cast<cudf::strings::side_type>(side);
    std::string fill(fill_char.data(), fill_char.size());
    auto result = cudf::strings::pad(col.view(), width, side_type, fill, stream, mr);
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> str_zfill(
    const OwnedColumn& col, int32_t width)
{
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    auto result = cudf::strings::zfill(col.view(), width, stream, mr);
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> str_zfill_by_widths(
    const OwnedColumn& col, const OwnedColumn& widths)
{
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    auto result = cudf::strings::zfill_by_widths(col.view(), widths.view(), stream, mr);
    return std::make_unique<OwnedColumn>(std::move(result));
}

} // namespace cudf_shims
