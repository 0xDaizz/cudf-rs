#include "timezone_shim.h"
#include <cudf/timezone.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <string>
#include <optional>

namespace cudf_shims {

std::unique_ptr<OwnedTable> make_timezone_transition_table(
    rust::Str timezone_name)
{
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    std::string tz(timezone_name.data(), timezone_name.size());
    auto result = cudf::make_timezone_transition_table(
        std::nullopt, tz, stream, mr);
    return std::make_unique<OwnedTable>(std::move(result));
}

} // namespace cudf_shims
