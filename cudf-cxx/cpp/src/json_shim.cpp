#include "json_shim.h"
#include <cudf/json/json.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <string>

namespace cudf_shims {

std::unique_ptr<OwnedColumn> get_json_object(
    const OwnedColumn& col,
    rust::Str json_path,
    bool allow_single_quotes,
    bool strip_quotes,
    bool missing_fields_as_nulls)
{
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    std::string path(json_path.data(), json_path.size());
    cudf::string_scalar scalar_path(path, true, stream);

    cudf::get_json_object_options options;
    options.set_allow_single_quotes(allow_single_quotes);
    options.set_strip_quotes_from_single_strings(strip_quotes);
    options.set_missing_fields_as_nulls(missing_fields_as_nulls);

    auto result = cudf::get_json_object(
        col.view(), scalar_path, options, stream, mr);
    return std::make_unique<OwnedColumn>(std::move(result));
}

} // namespace cudf_shims
