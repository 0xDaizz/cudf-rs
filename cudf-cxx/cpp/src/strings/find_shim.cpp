#include "strings/find_shim.h"
#include <cudf/strings/find.hpp>
#include <cudf/strings/find_multiple.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <string>

namespace cudf_shims {

std::unique_ptr<OwnedColumn> str_find(
    const OwnedColumn& col, rust::Str target, int32_t start)
{
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    std::string tgt(target.data(), target.size());
    cudf::string_scalar scalar_target(tgt, true, stream);
    auto result = cudf::strings::find(col.view(), scalar_target, start, -1, stream, mr);
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> str_rfind(
    const OwnedColumn& col, rust::Str target)
{
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    std::string tgt(target.data(), target.size());
    cudf::string_scalar scalar_target(tgt, true, stream);
    auto result = cudf::strings::rfind(col.view(), scalar_target, 0, -1, stream, mr);
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> str_starts_with(
    const OwnedColumn& col, rust::Str target)
{
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    std::string tgt(target.data(), target.size());
    cudf::string_scalar scalar_target(tgt, true, stream);
    auto result = cudf::strings::starts_with(col.view(), scalar_target, stream, mr);
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> str_ends_with(
    const OwnedColumn& col, rust::Str target)
{
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    std::string tgt(target.data(), target.size());
    cudf::string_scalar scalar_target(tgt, true, stream);
    auto result = cudf::strings::ends_with(col.view(), scalar_target, stream, mr);
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedTable> str_contains_multiple(
    const OwnedColumn& col, const OwnedColumn& targets)
{
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    auto result = cudf::strings::contains_multiple(
        col.view(), targets.view(), stream, mr);
    return std::make_unique<OwnedTable>(std::move(result));
}

std::unique_ptr<OwnedColumn> str_find_multiple(
    const OwnedColumn& col, const OwnedColumn& targets)
{
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    auto result = cudf::strings::find_multiple(
        col.view(), targets.view(), stream, mr);
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> str_find_column(
    const OwnedColumn& col, const OwnedColumn& targets, int32_t start)
{
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    auto result = cudf::strings::find(
        cudf::strings_column_view(col.view()),
        cudf::strings_column_view(targets.view()),
        start, stream, mr);
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> str_find_instance(
    const OwnedColumn& col, rust::Str target, int32_t instance)
{
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    std::string tgt(target.data(), target.size());
    cudf::string_scalar scalar_target(tgt, true, stream);
    auto result = cudf::strings::find_instance(
        cudf::strings_column_view(col.view()),
        scalar_target, instance, stream, mr);
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> str_contains_column(
    const OwnedColumn& col, const OwnedColumn& targets)
{
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    auto result = cudf::strings::contains(
        cudf::strings_column_view(col.view()),
        cudf::strings_column_view(targets.view()),
        stream, mr);
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> str_starts_with_column(
    const OwnedColumn& col, const OwnedColumn& targets)
{
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    auto result = cudf::strings::starts_with(
        cudf::strings_column_view(col.view()),
        cudf::strings_column_view(targets.view()),
        stream, mr);
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> str_ends_with_column(
    const OwnedColumn& col, const OwnedColumn& targets)
{
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    auto result = cudf::strings::ends_with(
        cudf::strings_column_view(col.view()),
        cudf::strings_column_view(targets.view()),
        stream, mr);
    return std::make_unique<OwnedColumn>(std::move(result));
}

} // namespace cudf_shims
