#include "strings/convert_shim.h"
#include <cudf/strings/convert/convert_integers.hpp>
#include <cudf/strings/convert/convert_floats.hpp>
#include <cudf/strings/convert/convert_booleans.hpp>
#include <cudf/strings/convert/convert_datetime.hpp>
#include <cudf/strings/convert/convert_durations.hpp>
#include <cudf/strings/convert/convert_fixed_point.hpp>
#include <cudf/strings/convert/convert_ipv4.hpp>
#include <cudf/strings/convert/convert_urls.hpp>
#include <cudf/strings/convert/int_cast.hpp>
#include <cudf/strings/convert/convert_lists.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <string>

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

// ── Booleans ──────────────────────────────────────────────────

std::unique_ptr<OwnedColumn> str_to_booleans(
    const OwnedColumn& col, rust::Str true_str)
{
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    std::string ts(true_str.data(), true_str.size());
    cudf::string_scalar scalar_true(ts, true, stream);
    auto result = cudf::strings::to_booleans(col.view(), scalar_true, stream, mr);
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> str_from_booleans(
    const OwnedColumn& col, rust::Str true_str, rust::Str false_str)
{
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    std::string ts(true_str.data(), true_str.size());
    std::string fs(false_str.data(), false_str.size());
    cudf::string_scalar scalar_true(ts, true, stream);
    cudf::string_scalar scalar_false(fs, true, stream);
    auto result = cudf::strings::from_booleans(col.view(), scalar_true, scalar_false, stream, mr);
    return std::make_unique<OwnedColumn>(std::move(result));
}

// ── Timestamps ────────────────────────────────────────────────

std::unique_ptr<OwnedColumn> str_to_timestamps(
    const OwnedColumn& col, rust::Str format, int32_t type_id)
{
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    auto tid = static_cast<cudf::type_id>(type_id);
    std::string fmt(format.data(), format.size());
    auto result = cudf::strings::to_timestamps(
        col.view(), cudf::data_type{tid}, fmt, stream, mr);
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> str_from_timestamps(
    const OwnedColumn& col, rust::Str format)
{
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    std::string fmt(format.data(), format.size());
    auto result = cudf::strings::from_timestamps(col.view(), fmt, cudf::strings_column_view(cudf::column_view{cudf::data_type{cudf::type_id::STRING}, 0, nullptr, nullptr, 0}), stream, mr);
    return std::make_unique<OwnedColumn>(std::move(result));
}

// ── Durations ─────────────────────────────────────────────────

std::unique_ptr<OwnedColumn> str_to_durations(
    const OwnedColumn& col, rust::Str format, int32_t type_id)
{
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    auto tid = static_cast<cudf::type_id>(type_id);
    std::string fmt(format.data(), format.size());
    auto result = cudf::strings::to_durations(
        col.view(), cudf::data_type{tid}, fmt, stream, mr);
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> str_from_durations(
    const OwnedColumn& col, rust::Str format)
{
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    std::string fmt(format.data(), format.size());
    auto result = cudf::strings::from_durations(col.view(), fmt, stream, mr);
    return std::make_unique<OwnedColumn>(std::move(result));
}

// ── Fixed Point ───────────────────────────────────────────────

std::unique_ptr<OwnedColumn> str_to_fixed_point(
    const OwnedColumn& col, int32_t type_id, int32_t scale)
{
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    auto tid = static_cast<cudf::type_id>(type_id);
    auto result = cudf::strings::to_fixed_point(
        col.view(), cudf::data_type{tid, scale}, stream, mr);
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> str_from_fixed_point(const OwnedColumn& col) {
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    auto result = cudf::strings::from_fixed_point(col.view(), stream, mr);
    return std::make_unique<OwnedColumn>(std::move(result));
}

// ── Type Checks ───────────────────────────────────────────────

std::unique_ptr<OwnedColumn> str_is_integer(const OwnedColumn& col) {
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    auto result = cudf::strings::is_integer(col.view(), stream, mr);
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> str_is_float(const OwnedColumn& col) {
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    auto result = cudf::strings::is_float(col.view(), stream, mr);
    return std::make_unique<OwnedColumn>(std::move(result));
}

// ── Hex ───────────────────────────────────────────────────────

std::unique_ptr<OwnedColumn> str_hex_to_integers(
    const OwnedColumn& col, int32_t type_id)
{
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    auto tid = static_cast<cudf::type_id>(type_id);
    auto result = cudf::strings::hex_to_integers(
        col.view(), cudf::data_type{tid}, stream, mr);
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> str_integers_to_hex(const OwnedColumn& col) {
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    auto result = cudf::strings::integers_to_hex(col.view(), stream, mr);
    return std::make_unique<OwnedColumn>(std::move(result));
}

// ── IPv4 ──────────────────────────────────────────────────────

std::unique_ptr<OwnedColumn> str_ipv4_to_integers(const OwnedColumn& col) {
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    auto result = cudf::strings::ipv4_to_integers(col.view(), stream, mr);
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> str_integers_to_ipv4(const OwnedColumn& col) {
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    auto result = cudf::strings::integers_to_ipv4(col.view(), stream, mr);
    return std::make_unique<OwnedColumn>(std::move(result));
}

// ── URL Encoding ──────────────────────────────────────────────

std::unique_ptr<OwnedColumn> str_url_encode(const OwnedColumn& col) {
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    auto result = cudf::strings::url_encode(col.view(), stream, mr);
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> str_url_decode(const OwnedColumn& col) {
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    auto result = cudf::strings::url_decode(col.view(), stream, mr);
    return std::make_unique<OwnedColumn>(std::move(result));
}

// ── Fixed-point check ────────────────────────────────────────

std::unique_ptr<OwnedColumn> str_is_fixed_point(
    const OwnedColumn& col, int32_t type_id)
{
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    auto tid = static_cast<cudf::type_id>(type_id);
    auto result = cudf::strings::is_fixed_point(
        col.view(), cudf::data_type{tid}, stream, mr);
    return std::make_unique<OwnedColumn>(std::move(result));
}

// ── Integer cast (encode/decode strings as integers) ─────────

std::unique_ptr<OwnedColumn> str_cast_to_integer(
    const OwnedColumn& col, int32_t type_id)
{
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    auto tid = static_cast<cudf::type_id>(type_id);
    auto result = cudf::strings::cast_to_integer(
        col.view(), cudf::data_type{tid}, cudf::strings::endian::LITTLE, stream, mr);
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> str_cast_from_integer(const OwnedColumn& col)
{
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    auto result = cudf::strings::cast_from_integer(
        col.view(), cudf::strings::endian::LITTLE, stream, mr);
    return std::make_unique<OwnedColumn>(std::move(result));
}

// ── Format list column ───────────────────────────────────────

std::unique_ptr<OwnedColumn> str_format_list_column(const OwnedColumn& col)
{
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    cudf::string_scalar na_rep("", true, stream);
    auto result = cudf::strings::format_list_column(
        cudf::lists_column_view(col.view()), na_rep,
        cudf::strings_column_view(cudf::column_view{
            cudf::data_type{cudf::type_id::STRING}, 0, nullptr, nullptr, 0}),
        stream, mr);
    return std::make_unique<OwnedColumn>(std::move(result));
}

} // namespace cudf_shims
