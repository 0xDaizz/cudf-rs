#include "datetime_shim.h"
#include <cudf/datetime.hpp>
#include <cudf/utilities/default_stream.hpp>

namespace cudf_shims {

std::unique_ptr<OwnedColumn> extract_year(const OwnedColumn& col) {
    auto result = cudf::datetime::extract_datetime_component(
        col.view(), cudf::datetime::datetime_component::YEAR);
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> extract_month(const OwnedColumn& col) {
    auto result = cudf::datetime::extract_datetime_component(
        col.view(), cudf::datetime::datetime_component::MONTH);
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> extract_day(const OwnedColumn& col) {
    auto result = cudf::datetime::extract_datetime_component(
        col.view(), cudf::datetime::datetime_component::DAY);
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> extract_hour(const OwnedColumn& col) {
    auto result = cudf::datetime::extract_datetime_component(
        col.view(), cudf::datetime::datetime_component::HOUR);
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> extract_minute(const OwnedColumn& col) {
    auto result = cudf::datetime::extract_datetime_component(
        col.view(), cudf::datetime::datetime_component::MINUTE);
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> extract_second(const OwnedColumn& col) {
    auto result = cudf::datetime::extract_datetime_component(
        col.view(), cudf::datetime::datetime_component::SECOND);
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> extract_weekday(const OwnedColumn& col) {
    auto result = cudf::datetime::extract_datetime_component(
        col.view(), cudf::datetime::datetime_component::WEEKDAY);
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> extract_day_of_year(const OwnedColumn& col) {
    auto result = cudf::datetime::day_of_year(col.view());
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> last_day_of_month(const OwnedColumn& col) {
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    auto result = cudf::datetime::last_day_of_month(col.view(), stream, mr);
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> add_calendrical_months_scalar(
    const OwnedColumn& col, const OwnedScalar& months)
{
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    auto result = cudf::datetime::add_calendrical_months(
        col.view(), *months.inner, stream, mr);
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> add_calendrical_months_column(
    const OwnedColumn& col, const OwnedColumn& months)
{
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    auto result = cudf::datetime::add_calendrical_months(
        col.view(), months.view(), stream, mr);
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> is_leap_year(const OwnedColumn& col) {
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    auto result = cudf::datetime::is_leap_year(col.view(), stream, mr);
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> days_in_month(const OwnedColumn& col) {
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    auto result = cudf::datetime::days_in_month(col.view(), stream, mr);
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> extract_quarter(const OwnedColumn& col) {
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    auto result = cudf::datetime::extract_quarter(col.view(), stream, mr);
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> ceil_datetimes(const OwnedColumn& col, int32_t freq) {
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    auto f = static_cast<cudf::datetime::rounding_frequency>(freq);
    auto result = cudf::datetime::ceil_datetimes(col.view(), f, stream, mr);
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> floor_datetimes(const OwnedColumn& col, int32_t freq) {
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    auto f = static_cast<cudf::datetime::rounding_frequency>(freq);
    auto result = cudf::datetime::floor_datetimes(col.view(), f, stream, mr);
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> round_datetimes(const OwnedColumn& col, int32_t freq) {
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    auto f = static_cast<cudf::datetime::rounding_frequency>(freq);
    auto result = cudf::datetime::round_datetimes(col.view(), f, stream, mr);
    return std::make_unique<OwnedColumn>(std::move(result));
}

} // namespace cudf_shims
