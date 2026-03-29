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

} // namespace cudf_shims
