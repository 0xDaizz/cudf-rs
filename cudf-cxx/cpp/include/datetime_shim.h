#pragma once

#include <cudf/datetime.hpp>
#include <cudf/column/column.hpp>
#include <memory>
#include "rust/cxx.h"
#include "column_shim.h"
#include "scalar_shim.h"

namespace cudf_shims {

/// Extract year component from a timestamp column.
std::unique_ptr<OwnedColumn> extract_year(const OwnedColumn& col);

/// Extract month component from a timestamp column.
std::unique_ptr<OwnedColumn> extract_month(const OwnedColumn& col);

/// Extract day component from a timestamp column.
std::unique_ptr<OwnedColumn> extract_day(const OwnedColumn& col);

/// Extract hour component from a timestamp column.
std::unique_ptr<OwnedColumn> extract_hour(const OwnedColumn& col);

/// Extract minute component from a timestamp column.
std::unique_ptr<OwnedColumn> extract_minute(const OwnedColumn& col);

/// Extract second component from a timestamp column.
std::unique_ptr<OwnedColumn> extract_second(const OwnedColumn& col);

/// Extract weekday from a timestamp column.
std::unique_ptr<OwnedColumn> extract_weekday(const OwnedColumn& col);

/// Extract day-of-year from a timestamp column.
std::unique_ptr<OwnedColumn> extract_day_of_year(const OwnedColumn& col);

/// Get the last day of the month for each timestamp.
std::unique_ptr<OwnedColumn> last_day_of_month(const OwnedColumn& col);

/// Add calendrical months (scalar).
std::unique_ptr<OwnedColumn> add_calendrical_months_scalar(
    const OwnedColumn& col, const OwnedScalar& months);

/// Add calendrical months (column).
std::unique_ptr<OwnedColumn> add_calendrical_months_column(
    const OwnedColumn& col, const OwnedColumn& months);

/// Check if year is a leap year.
std::unique_ptr<OwnedColumn> is_leap_year(const OwnedColumn& col);

/// Get days in month for each timestamp.
std::unique_ptr<OwnedColumn> days_in_month(const OwnedColumn& col);

/// Extract quarter from timestamp.
std::unique_ptr<OwnedColumn> extract_quarter(const OwnedColumn& col);

/// Ceil datetimes to frequency.
std::unique_ptr<OwnedColumn> ceil_datetimes(const OwnedColumn& col, int32_t freq);

/// Floor datetimes to frequency.
std::unique_ptr<OwnedColumn> floor_datetimes(const OwnedColumn& col, int32_t freq);

/// Round datetimes to frequency.
std::unique_ptr<OwnedColumn> round_datetimes(const OwnedColumn& col, int32_t freq);

} // namespace cudf_shims
