#pragma once

#include <cudf/datetime.hpp>
#include <cudf/column/column.hpp>
#include <memory>
#include "rust/cxx.h"
#include "column_shim.h"

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

} // namespace cudf_shims
