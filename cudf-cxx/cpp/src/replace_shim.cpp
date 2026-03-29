#include "replace_shim.h"
#include <cudf/replace.hpp>
#include <cudf/utilities/default_stream.hpp>

namespace cudf_shims {

std::unique_ptr<OwnedColumn> replace_nulls_column(
    const OwnedColumn& col,
    const OwnedColumn& replacement)
{
    auto result = cudf::replace_nulls(col.view(), replacement.view());
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> replace_nulls_scalar(
    const OwnedColumn& col,
    const OwnedScalar& replacement)
{
    auto result = cudf::replace_nulls(col.view(), *replacement.inner);
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> replace_nans_scalar(
    const OwnedColumn& col,
    const OwnedScalar& replacement)
{
    auto result = cudf::replace_nans(col.view(), *replacement.inner);
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> replace_nans_column(
    const OwnedColumn& col,
    const OwnedColumn& replacement)
{
    auto result = cudf::replace_nans(col.view(), replacement.view());
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> clamp(
    const OwnedColumn& col,
    const OwnedScalar& lo,
    const OwnedScalar& hi)
{
    auto result = cudf::clamp(col.view(), *lo.inner, *hi.inner);
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> normalize_nans_and_zeros(
    const OwnedColumn& col)
{
    auto result = cudf::normalize_nans_and_zeros(col.view());
    return std::make_unique<OwnedColumn>(std::move(result));
}

} // namespace cudf_shims
