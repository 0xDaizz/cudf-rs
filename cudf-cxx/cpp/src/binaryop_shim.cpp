#include "binaryop_shim.h"
#include <cudf/binaryop.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <stdexcept>

namespace cudf_shims {

std::unique_ptr<OwnedColumn> binary_operation_col_col(
    const OwnedColumn& lhs,
    const OwnedColumn& rhs,
    int32_t op,
    int32_t output_type)
{
    auto cudf_op = static_cast<cudf::binary_operator>(op);
    auto out_tid = static_cast<cudf::type_id>(output_type);
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();

    auto result = cudf::binary_operation(
        lhs.view(), rhs.view(),
        cudf_op, cudf::data_type{out_tid},
        stream, mr);
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> binary_operation_col_scalar(
    const OwnedColumn& lhs,
    const OwnedScalar& rhs,
    int32_t op,
    int32_t output_type)
{
    auto cudf_op = static_cast<cudf::binary_operator>(op);
    auto out_tid = static_cast<cudf::type_id>(output_type);
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();

    auto result = cudf::binary_operation(
        lhs.view(), *rhs.inner,
        cudf_op, cudf::data_type{out_tid},
        stream, mr);
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> binary_operation_scalar_col(
    const OwnedScalar& lhs,
    const OwnedColumn& rhs,
    int32_t op,
    int32_t output_type)
{
    auto cudf_op = static_cast<cudf::binary_operator>(op);
    auto out_tid = static_cast<cudf::type_id>(output_type);
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();

    auto result = cudf::binary_operation(
        *lhs.inner, rhs.view(),
        cudf_op, cudf::data_type{out_tid},
        stream, mr);
    return std::make_unique<OwnedColumn>(std::move(result));
}

} // namespace cudf_shims
