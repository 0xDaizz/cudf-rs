#include "unary_shim.h"
#include <cudf/unary.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <limits>
#include <cudf/utilities/default_stream.hpp>
#include <stdexcept>

namespace cudf_shims {

std::unique_ptr<OwnedColumn> unary_operation(
    const OwnedColumn& input, int32_t op)
{
    auto cudf_op = static_cast<cudf::unary_operator>(op);
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();

    auto result = cudf::unary_operation(input.view(), cudf_op, stream, mr);
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> is_null(const OwnedColumn& input) {
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();

    auto result = cudf::is_null(input.view(), stream, mr);
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> is_valid(const OwnedColumn& input) {
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();

    auto result = cudf::is_valid(input.view(), stream, mr);
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> is_nan(const OwnedColumn& input) {
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();

    auto result = cudf::is_nan(input.view(), stream, mr);
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> is_not_nan(const OwnedColumn& input) {
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();

    auto result = cudf::is_not_nan(input.view(), stream, mr);
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> cast(const OwnedColumn& input, int32_t type_id) {
    auto tid = static_cast<cudf::type_id>(type_id);
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();

    auto result = cudf::cast(input.view(), cudf::data_type{tid}, stream, mr);
    return std::make_unique<OwnedColumn>(std::move(result));
}

bool is_supported_cast(int32_t from_type_id, int32_t to_type_id) {
    auto from = cudf::data_type{static_cast<cudf::type_id>(from_type_id)};
    auto to = cudf::data_type{static_cast<cudf::type_id>(to_type_id)};
    return cudf::is_supported_cast(from, to);
}


std::unique_ptr<OwnedColumn> is_inf(const OwnedColumn& input) {
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();

    // Compute abs(x)
    auto abs_col = cudf::unary_operation(input.view(), cudf::unary_operator::ABS, stream, mr);
    // Create a scalar with INFINITY
    auto inf_scalar = cudf::make_fixed_width_scalar<double>(std::numeric_limits<double>::infinity(), stream, mr);
    // Compare: abs(x) == inf
    auto result = cudf::binary_operation(
        abs_col->view(),
        *inf_scalar,
        cudf::binary_operator::EQUAL,
        cudf::data_type{cudf::type_id::BOOL8},
        stream, mr);
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> is_not_inf(const OwnedColumn& input) {
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();

    // Compute is_inf, then NOT
    auto inf_col = is_inf(input);
    auto result = cudf::unary_operation(inf_col->view(), cudf::unary_operator::NOT, stream, mr);
    return std::make_unique<OwnedColumn>(std::move(result));
}

} // namespace cudf_shims
