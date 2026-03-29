#include "filling_shim.h"
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <stdexcept>

namespace cudf_shims {

void fill_in_place(
    OwnedColumn& destination,
    int32_t begin,
    int32_t end,
    const OwnedScalar& value)
{
    auto mut_view = destination.mutable_view();
    cudf::fill_in_place(mut_view, begin, end, *value.inner);
}

std::unique_ptr<OwnedColumn> fill(
    const OwnedColumn& input,
    int32_t begin,
    int32_t end,
    const OwnedScalar& value)
{
    auto result = cudf::fill(input.view(), begin, end, *value.inner);
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedTable> repeat_table(
    const OwnedTable& table,
    int32_t count)
{
    auto result = cudf::repeat(table.view(), count);
    return std::make_unique<OwnedTable>(std::move(result));
}

std::unique_ptr<OwnedTable> repeat_table_variable(
    const OwnedTable& table,
    const OwnedColumn& counts)
{
    auto result = cudf::repeat(table.view(), counts.view());
    return std::make_unique<OwnedTable>(std::move(result));
}

namespace {

template <typename T>
std::unique_ptr<OwnedColumn> make_sequence(
    int32_t size, T init_val, T step_val, cudf::type_id tid)
{
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();

    auto init = cudf::make_numeric_scalar(cudf::data_type{tid}, stream, mr);
    auto step = cudf::make_numeric_scalar(cudf::data_type{tid}, stream, mr);

    using ScalarType = cudf::numeric_scalar<T>;
    static_cast<ScalarType*>(init.get())->set_value(init_val, stream);
    static_cast<ScalarType*>(step.get())->set_value(step_val, stream);
    init->set_valid_async(true, stream);
    step->set_valid_async(true, stream);

    auto result = cudf::sequence(size, *init, *step);
    return std::make_unique<OwnedColumn>(std::move(result));
}

} // anonymous namespace

std::unique_ptr<OwnedColumn> sequence_i32(
    int32_t size, int32_t init, int32_t step)
{
    return make_sequence<int32_t>(size, init, step, cudf::type_id::INT32);
}

std::unique_ptr<OwnedColumn> sequence_i64(
    int32_t size, int64_t init, int64_t step)
{
    return make_sequence<int64_t>(size, init, step, cudf::type_id::INT64);
}

std::unique_ptr<OwnedColumn> sequence_f32(
    int32_t size, float init, float step)
{
    return make_sequence<float>(size, init, step, cudf::type_id::FLOAT32);
}

std::unique_ptr<OwnedColumn> sequence_f64(
    int32_t size, double init, double step)
{
    return make_sequence<double>(size, init, step, cudf::type_id::FLOAT64);
}

} // namespace cudf_shims
