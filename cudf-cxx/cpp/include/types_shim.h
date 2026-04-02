#pragma once

#include <cudf/types.hpp>
#include <memory>
#include <stdexcept>
#include <string>
#include "rust/cxx.h"

namespace cudf_shims {

/// Wrapper around cudf::data_type for cxx opaque type support.
struct DataType {
    cudf::data_type inner;

    DataType(cudf::type_id id) : inner(id) {}
    DataType(cudf::type_id id, int32_t scale) : inner(id, scale) {}
};

/// Maps our bridge TypeId enum values to cudf::type_id.
/// The enum values are kept in sync manually.
inline cudf::type_id to_cudf_type_id(int32_t id) {
    return static_cast<cudf::type_id>(id);
}

/// Validates that an integer is within the valid cudf::type_id range before casting.
inline cudf::type_id validated_type_id(int32_t id) {
    if (id < 0 || id > 28) {  // 28 = max cudf::type_id value
        throw std::runtime_error("cudf: invalid type_id " + std::to_string(id));
    }
    return static_cast<cudf::type_id>(id);
}

std::unique_ptr<DataType> make_data_type(int32_t id);
std::unique_ptr<DataType> make_data_type_with_scale(int32_t id, int32_t scale);
int32_t data_type_id(const DataType& dt);
int32_t data_type_scale(const DataType& dt);

} // namespace cudf_shims
