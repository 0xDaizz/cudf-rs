#include "types_shim.h"
#include "enum_verify.h"

namespace cudf_shims {

std::unique_ptr<DataType> make_data_type(int32_t id) {
    return std::make_unique<DataType>(static_cast<cudf::type_id>(id));
}

std::unique_ptr<DataType> make_data_type_with_scale(int32_t id, int32_t scale) {
    return std::make_unique<DataType>(static_cast<cudf::type_id>(id), scale);
}

int32_t data_type_id(const DataType& dt) {
    return static_cast<int32_t>(dt.inner.id());
}

int32_t data_type_scale(const DataType& dt) {
    return dt.inner.scale();
}

} // namespace cudf_shims
