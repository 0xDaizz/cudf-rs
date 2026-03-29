#include "scalar_shim.h"
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <stdexcept>

namespace cudf_shims {

// ── Construction ──────────────────────────────────────────────

std::unique_ptr<OwnedScalar> make_numeric_scalar(int32_t type_id) {
    auto tid = static_cast<cudf::type_id>(type_id);
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();

    auto s = cudf::make_numeric_scalar(cudf::data_type{tid}, stream, mr);
    // Initially invalid (null)
    s->set_valid_async(false, stream);
    return std::make_unique<OwnedScalar>(std::move(s));
}

// ── Setters ───────────────────────────────────────────────────

void scalar_set_i32(OwnedScalar& s, int32_t value) {
    auto* typed = dynamic_cast<cudf::numeric_scalar<int32_t>*>(s.inner.get());
    if (!typed) throw std::runtime_error("scalar is not INT32");
    typed->set_value(value, cudf::get_default_stream());
    typed->set_valid_async(true, cudf::get_default_stream());
}

void scalar_set_i64(OwnedScalar& s, int64_t value) {
    auto* typed = dynamic_cast<cudf::numeric_scalar<int64_t>*>(s.inner.get());
    if (!typed) throw std::runtime_error("scalar is not INT64");
    typed->set_value(value, cudf::get_default_stream());
    typed->set_valid_async(true, cudf::get_default_stream());
}

void scalar_set_f32(OwnedScalar& s, float value) {
    auto* typed = dynamic_cast<cudf::numeric_scalar<float>*>(s.inner.get());
    if (!typed) throw std::runtime_error("scalar is not FLOAT32");
    typed->set_value(value, cudf::get_default_stream());
    typed->set_valid_async(true, cudf::get_default_stream());
}

void scalar_set_f64(OwnedScalar& s, double value) {
    auto* typed = dynamic_cast<cudf::numeric_scalar<double>*>(s.inner.get());
    if (!typed) throw std::runtime_error("scalar is not FLOAT64");
    typed->set_value(value, cudf::get_default_stream());
    typed->set_valid_async(true, cudf::get_default_stream());
}

void scalar_set_i8(OwnedScalar& s, int8_t value) {
    auto* typed = dynamic_cast<cudf::numeric_scalar<int8_t>*>(s.inner.get());
    if (!typed) throw std::runtime_error("scalar is not INT8");
    typed->set_value(value, cudf::get_default_stream());
    typed->set_valid_async(true, cudf::get_default_stream());
}

void scalar_set_i16(OwnedScalar& s, int16_t value) {
    auto* typed = dynamic_cast<cudf::numeric_scalar<int16_t>*>(s.inner.get());
    if (!typed) throw std::runtime_error("scalar is not INT16");
    typed->set_value(value, cudf::get_default_stream());
    typed->set_valid_async(true, cudf::get_default_stream());
}

void scalar_set_u8(OwnedScalar& s, uint8_t value) {
    auto* typed = dynamic_cast<cudf::numeric_scalar<uint8_t>*>(s.inner.get());
    if (!typed) throw std::runtime_error("scalar is not UINT8");
    typed->set_value(value, cudf::get_default_stream());
    typed->set_valid_async(true, cudf::get_default_stream());
}

void scalar_set_u16(OwnedScalar& s, uint16_t value) {
    auto* typed = dynamic_cast<cudf::numeric_scalar<uint16_t>*>(s.inner.get());
    if (!typed) throw std::runtime_error("scalar is not UINT16");
    typed->set_value(value, cudf::get_default_stream());
    typed->set_valid_async(true, cudf::get_default_stream());
}

void scalar_set_u32(OwnedScalar& s, uint32_t value) {
    auto* typed = dynamic_cast<cudf::numeric_scalar<uint32_t>*>(s.inner.get());
    if (!typed) throw std::runtime_error("scalar is not UINT32");
    typed->set_value(value, cudf::get_default_stream());
    typed->set_valid_async(true, cudf::get_default_stream());
}

void scalar_set_u64(OwnedScalar& s, uint64_t value) {
    auto* typed = dynamic_cast<cudf::numeric_scalar<uint64_t>*>(s.inner.get());
    if (!typed) throw std::runtime_error("scalar is not UINT64");
    typed->set_value(value, cudf::get_default_stream());
    typed->set_valid_async(true, cudf::get_default_stream());
}

void scalar_set_valid(OwnedScalar& s, bool valid) {
    s.inner->set_valid_async(valid, cudf::get_default_stream());
}

// ── Getters ───────────────────────────────────────────────────

int32_t scalar_get_i32(const OwnedScalar& s) {
    auto* typed = dynamic_cast<const cudf::numeric_scalar<int32_t>*>(s.inner.get());
    if (!typed) throw std::runtime_error("scalar is not INT32");
    return typed->value(cudf::get_default_stream());
}

int64_t scalar_get_i64(const OwnedScalar& s) {
    auto* typed = dynamic_cast<const cudf::numeric_scalar<int64_t>*>(s.inner.get());
    if (!typed) throw std::runtime_error("scalar is not INT64");
    return typed->value(cudf::get_default_stream());
}

float scalar_get_f32(const OwnedScalar& s) {
    auto* typed = dynamic_cast<const cudf::numeric_scalar<float>*>(s.inner.get());
    if (!typed) throw std::runtime_error("scalar is not FLOAT32");
    return typed->value(cudf::get_default_stream());
}

double scalar_get_f64(const OwnedScalar& s) {
    auto* typed = dynamic_cast<const cudf::numeric_scalar<double>*>(s.inner.get());
    if (!typed) throw std::runtime_error("scalar is not FLOAT64");
    return typed->value(cudf::get_default_stream());
}

int8_t scalar_get_i8(const OwnedScalar& s) {
    auto* typed = dynamic_cast<const cudf::numeric_scalar<int8_t>*>(s.inner.get());
    if (!typed) throw std::runtime_error("scalar is not INT8");
    return typed->value(cudf::get_default_stream());
}

int16_t scalar_get_i16(const OwnedScalar& s) {
    auto* typed = dynamic_cast<const cudf::numeric_scalar<int16_t>*>(s.inner.get());
    if (!typed) throw std::runtime_error("scalar is not INT16");
    return typed->value(cudf::get_default_stream());
}

uint8_t scalar_get_u8(const OwnedScalar& s) {
    auto* typed = dynamic_cast<const cudf::numeric_scalar<uint8_t>*>(s.inner.get());
    if (!typed) throw std::runtime_error("scalar is not UINT8");
    return typed->value(cudf::get_default_stream());
}

uint16_t scalar_get_u16(const OwnedScalar& s) {
    auto* typed = dynamic_cast<const cudf::numeric_scalar<uint16_t>*>(s.inner.get());
    if (!typed) throw std::runtime_error("scalar is not UINT16");
    return typed->value(cudf::get_default_stream());
}

uint32_t scalar_get_u32(const OwnedScalar& s) {
    auto* typed = dynamic_cast<const cudf::numeric_scalar<uint32_t>*>(s.inner.get());
    if (!typed) throw std::runtime_error("scalar is not UINT32");
    return typed->value(cudf::get_default_stream());
}

uint64_t scalar_get_u64(const OwnedScalar& s) {
    auto* typed = dynamic_cast<const cudf::numeric_scalar<uint64_t>*>(s.inner.get());
    if (!typed) throw std::runtime_error("scalar is not UINT64");
    return typed->value(cudf::get_default_stream());
}

} // namespace cudf_shims
