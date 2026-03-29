#pragma once

#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/types.hpp>
#include <memory>
#include "rust/cxx.h"
#include "types_shim.h"

namespace cudf_shims {

/// Owning wrapper around std::unique_ptr<cudf::scalar>.
///
/// This struct is exposed as an opaque type to Rust via cxx.
/// It owns GPU memory and frees it on destruction.
struct OwnedScalar {
    std::unique_ptr<cudf::scalar> inner;

    explicit OwnedScalar(std::unique_ptr<cudf::scalar> s)
        : inner(std::move(s)) {}

    /// Type ID of the scalar's element type.
    int32_t type_id() const { return static_cast<int32_t>(inner->type().id()); }

    /// Whether this scalar holds a valid (non-null) value.
    bool is_valid() const {
        return inner->is_valid(cudf::get_default_stream());
    }
};

// ── Construction ───────────────────────────────────────────────

/// Create a numeric scalar of the given type. Initially invalid (null).
std::unique_ptr<OwnedScalar> make_numeric_scalar(int32_t type_id);

// ── Setters ────────────────────────────────────────────────────

void scalar_set_i32(OwnedScalar& s, int32_t value);
void scalar_set_i64(OwnedScalar& s, int64_t value);
void scalar_set_f32(OwnedScalar& s, float value);
void scalar_set_f64(OwnedScalar& s, double value);
void scalar_set_i8(OwnedScalar& s, int8_t value);
void scalar_set_i16(OwnedScalar& s, int16_t value);
void scalar_set_u8(OwnedScalar& s, uint8_t value);
void scalar_set_u16(OwnedScalar& s, uint16_t value);
void scalar_set_u32(OwnedScalar& s, uint32_t value);
void scalar_set_u64(OwnedScalar& s, uint64_t value);
void scalar_set_bool(OwnedScalar& s, bool value);
void scalar_set_valid(OwnedScalar& s, bool valid);

// ── Getters ────────────────────────────────────────────────────

int32_t scalar_get_i32(const OwnedScalar& s);
int64_t scalar_get_i64(const OwnedScalar& s);
float scalar_get_f32(const OwnedScalar& s);
double scalar_get_f64(const OwnedScalar& s);
int8_t scalar_get_i8(const OwnedScalar& s);
int16_t scalar_get_i16(const OwnedScalar& s);
uint8_t scalar_get_u8(const OwnedScalar& s);
uint16_t scalar_get_u16(const OwnedScalar& s);
uint32_t scalar_get_u32(const OwnedScalar& s);
uint64_t scalar_get_u64(const OwnedScalar& s);
bool scalar_get_bool(const OwnedScalar& s);

} // namespace cudf_shims
