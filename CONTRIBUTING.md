# Contributing to cudf-rs

Thank you for your interest in contributing to cudf-rs! This guide covers the process for adding new bindings, code conventions, and testing requirements.

## How to Add a New Binding

Adding a new libcudf binding touches all three crates. Follow these steps:

### Step 1: C++ Shim (cudf-cxx/cpp/)

Create the shim header and implementation:

```
cudf-cxx/cpp/include/new_module_shim.h
cudf-cxx/cpp/src/new_module_shim.cpp
```

**Conventions:**
- Namespace: `cudf_shims`
- Owning wrapper: `OwnedX` for types that own GPU resources (e.g., `OwnedColumn`, `OwnedTable`)
- All parameters must be cxx-compatible (primitives, `rust::Str`, `rust::Slice<T>`, `std::unique_ptr<OwnedX>`, etc.)
- Do NOT catch exceptions -- let cxx handle them via `-> Result<T>` on the bridge side
- Use `cudf::get_default_stream()` and `cudf::get_current_device_resource_ref()` for stream and memory resource
- Construct any options/config structs inside the shim from flat parameters (do NOT pass them across FFI)

```cpp
// cudf-cxx/cpp/include/new_module_shim.h
#pragma once
#include <memory>
#include "rust/cxx.h"
#include "column_shim.h"

namespace cudf_shims {

std::unique_ptr<OwnedColumn> new_module_operation(
    const OwnedColumn& input,
    int32_t some_param);

}  // namespace cudf_shims
```

### Step 2: cxx Bridge (cudf-cxx/src/)

Create the Rust bridge module:

```rust
// cudf-cxx/src/new_module.rs
#[cxx::bridge(namespace = "cudf_shims")]
pub mod ffi {
    unsafe extern "C++" {
        include!("cudf-cxx/cpp/include/new_module_shim.h");

        type OwnedColumn = crate::column::ffi::OwnedColumn;

        fn new_module_operation(
            input: &OwnedColumn,
            some_param: i32,
        ) -> Result<UniquePtr<OwnedColumn>>;
    }
}
```

**Then register the bridge:**

1. Add `pub mod new_module;` to `cudf-cxx/src/lib.rs`
2. Add `"src/new_module.rs"` to the `cxx_build::bridges()` call in `cudf-cxx/build.rs`
3. Add the `.cpp` file to the build sources list in `build.rs`

### Step 3: Safe Wrapper (cudf/src/)

Create the safe Rust API:

```rust
// cudf/src/new_module.rs
//! New module description.
//!
//! # Examples
//!
//! ```rust,no_run
//! use cudf::Column;
//! // example usage
//! ```

use crate::column::Column;
use crate::error::{CudfError, Result};

impl Column {
    /// Documentation for the operation.
    pub fn new_module_op(&self, some_param: i32) -> Result<Column> {
        let result = cudf_cxx::new_module::ffi::new_module_operation(
            &self.inner, some_param,
        ).map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }
}
```

**Then register the module:**

1. Add `pub mod new_module;` to `cudf/src/lib.rs`
2. Add re-exports to `cudf/src/lib.rs` if the module introduces public types

### Step 4: Tests

Add tests in the module file or in a dedicated test:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_module_op() {
        let col = Column::from_slice(&[1i32, 2, 3]).unwrap();
        let result = col.new_module_op(42).unwrap();
        assert_eq!(result.len(), 3);
    }
}
```

## Code Conventions

### Naming

| Entity | Convention | Example |
|--------|-----------|---------|
| C++ shim file | `{module}_shim.h/cpp` | `sorting_shim.h` |
| C++ namespace | `cudf_shims` | always |
| Owning C++ type | `Owned{Type}` | `OwnedColumn` |
| Rust bridge module | `{module}::ffi` | `sorting::ffi` |
| Safe Rust method | snake_case, on `Column`/`Table` | `col.reduce()` |
| Enum variant | PascalCase | `SortOrder::Ascending` |

### cxx Rules

- No templates in bridge declarations -- instantiate in C++ shim
- No default parameters -- make them explicit on the Rust side
- Return `Result<T>` for any function that can throw
- Use `UniquePtr<OwnedX>` for owning returns
- Use `&OwnedX` for borrowed inputs
- String parameters: `&str` on Rust side maps to `rust::Str` on C++ side

### Error Handling

- All public functions return `Result<T>` using `CudfError`
- Convert cxx exceptions with `CudfError::from_cxx(e)`
- Use `CudfError::InvalidArgument` for Rust-side validation
- Do NOT `unwrap()` in library code

### Documentation

- Every public type and function needs a doc comment
- Include `# Examples` with `rust,no_run` code blocks
- Module-level doc comments describe the module purpose

## Testing Requirements

**A GPU is required to run tests.** All cudf operations execute on the GPU.

```sh
# Run all tests
cargo test

# Run tests for a specific module
cargo test --package cudf -- sorting

# Run with output
cargo test -- --nocapture
```

### What to Test

- Basic operation correctness (known input -> expected output)
- Edge cases: empty columns, single-element columns, all-null columns
- Type variations: test with i32, i64, f32, f64 where applicable
- Error cases: invalid type combinations, out-of-bounds indices

### GPU Feature Flag

Tests that require a GPU are gated behind the `gpu-tests` feature:

```sh
# cudf crate tests
cargo test -p cudf --features gpu-tests

# cudf-polars e2e tests
cargo test -p cudf-polars --features gpu-tests
```

### LD_LIBRARY_PATH Setup

When using pip-installed libcudf (not conda), set all required library paths:

```sh
export LD_LIBRARY_PATH=/path/to/pyarrow:/path/to/libcudf/lib64:/path/to/librmm/lib64:/path/to/libnvcomp/lib64:/path/to/rapids_logger/lib64:/usr/local/cuda/lib64
```

### Python Integration Tests

The Python test suite validates the Polars GPU engine (cudf-polars-cu12) end-to-end:

```sh
pip install polars cudf-polars-cu12 --extra-index-url=https://pypi.nvidia.com
python tests/polars_gpu_integration.py
```

## PR Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/new-module`
3. Implement changes following the 3-step binding process above
4. Ensure `cargo build --release` succeeds
5. Run tests: `cargo test`
6. Run `cargo clippy` and fix any warnings
7. Run `cargo fmt`
8. Submit a PR with:
   - Description of what libcudf API is being wrapped
   - Link to the relevant libcudf documentation
   - Test results
