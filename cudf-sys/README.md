# cudf-sys

Native build crate for linking against NVIDIA's [libcudf](https://github.com/rapidsai/cudf).

This crate contains no Rust code — it exists solely to locate and link `libcudf.so` via its build script. Downstream crates (`cudf-cxx`, `cudf`) depend on this to inherit correct linker flags.

## Prerequisites

libcudf must be installed on your system. Supported discovery methods (in priority order):

### 1. `CUDF_ROOT` environment variable

```sh
export CUDF_ROOT=/path/to/libcudf/prefix
# expects: $CUDF_ROOT/lib/libcudf.so and $CUDF_ROOT/include/cudf/
```

### 2. Conda (recommended)

```sh
conda install -c rapidsai -c conda-forge libcudf cuda-version=12.2
# CONDA_PREFIX is automatically set when the environment is active
```

### 3. pkg-config

If libcudf installs a `.pc` file, `pkg-config` will find it automatically.

## CUDA Runtime

The CUDA runtime (`libcudart.so`) is also required. Set `CUDA_PATH` if it's not in a standard location:

```sh
export CUDA_PATH=/usr/local/cuda
```

## System Requirements

- CUDA 12.2+
- GPU: NVIDIA Volta (compute capability 7.0) or newer
- Linux (libcudf does not support macOS or Windows)
