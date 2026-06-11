// Batched correlated-k opacity mixing as an XLA FFI custom call (CPU).
//
// Replaces the former Fortran combine_ck_opacities kernel (pure_callback
// path) while reproducing its semantics exactly: per flattened
// (frequency, layer) cell, species are folded into the running mixture one
// at a time via the random-overlap approximation — all n_g x n_g pairwise
// opacity sums are formed, stably sorted, reweighted to a cumulative g
// coordinate, and linearly reinterpolated onto the input Gaussian g grid.
// Species whose strongest g-point falls below 1% of the dominant species'
// weakest g-point (continuum included) are skipped, as in the Fortran code.
//
// Design notes:
//   * float32 and float64 handlers (templated implementation); the per-cell
//     arithmetic is always performed in double precision, matching the
//     Fortran kernel.
//   * Parallelized over cells through the XLA intra-op thread pool
//     (ffi::ThreadPool), with several tasks per worker because the
//     data-dependent threshold skips make per-cell cost uneven.
//   * Inputs are the mass-fraction-weighted opacities in their natural
//     (n_frequencies * n_layers, n_species, n_g) row-major layout plus the
//     flattened continuum; no Fortran-order transpose is needed.

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <vector>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace {

struct PairSample {
  double opacity;
  double weight;
};

struct Scratch {
  std::vector<PairSample> samples;
  std::vector<double> current;
  std::vector<double> next_current;
  std::vector<double> species;
  std::vector<double> g_out;
  std::vector<double> k_out;

  explicit Scratch(int n_g)
      : samples(static_cast<size_t>(n_g) * n_g),
        current(n_g),
        next_current(n_g),
        species(n_g),
        g_out(static_cast<size_t>(n_g) * n_g + 1),
        k_out(static_cast<size_t>(n_g) * n_g + 1) {}
};

double InterpolateLinear(const std::vector<double>& x, const std::vector<double>& y,
                         int n, double x_out) {
  int idx = 0;
  if (x_out >= x[n - 1]) {
    idx = n - 2;
  } else if (x_out <= x[0]) {
    idx = 0;
  } else {
    int lo = 0;
    int hi = n - 1;
    while (hi - lo > 1) {
      const int mid = (hi + lo) / 2;
      if (x_out >= x[mid]) {
        lo = mid;
      } else {
        hi = mid;
      }
    }
    idx = lo;
  }

  const double dx = x[idx + 1] - x[idx];
  if (dx == 0.0) {
    return y[idx];
  }
  const double t = (x_out - x[idx]) / dx;
  return y[idx] + (y[idx + 1] - y[idx]) * t;
}

template <typename T>
void MixCells(const T* weighted_ptr, const T* continuum_ptr,
              const std::vector<double>& g_points,
              const std::vector<double>& gauss_weights, int fl_begin, int fl_end,
              int n_species, int n_g, Scratch& scratch, T* output_ptr) {
  const int n_sample = n_g * n_g;
  auto& samples = scratch.samples;
  auto& current = scratch.current;
  auto& next_current = scratch.next_current;
  auto& species = scratch.species;
  auto& g_out = scratch.g_out;
  auto& k_out = scratch.k_out;

  for (int fl = fl_begin; fl < fl_end; ++fl) {
    const T* cell_ptr = weighted_ptr + static_cast<size_t>(fl) * n_species * n_g;
    const double continuum = static_cast<double>(continuum_ptr[fl]);

    double threshold = static_cast<double>(cell_ptr[0]) + continuum;
    for (int species_idx = 1; species_idx < n_species; ++species_idx) {
      threshold = std::max(threshold, static_cast<double>(cell_ptr[species_idx * n_g]));
    }
    threshold *= 1e-2;

    for (int g_idx = 0; g_idx < n_g; ++g_idx) {
      current[g_idx] = static_cast<double>(cell_ptr[g_idx]) + continuum;
    }

    for (int species_idx = 1; species_idx < n_species; ++species_idx) {
      const T* species_ptr = cell_ptr + species_idx * n_g;
      if (static_cast<double>(species_ptr[n_g - 1]) < threshold) {
        continue;
      }

      for (int g_idx = 0; g_idx < n_g; ++g_idx) {
        species[g_idx] = static_cast<double>(species_ptr[g_idx]);
      }

      int sample_index = 0;
      double weight_sum = 0.0;
      for (int i = 0; i < n_g; ++i) {
        for (int j = 0; j < n_g; ++j) {
          const double weight = gauss_weights[i] * gauss_weights[j];
          samples[sample_index++] = PairSample{current[i] + species[j], weight};
          weight_sum += weight;
        }
      }

      std::stable_sort(samples.begin(), samples.end(),
                       [](const PairSample& lhs, const PairSample& rhs) {
                         return lhs.opacity < rhs.opacity;
                       });

      double cumulative = 0.0;
      const double inv_weight_sum = 1.0 / weight_sum;
      for (int i = 0; i < n_sample; ++i) {
        samples[i].weight *= inv_weight_sum;
        g_out[i] = samples[i].weight * 0.5 + cumulative;
        cumulative += samples[i].weight;
        k_out[i] = samples[i].opacity;
      }
      g_out[n_sample] = 1.0;
      k_out[0] = current[0] + species[0];
      k_out[n_sample] = current[n_g - 1] + species[n_g - 1];

      for (int g_idx = 0; g_idx < n_g; ++g_idx) {
        next_current[g_idx] = InterpolateLinear(g_out, k_out, n_sample + 1,
                                                g_points[g_idx]);
      }
      current.swap(next_current);
    }

    T* output_cell = output_ptr + static_cast<size_t>(fl) * n_g;
    for (int g_idx = 0; g_idx < n_g; ++g_idx) {
      output_cell[g_idx] = static_cast<T>(current[g_idx]);
    }
  }
}

template <ffi::DataType DT>
ffi::Error PrtCombineCkMixImpl(ffi::ThreadPool thread_pool,
                                 ffi::Buffer<DT> weighted_all,
                                 ffi::Buffer<DT> continuum_flat,
                                 ffi::Buffer<DT> g_gauss,
                                 ffi::Buffer<DT> weights_gauss,
                                 ffi::ResultBuffer<DT> output) {
  using T = ffi::NativeType<DT>;

  if (weighted_all.dimensions().size() != 3) {
    return ffi::Error::InvalidArgument("weighted_all must be rank-3");
  }
  if (continuum_flat.dimensions().size() != 1 || g_gauss.dimensions().size() != 1 ||
      weights_gauss.dimensions().size() != 1) {
    return ffi::Error::InvalidArgument(
        "continuum_flat, g_gauss, and weights_gauss must be rank-1");
  }
  if (output->dimensions().size() != 2) {
    return ffi::Error::InvalidArgument("output must be rank-2");
  }

  const int n_fl = static_cast<int>(weighted_all.dimensions()[0]);
  const int n_species = static_cast<int>(weighted_all.dimensions()[1]);
  const int n_g = static_cast<int>(weighted_all.dimensions()[2]);

  if (continuum_flat.dimensions()[0] != n_fl) {
    return ffi::Error::InvalidArgument("continuum_flat length does not match n_fl");
  }
  if (g_gauss.dimensions()[0] != n_g || weights_gauss.dimensions()[0] != n_g) {
    return ffi::Error::InvalidArgument("quadrature arrays do not match n_g");
  }
  if (output->dimensions()[0] != n_fl || output->dimensions()[1] != n_g) {
    return ffi::Error::InvalidArgument("output shape does not match weighted_all");
  }
  if (n_species < 1) {
    return ffi::Error::InvalidArgument("n_species must be at least 1");
  }

  std::vector<double> g_points(n_g);
  std::vector<double> gauss_weights(n_g);
  for (int i = 0; i < n_g; ++i) {
    g_points[i] = static_cast<double>(g_gauss.typed_data()[i]);
    gauss_weights[i] = static_cast<double>(weights_gauss.typed_data()[i]);
  }

  const T* weighted_ptr = weighted_all.typed_data();
  const T* continuum_ptr = continuum_flat.typed_data();
  T* output_ptr = output->typed_data();

  const int64_t pool_threads = thread_pool.num_threads();
  // Oversplit relative to the worker count: per-cell cost is data dependent
  // (threshold skips), so small tasks load-balance better.
  const int max_tasks = static_cast<int>(
      std::max<int64_t>(1, std::min<int64_t>(pool_threads * 8, n_fl)));
  const int chunk = (n_fl + max_tasks - 1) / max_tasks;
  const int n_tasks = (n_fl + chunk - 1) / chunk;

  if (n_tasks <= 1 || pool_threads <= 1) {
    Scratch scratch(n_g);
    MixCells<T>(weighted_ptr, continuum_ptr, g_points, gauss_weights, 0, n_fl,
                n_species, n_g, scratch, output_ptr);
    return ffi::Error::Success();
  }

  std::atomic<int> remaining(n_tasks - 1);
  std::mutex done_mutex;
  std::condition_variable done_cv;

  for (int task = 1; task < n_tasks; ++task) {
    const int begin = task * chunk;
    const int end = std::min(begin + chunk, n_fl);
    thread_pool.Schedule([=, &remaining, &done_mutex, &done_cv, &g_points,
                          &gauss_weights]() {
      Scratch scratch(n_g);
      MixCells<T>(weighted_ptr, continuum_ptr, g_points, gauss_weights, begin, end,
                  n_species, n_g, scratch, output_ptr);
      if (remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        std::lock_guard<std::mutex> lock(done_mutex);
        done_cv.notify_one();
      }
    });
  }

  {
    Scratch scratch(n_g);
    MixCells<T>(weighted_ptr, continuum_ptr, g_points, gauss_weights, 0,
                std::min(chunk, n_fl), n_species, n_g, scratch, output_ptr);
  }

  std::unique_lock<std::mutex> lock(done_mutex);
  done_cv.wait(lock, [&remaining]() {
    return remaining.load(std::memory_order_acquire) == 0;
  });

  return ffi::Error::Success();
}

ffi::Error PrtCombineCkMixF32Impl(ffi::ThreadPool thread_pool,
                                      ffi::Buffer<ffi::F32> weighted_all,
                                      ffi::Buffer<ffi::F32> continuum_flat,
                                      ffi::Buffer<ffi::F32> g_gauss,
                                      ffi::Buffer<ffi::F32> weights_gauss,
                                      ffi::ResultBuffer<ffi::F32> output) {
  return PrtCombineCkMixImpl<ffi::F32>(thread_pool, weighted_all, continuum_flat,
                                         g_gauss, weights_gauss, output);
}

ffi::Error PrtCombineCkMixF64Impl(ffi::ThreadPool thread_pool,
                                      ffi::Buffer<ffi::F64> weighted_all,
                                      ffi::Buffer<ffi::F64> continuum_flat,
                                      ffi::Buffer<ffi::F64> g_gauss,
                                      ffi::Buffer<ffi::F64> weights_gauss,
                                      ffi::ResultBuffer<ffi::F64> output) {
  return PrtCombineCkMixImpl<ffi::F64>(thread_pool, weighted_all, continuum_flat,
                                         g_gauss, weights_gauss, output);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    PrtCombineCkMixF32, PrtCombineCkMixF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::ThreadPool>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    PrtCombineCkMixF64, PrtCombineCkMixF64Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::ThreadPool>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::F64>>());

// ---------------------------------------------------------------------------
// Python module: expose the handlers as PyCapsules for
// jax.ffi.register_ffi_target.
// ---------------------------------------------------------------------------

PyObject* NewCapsule(void* pointer) {
  return PyCapsule_New(pointer, nullptr, nullptr);
}

PyObject* CombineCkMixF32Capsule(PyObject*, PyObject*) {
  return NewCapsule(reinterpret_cast<void*>(PrtCombineCkMixF32));
}

PyObject* CombineCkMixF64Capsule(PyObject*, PyObject*) {
  return NewCapsule(reinterpret_cast<void*>(PrtCombineCkMixF64));
}

PyMethodDef kMethods[] = {
    {"combine_ck_mix_f32_capsule", CombineCkMixF32Capsule, METH_NOARGS, nullptr},
    {"combine_ck_mix_f64_capsule", CombineCkMixF64Capsule, METH_NOARGS, nullptr},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef kModule = {
    PyModuleDef_HEAD_INIT,
    "prt_ck_mixing_ffi",
    nullptr,
    -1,
    kMethods,
};

}  // namespace

PyMODINIT_FUNC PyInit_prt_ck_mixing_ffi(void) {
  return PyModule_Create(&kModule);
}
