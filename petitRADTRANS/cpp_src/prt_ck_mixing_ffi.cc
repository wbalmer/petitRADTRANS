// Batched correlated-k opacity mixing as XLA FFI custom calls (CPU): the mixed
// opacities and their forward- and reverse-mode derivatives.
//
// Evert: I hope no one ever has to edit this file.
// Add to the counter if I've made you suffer: 0
//
// Replaces the former Fortran combine_ck_opacities kernel
// and computes the same mixing as the JAX implementation
// (_combine_ck_opacities_jax in radiative_transfer_core.py): per flattened
// (frequency, layer) cell, species are folded into the running mixture one
// at a time via the random-overlap approximation — all n_g x n_g pairwise
// opacity sums are formed and sorted into a quantile function over a
// cumulative g coordinate, which is averaged over a window around each node
// of the input Gaussian g grid. The window means are continuous in the
// opacities, unlike the Fortran code's linear reinterpolation of the sorted
// sums, which jumps whenever two sums of unequal weight swap places.
// Species whose strongest g-point falls below 1% of the dominant species'
// weakest g-point (continuum included) are skipped, and switched on smoothly
// between 0.5% and 1%.
// This ensures a smooth gradient, and removes the discontinuities present
// in the previous implementation.
//
// Design notes:
//   * float32 and float64 handlers (templated implementation); the per-cell
//     arithmetic is always performed in double precision.
//   * Parallelized over cells through the XLA intra-op thread pool
//     (ffi::ThreadPool), with several tasks per worker because the
//     data-dependent threshold skips make per-cell cost uneven.
//   * Inputs are the mass-fraction-weighted opacities in their natural
//     (n_frequencies * n_layers, n_species, n_g) row-major layout plus the
//     flattened continuum; no Fortran-order transpose is needed.
//   * The pairwise sums are ordered by merging the sorted runs they form
//     (k-table opacities ascend along g), ties by sample index, as a stable
//     sort would; JAX's stable argsort orders them identically.
//   * Derivatives: once the sums are ordered, a mixing step is linear in the
//     mixture and the species, a window mean weighting each sorted sum by the
//     fraction of the window it covers. The derivative kernels redo the
//     forward pass of each cell and apply these weights to the tangents
//     (forward mode) or cotangents (reverse mode), with the derivative of the
//     smooth threshold ramp. They reproduce the derivatives of the JAX
//     implementation, which handles everything beyond first order.

#define PY_SSIZE_T_CLEAN
#include <Python.h>

// Most of these are for dealing with parallelisation across cpu cores
#include <algorithm>
#include <atomic> // deal with concurrence. Allows definition of thread-safe types.
#include <condition_variable> // used to block a thread until another thread has finished
#include <cstdint> // fixed size integers
#include <mutex> // protect data that can be accessed from multiple threads
#include <vector>

// import the xla foreign function interface, interface.
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace {

// Keep in sync with CK_MIXING_THRESHOLD, CK_MIXING_RAMP_START and
// CK_MIXING_WINDOW_FRACTION in radiative_transfer_core.py.
constexpr double kMixingThreshold = 1e-2;
constexpr double kMixingRampStart = 0.5;
constexpr double kMixingWindowFraction = 1.0;

struct Sample {
  double opacity;
  int index;  // i * n_g + j: mixture node i, species node j
};

// Quantities shared by every cell: the pairwise quadrature weights and the
// averaging window of each g node.
struct Grid {
  int n_g = 0;
  int n_sample = 0;
  std::vector<double> pair_weights;  // (n_sample), summing to 1
  std::vector<double> window_low;    // (n_g)
  std::vector<double> window_high;   // (n_g)
};

// Build the n_g by n_g grid of g points and weights.
Grid BuildGrid(const std::vector<double>& g_points,
               const std::vector<double>& gauss_weights) {
  Grid grid;
  const int n_g = static_cast<int>(g_points.size());
  grid.n_g = n_g;
  grid.n_sample = n_g * n_g;

  // Pairwise weights as build_ck_combination_state in radiative_transfer_core.py
  // forms them, so that both implementations use the same values: products in
  // single precision (WORKING_JAX_DTYPE), normalised by their sum accumulated in
  // double precision and rounded back to single precision.
  grid.pair_weights.resize(grid.n_sample);
  double weight_sum = 0.0;
  for (int i = 0; i < n_g; ++i) {
    for (int j = 0; j < n_g; ++j) {
      const float weight =
          static_cast<float>(gauss_weights[i]) * static_cast<float>(gauss_weights[j]);
      grid.pair_weights[i * n_g + j] = static_cast<double>(weight);
      weight_sum += static_cast<double>(weight);
    }
  }
  for (double& weight : grid.pair_weights) {
    weight = static_cast<double>(static_cast<float>(weight / weight_sum));
  }

  // Averaging window of each g node: centred on the node and reaching
  // kMixingWindowFraction of the way to the nearer edge of its quadrature bin,
  // the bins being delimited by the cumulative quadrature weights. A node on a
  // bin edge gets no window and takes the quantile function at the node.
  // This replaces the linear interpolation for gradient smoothness,
  // and improves accuracy.
  std::vector<double> bin_edges(n_g + 1, 0.0);
  for (int i = 0; i < n_g; ++i) {
    bin_edges[i + 1] = bin_edges[i] + gauss_weights[i];
  }
  const double total_weight = bin_edges[n_g];
  for (double& edge : bin_edges) {
    edge /= total_weight;
  }
  grid.window_low.resize(n_g);
  grid.window_high.resize(n_g);
  for (int i = 0; i < n_g; ++i) {
    const double half_width =
        kMixingWindowFraction *
        std::max(0.0, std::min(g_points[i] - bin_edges[i], bin_edges[i + 1] - g_points[i]));
    grid.window_low[i] = g_points[i] - half_width;
    grid.window_high[i] = g_points[i] + half_width;
  }
  return grid;
}

// One mixing step of a cell, as the reverse-mode derivative needs it.
struct StepRecord {
  bool mixed = false;          // false: the species was skipped (weight 0)
  double mixing_weight = 0.0;  // lambda
  double weight_slope = 0.0;   // d lambda / d ratio
  double ratio = 0.0;          // strongest species opacity / threshold
  std::vector<int> order;                  // (n_sample) sample index per position
  std::vector<double> cumulative_weights;  // (n_sample)
  std::vector<int> covering;               // (2 n_g) position covering each bound
  std::vector<double> before;              // (n_g) mixture before the step
  std::vector<double> window_means;        // (n_g)
};

// Scratch memory for one task of the thread pool. Each task owns its
// workspace, so threads never share one, and reuses it for every cell it
// processes, so the per-cell loops never allocate. The arrays used in every
// mode hold one mixing step: the pairwise sums plus a second buffer to merge
// them into, the mixture, the species being mixed in, the cumulative weights
// and opacities, the sample covering each window bound, and the window means.
// The forward- and reverse-mode arrays are allocated only in those modes. In
// reverse mode the forward pass records every step (StepRecord), which the
// backward pass through the same cell then reads in reverse order.
struct Workspace {
  // Per-cell samples and intermediates
  std::vector<Sample> samples;
  std::vector<Sample> buffer;
  std::vector<double> current;
  std::vector<double> species;
  std::vector<double> cumulative_weights;
  std::vector<double> cumulative_opacities;
  std::vector<int> covering;  // window low bounds, then high bounds
  std::vector<double> window_means;
  // Forward mode
  std::vector<double> tangents;  // (n_tangents, n_g) of the mixture
  std::vector<double> threshold_tangents;
  std::vector<double> sample_tangents;
  // Reverse mode
  std::vector<StepRecord> steps;  // (n_species)
  std::vector<double> ct_sorted;  // cotangents of the sorted sums
  std::vector<double> ct_current;
  std::vector<double> ct_next;
  std::vector<double> ct_species;  // (n_species, n_g)

  Workspace(int n_g, int n_species, int n_tangents, bool forward, bool reverse)
      : samples(static_cast<size_t>(n_g) * n_g),
        buffer(static_cast<size_t>(n_g) * n_g),
        current(n_g),
        species(n_g),
        cumulative_weights(static_cast<size_t>(n_g) * n_g),
        cumulative_opacities(static_cast<size_t>(n_g) * n_g),
        covering(2 * n_g),
        window_means(n_g) {
    const size_t n_sample = static_cast<size_t>(n_g) * n_g;
    if (forward) {
      tangents.resize(static_cast<size_t>(n_tangents) * n_g);
      threshold_tangents.resize(n_tangents);
      sample_tangents.resize(n_sample);
    }
    if (reverse) {
      steps.resize(n_species);
      for (StepRecord& step : steps) {
        step.order.resize(n_sample);
        step.cumulative_weights.resize(n_sample);
        step.covering.resize(2 * n_g);
        step.before.resize(n_g);
        step.window_means.resize(n_g);
      }
      ct_sorted.resize(n_sample);
      ct_current.resize(n_g);
      ct_next.resize(n_g);
      ct_species.resize(static_cast<size_t>(n_species) * n_g);
    }
  }
};

void MixingWeight(double ratio, double* weight, double* slope) {
  // Smoothstep weight with which a species is mixed in, from its ratio to the
  // threshold: 0 below kMixingRampStart, 1 from 1 on; and its derivative.
  const double ramp = std::clamp(
      (ratio - kMixingRampStart) / (1.0 - kMixingRampStart), 0.0, 1.0);
  *weight = ramp * ramp * (3.0 - 2.0 * ramp);
  *slope = (ramp > 0.0 && ramp < 1.0)
               ? 6.0 * ramp * (1.0 - ramp) / (1.0 - kMixingRampStart)
               : 0.0;
}

int CoveringSample(const std::vector<double>& cumulative_weights, int n_sample,
                   double g) {
  // First sample whose cumulative weight reaches g (the last one if none does).
  const auto first = cumulative_weights.begin();
  const int index = static_cast<int>(
      std::lower_bound(first, first + n_sample, g) - first);
  return std::min(index, n_sample - 1);
}

const Sample* OrderPairSums(const Grid& grid, Workspace& ws) {
  // Orders the pairwise sums by opacity, ties by sample index. Row i,
  // current_i + species_j, is a sorted run when the species' opacities ascend
  // along g; the runs are then merged pairwise (std::merge keeps the first
  // range first on ties), which is cheaper than sorting. Returns the array
  // holding the result.
  const int n_g = grid.n_g;
  const int n_sample = grid.n_sample;
  const auto less = [](const Sample& lhs, const Sample& rhs) {
    return lhs.opacity < rhs.opacity;
  };
  if (!std::is_sorted(ws.species.begin(), ws.species.end())) { // if not sorted, sort
    std::stable_sort(ws.samples.begin(), ws.samples.end(), less);
    return ws.samples.data();
  }
  Sample* source = ws.samples.data();
  Sample* destination = ws.buffer.data();
  for (int run = n_g; run < n_sample; run *= 2) {
    for (int start = 0; start < n_sample; start += 2 * run) {
      const int middle = std::min(start + run, n_sample);
      const int end = std::min(start + 2 * run, n_sample);
      std::merge(source + start, source + middle, source + middle, source + end,
                 destination + start, less);
    }
    std::swap(source, destination);
  }
  return source;
}

double WindowSum(const Grid& grid, const Sample* sorted,
                 const std::vector<double>& cumulative_weights,
                 const std::vector<double>& values, int low_sample, int high_sample,
                 double low, double high) {
  // Integral over [low, high] of the step function taking values[l] on the
  // weight interval (C_{l-1}, C_l] of sorted sample l, summed over the samples
  // inside the window: the sorted sums weighted by the part of the window they
  // cover. Summing only these (rather than differencing cumulative sums) keeps
  // the derivatives free of cancellation.
  const double below_low = low_sample > 0 ? cumulative_weights[low_sample - 1] : 0.0;
  const double below_high = high_sample > 0 ? cumulative_weights[high_sample - 1] : 0.0;
  double sum = (high - below_high) * values[high_sample] - (low - below_low) * values[low_sample];
  for (int l = low_sample; l < high_sample; ++l) {
    sum += grid.pair_weights[sorted[l].index] * values[l];
  }
  return sum;
}

const Sample* WindowMeans(const Grid& grid, Workspace& ws) {
  // Window means of the pairwise sums of ws.current and ws.species. Sample m
  // covers g in (C_{m-1}, C_m] of the quantile function, C being the
  // cumulative weights; its integral is Phi(g) = P_m - (C_m - g) k_m, with P
  // the cumulative weighted opacities, and the window means are differences
  // of Phi. Returns the ordered samples.
  const int n_g = grid.n_g;
  const int n_sample = grid.n_sample;
  for (int i = 0; i < n_g; ++i) {
    for (int j = 0; j < n_g; ++j) {
      ws.samples[i * n_g + j] = Sample{ws.current[i] + ws.species[j], i * n_g + j};
    }
  }
  const Sample* sorted = OrderPairSums(grid, ws);

  double cumulative_weight = 0.0;
  double cumulative_opacity = 0.0;
  for (int l = 0; l < n_sample; ++l) {
    const double weight = grid.pair_weights[sorted[l].index];
    cumulative_weight += weight;
    cumulative_opacity += weight * sorted[l].opacity;
    ws.cumulative_weights[l] = cumulative_weight;
    ws.cumulative_opacities[l] = cumulative_opacity;
  }

  for (int g_idx = 0; g_idx < n_g; ++g_idx) {
    const double low = grid.window_low[g_idx];
    const double high = grid.window_high[g_idx];
    const int low_sample = CoveringSample(ws.cumulative_weights, n_sample, low);
    const int high_sample = CoveringSample(ws.cumulative_weights, n_sample, high);
    ws.covering[g_idx] = low_sample;
    ws.covering[n_g + g_idx] = high_sample;
    double mean = sorted[high_sample].opacity;
    if (high > low) {
      const double integral_high =
          ws.cumulative_opacities[high_sample] -
          (ws.cumulative_weights[high_sample] - high) * sorted[high_sample].opacity;
      const double integral_low =
          ws.cumulative_opacities[low_sample] -
          (ws.cumulative_weights[low_sample] - low) * sorted[low_sample].opacity;
      mean = (integral_high - integral_low) / (high - low);
    }
    ws.window_means[g_idx] = mean;
  }
  return sorted;
}

// class Mode
// What MixCells computes for each cell, chosen at compile time (a template
// argument, so each FFI handler gets its own specialised loop):
//   kValue    the mixed opacities;
//   kForward  the tangents of the mixed opacities, propagated alongside the
//             mixing from the input tangents (forward-mode derivative);
//   kReverse  the cotangents of the inputs: the mixing runs forward recording
//             every step, then BackPropagateCell runs back through the
//             steps from the output cotangents (reverse-mode derivative).
enum class Mode { kValue, kForward, kReverse };

// struct to store the values, tangents, and cotangents
template <typename T>
struct Buffers {
  const T* weighted;            // (n_fl, n_species, n_g)
  const T* continuum;           // (n_fl)
  const T* tangents;            // forward: (n_tangents, n_fl, n_species, n_g)
  const T* continuum_tangents;  // forward: (n_tangents, n_fl)
  const T* cotangents;          // reverse: (n_tangents, n_fl, n_g)
  T* output;                    // value: (n_fl, n_g); forward: (n_tangents, n_fl, n_g);
                                // reverse: (n_tangents, n_fl, n_species, n_g)
  T* continuum_output;          // reverse: (n_tangents, n_fl)
  int n_fl;
  int n_species;
  int n_g;
  int n_tangents;
};

template <typename T>
void BackPropagateCell(const Buffers<T>& b, const Grid& grid, int fl,
                       int strongest_species, double threshold, Workspace& ws) {
  // Reverse-mode derivative of one cell for each cotangent, through the steps
  // recorded by the forward pass.
  const int n_g = b.n_g;
  const int n_species = b.n_species;
  const int n_sample = grid.n_sample;

  for (int t = 0; t < b.n_tangents; ++t) {
    const T* cotangent = b.cotangents + (static_cast<size_t>(t) * b.n_fl + fl) * n_g;
    for (int g_idx = 0; g_idx < n_g; ++g_idx) {
      ws.ct_current[g_idx] = static_cast<double>(cotangent[g_idx]);
    }
    std::fill(ws.ct_species.begin(), ws.ct_species.end(), 0.0);
    double ct_threshold = 0.0;

    for (int s = n_species - 1; s >= 1; --s) {
      const StepRecord& step = ws.steps[s];
      if (!step.mixed) {
        continue;
      }
      // current + weight * (window_means - current)
      const double weight = step.mixing_weight;
      double ct_weight = 0.0;
      for (int g_idx = 0; g_idx < n_g; ++g_idx) {
        ct_weight += ws.ct_current[g_idx] * (step.window_means[g_idx] - step.before[g_idx]);
        ws.ct_next[g_idx] = (1.0 - weight) * ws.ct_current[g_idx];
      }

      // Window means -> sorted sums: each window mean weights the sorted sums
      // by the part of the window they cover (WindowSum)
      std::fill(ws.ct_sorted.begin(), ws.ct_sorted.end(), 0.0);
      for (int g_idx = 0; g_idx < n_g; ++g_idx) {
        const double ct_mean = weight * ws.ct_current[g_idx];
        const double low = grid.window_low[g_idx];
        const double high = grid.window_high[g_idx];
        const int low_sample = step.covering[g_idx];
        const int high_sample = step.covering[n_g + g_idx];
        if (!(high > low)) {
          ws.ct_sorted[high_sample] += ct_mean;
          continue;
        }
        const double scale = ct_mean / (high - low);
        for (int l = low_sample; l < high_sample; ++l) {
          ws.ct_sorted[l] += scale * grid.pair_weights[step.order[l]];
        }
        ws.ct_sorted[high_sample] +=
            scale * (high - (high_sample > 0 ? step.cumulative_weights[high_sample - 1] : 0.0));
        ws.ct_sorted[low_sample] -=
            scale * (low - (low_sample > 0 ? step.cumulative_weights[low_sample - 1] : 0.0));
      }
      double* ct_species = ws.ct_species.data() + static_cast<size_t>(s) * n_g;
      for (int l = 0; l < n_sample; ++l) {
        const int index = step.order[l];
        ws.ct_next[index / n_g] += ws.ct_sorted[l];
        ct_species[index % n_g] += ws.ct_sorted[l];
      }

      // Mixing weight, through the ratio of the strongest opacity to the threshold
      if (step.weight_slope != 0.0) {
        const double ct_ratio = step.weight_slope * ct_weight;
        ct_species[n_g - 1] += ct_ratio / threshold;
        ct_threshold -= ct_ratio * step.ratio / threshold;
      }
      std::swap(ws.ct_current, ws.ct_next);
    }

    // First species plus continuum, and the threshold
    double ct_continuum = 0.0;
    for (int g_idx = 0; g_idx < n_g; ++g_idx) {
      ws.ct_species[g_idx] += ws.ct_current[g_idx];
      ct_continuum += ws.ct_current[g_idx];
    }
    const double ct_strongest = kMixingThreshold * ct_threshold;
    ws.ct_species[static_cast<size_t>(strongest_species) * n_g] += ct_strongest;
    if (strongest_species == 0) {
      ct_continuum += ct_strongest;
    }

    T* output = b.output + (static_cast<size_t>(t) * b.n_fl + fl) * n_species * n_g;
    for (int k = 0; k < n_species * n_g; ++k) {
      output[k] = static_cast<T>(ws.ct_species[k]);
    }
    b.continuum_output[static_cast<size_t>(t) * b.n_fl + fl] = static_cast<T>(ct_continuum);
  }
}

// This is the core function that does the heavy lifting.
template <Mode kMode, typename T>
void MixCells(const Buffers<T>& b, const Grid& grid, int fl_begin, int fl_end,
              Workspace& ws) {
  // Mixes the species of each cell in [fl_begin, fl_end) one at a time: form
  // the n_g x n_g pairwise sums with the current mixture, order them, average
  // the resulting quantile function over the window of each g node, and blend
  // the result in with the species' mixing weight (skipping species of weight
  // 0). Writes the mixed opacities, or their tangents (forward mode) or the
  // cotangents of the inputs (reverse mode).
  const int n_g = b.n_g;
  const int n_species = b.n_species;
  const int n_sample = grid.n_sample;

  for (int fl = fl_begin; fl < fl_end; ++fl) { // frequency chunks
    const T* cell_ptr = b.weighted + static_cast<size_t>(fl) * n_species * n_g;
    const double continuum = static_cast<double>(b.continuum[fl]);

    // Threshold from the largest weakest-g opacity, taken at the first species
    // that reaches it (as jnp.argmax does)
    int strongest_species = 0;
    double threshold = static_cast<double>(cell_ptr[0]) + continuum;
    for (int species_idx = 1; species_idx < n_species; ++species_idx) {
      const double opacity = static_cast<double>(cell_ptr[species_idx * n_g]);
      if (opacity > threshold) {
        threshold = opacity;
        strongest_species = species_idx;
      }
    }
    threshold *= kMixingThreshold;
    const bool has_threshold = threshold > 0.0;

    for (int g_idx = 0; g_idx < n_g; ++g_idx) {
      ws.current[g_idx] = static_cast<double>(cell_ptr[g_idx]) + continuum;
    }

    if constexpr (kMode == Mode::kForward) { // Forward mode gradient
      for (int t = 0; t < b.n_tangents; ++t) {
        const T* tangent = b.tangents + (static_cast<size_t>(t) * b.n_fl + fl) * n_species * n_g;
        const double continuum_tangent =
            static_cast<double>(b.continuum_tangents[static_cast<size_t>(t) * b.n_fl + fl]);
        for (int g_idx = 0; g_idx < n_g; ++g_idx) {
          ws.tangents[static_cast<size_t>(t) * n_g + g_idx] =
              static_cast<double>(tangent[g_idx]) + continuum_tangent;
        }
        const double strongest_tangent =
            strongest_species == 0
                ? static_cast<double>(tangent[0]) + continuum_tangent
                : static_cast<double>(tangent[strongest_species * n_g]);
        ws.threshold_tangents[t] = kMixingThreshold * strongest_tangent;
      }
    }

    // Apply the threshold
    for (int species_idx = 1; species_idx < n_species; ++species_idx) {
      const T* species_ptr = cell_ptr + species_idx * n_g;
      for (int g_idx = 0; g_idx < n_g; ++g_idx) {
        ws.species[g_idx] = static_cast<double>(species_ptr[g_idx]);
      }
      double weight = 1.0;
      double slope = 0.0;
      double ratio = 0.0;
      if (has_threshold) { // gradually turn on species rather than hard cutoff
        ratio = ws.species[n_g - 1] / threshold;
        MixingWeight(ratio, &weight, &slope);
      }
      if constexpr (kMode == Mode::kReverse) {
        ws.steps[species_idx].mixed = false;
      }
      if (!(weight > 0.0)) {
        continue;
      }

      const Sample* sorted = WindowMeans(grid, ws);

      if constexpr (kMode == Mode::kForward) { // Forward mode derivative
        for (int t = 0; t < b.n_tangents; ++t) {
          double* mixture_tangent = ws.tangents.data() + static_cast<size_t>(t) * n_g;
          const T* species_tangent =
              b.tangents + ((static_cast<size_t>(t) * b.n_fl + fl) * n_species + species_idx) * n_g;
          for (int l = 0; l < n_sample; ++l) {
            const int index = sorted[l].index;
            ws.sample_tangents[l] = mixture_tangent[index / n_g] +
                                    static_cast<double>(species_tangent[index % n_g]);
          }
          const double ratio_tangent =
              has_threshold
                  ? (static_cast<double>(species_tangent[n_g - 1]) - ratio * ws.threshold_tangents[t]) /
                        threshold
                  : 0.0;
          const double weight_tangent = slope * ratio_tangent;
          for (int g_idx = 0; g_idx < n_g; ++g_idx) {
            const double low = grid.window_low[g_idx];
            const double high = grid.window_high[g_idx];
            double mean_tangent = ws.sample_tangents[ws.covering[n_g + g_idx]];
            if (high > low) {
              mean_tangent = WindowSum(grid, sorted, ws.cumulative_weights, ws.sample_tangents,
                                       ws.covering[g_idx], ws.covering[n_g + g_idx], low, high) /
                             (high - low);
            }
            mixture_tangent[g_idx] += weight * (mean_tangent - mixture_tangent[g_idx]) +
                                      weight_tangent * (ws.window_means[g_idx] - ws.current[g_idx]);
          }
        }
      }

      if constexpr (kMode == Mode::kReverse) {
        StepRecord& step = ws.steps[species_idx];
        step.mixed = true;
        step.mixing_weight = weight;
        step.weight_slope = slope;
        step.ratio = ratio;
        for (int l = 0; l < n_sample; ++l) {
          step.order[l] = sorted[l].index;
        }
        step.cumulative_weights = ws.cumulative_weights;
        step.covering = ws.covering;
        step.before = ws.current;
        step.window_means = ws.window_means;
      }

      for (int g_idx = 0; g_idx < n_g; ++g_idx) {
        ws.current[g_idx] += weight * (ws.window_means[g_idx] - ws.current[g_idx]);
      }
    }

    if constexpr (kMode == Mode::kValue) {
      T* output_cell = b.output + static_cast<size_t>(fl) * n_g;
      for (int g_idx = 0; g_idx < n_g; ++g_idx) {
        output_cell[g_idx] = static_cast<T>(ws.current[g_idx]);
      }
    } else if constexpr (kMode == Mode::kForward) {
      for (int t = 0; t < b.n_tangents; ++t) {
        T* output_cell = b.output + (static_cast<size_t>(t) * b.n_fl + fl) * n_g;
        for (int g_idx = 0; g_idx < n_g; ++g_idx) {
          output_cell[g_idx] = static_cast<T>(ws.tangents[static_cast<size_t>(t) * n_g + g_idx]);
        }
      }
    } else {
      BackPropagateCell(b, grid, fl, strongest_species, threshold, ws);
    }
  }
}

template <Mode kMode, typename T>
void RunMixCells(ffi::ThreadPool& thread_pool, const Buffers<T>& b, const Grid& grid) {
  // Splits the cells into tasks on the XLA intra-op thread pool, each with its
  // own workspace.
  const int n_fl = b.n_fl;
  const bool forward = kMode == Mode::kForward;
  const bool reverse = kMode == Mode::kReverse;
  const int64_t pool_threads = thread_pool.num_threads();
  // Oversplit relative to the worker count: per-cell cost is data dependent
  // (threshold skips), so small tasks load-balance better.
  const int max_tasks = static_cast<int>(
      std::max<int64_t>(1, std::min<int64_t>(pool_threads * 8, n_fl)));
  const int chunk = (n_fl + max_tasks - 1) / max_tasks;
  const int n_tasks = (n_fl + chunk - 1) / chunk;

  // If the problem is too small, or we have only one worker thread, skip parallelism
  if (n_tasks <= 1 || pool_threads <= 1) {
    Workspace ws(b.n_g, b.n_species, b.n_tangents, forward, reverse);
    MixCells<kMode, T>(b, grid, 0, n_fl, ws);
    return;
  }

  std::atomic<int> remaining(n_tasks - 1);
  std::mutex done_mutex;
  std::condition_variable done_cv;

  // Schedule all but the first task, which we run immediately.
  for (int task = 1; task < n_tasks; ++task) {
    const int begin = task * chunk;
    const int end = std::min(begin + chunk, n_fl);
    thread_pool.Schedule([=, &b, &grid, &remaining, &done_mutex, &done_cv]() {
      Workspace ws(b.n_g, b.n_species, b.n_tangents, forward, reverse);
      MixCells<kMode, T>(b, grid, begin, end, ws);
      // Wait until everything is done.
      if (remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        std::lock_guard<std::mutex> lock(done_mutex);
        done_cv.notify_one();
      }
    });
  }

  {
    Workspace ws(b.n_g, b.n_species, b.n_tangents, forward, reverse);
    MixCells<kMode, T>(b, grid, 0, std::min(chunk, n_fl), ws);
  }

  std::unique_lock<std::mutex> lock(done_mutex);
  done_cv.wait(lock, [&remaining]() {
    return remaining.load(std::memory_order_acquire) == 0;
  });
}

template <ffi::DataType DT>
ffi::Error PrepareInputs(ffi::Buffer<DT>& weighted_all, ffi::Buffer<DT>& continuum_flat,
                         ffi::Buffer<DT>& g_gauss, ffi::Buffer<DT>& weights_gauss,
                         Buffers<ffi::NativeType<DT>>* b, Grid* grid) {
  // Validates the primal inputs, fills their pointers and sizes, and builds the grid.
  if (weighted_all.dimensions().size() != 3) {
    return ffi::Error::InvalidArgument("weighted_all must be rank-3");
  }
  if (continuum_flat.dimensions().size() != 1 || g_gauss.dimensions().size() != 1 ||
      weights_gauss.dimensions().size() != 1) {
    return ffi::Error::InvalidArgument(
        "continuum_flat, g_gauss, and weights_gauss must be rank-1");
  }
  b->n_fl = static_cast<int>(weighted_all.dimensions()[0]);
  b->n_species = static_cast<int>(weighted_all.dimensions()[1]);
  b->n_g = static_cast<int>(weighted_all.dimensions()[2]);
  if (continuum_flat.dimensions()[0] != b->n_fl) {
    return ffi::Error::InvalidArgument("continuum_flat length does not match n_fl");
  }
  if (g_gauss.dimensions()[0] != b->n_g || weights_gauss.dimensions()[0] != b->n_g) {
    return ffi::Error::InvalidArgument("quadrature arrays do not match n_g");
  }
  if (b->n_species < 1) {
    return ffi::Error::InvalidArgument("n_species must be at least 1");
  }
  b->weighted = weighted_all.typed_data();
  b->continuum = continuum_flat.typed_data();

  // Convert quadrature points and weights to double precision
  std::vector<double> g_points(b->n_g);
  std::vector<double> gauss_weights(b->n_g);
  for (int i = 0; i < b->n_g; ++i) {
    g_points[i] = static_cast<double>(g_gauss.typed_data()[i]);
    gauss_weights[i] = static_cast<double>(weights_gauss.typed_data()[i]);
  }
  *grid = BuildGrid(g_points, gauss_weights);
  return ffi::Error::Success();
}

template <ffi::DataType DT>
ffi::Error PrtCombineCkMixImpl(ffi::ThreadPool thread_pool,
                               ffi::Buffer<DT> weighted_all,
                               ffi::Buffer<DT> continuum_flat,
                               ffi::Buffer<DT> g_gauss,
                               ffi::Buffer<DT> weights_gauss,
                               ffi::ResultBuffer<DT> output) {
  // Mixed opacities (n_fl, n_g).
  Buffers<ffi::NativeType<DT>> b{};
  Grid grid;
  ffi::Error error = PrepareInputs(weighted_all, continuum_flat, g_gauss, weights_gauss, &b, &grid);
  if (error.failure()) {
    return error;
  }
  if (output->dimensions().size() != 2 || output->dimensions()[0] != b.n_fl ||
      output->dimensions()[1] != b.n_g) {
    return ffi::Error::InvalidArgument("output shape does not match weighted_all");
  }
  b.output = output->typed_data();
  b.n_tangents = 0;
  RunMixCells<Mode::kValue>(thread_pool, b, grid);
  return ffi::Error::Success();
}

// Forward mode derivative
template <ffi::DataType DT>
ffi::Error PrtCombineCkMixJvpImpl(ffi::ThreadPool thread_pool,
                                  ffi::Buffer<DT> weighted_all,
                                  ffi::Buffer<DT> continuum_flat,
                                  ffi::Buffer<DT> g_gauss,
                                  ffi::Buffer<DT> weights_gauss,
                                  ffi::Buffer<DT> weighted_tangents,
                                  ffi::Buffer<DT> continuum_tangents,
                                  ffi::ResultBuffer<DT> output_tangents) {
  // Tangents (n_tangents, n_fl, n_g) of the mixed opacities, from tangents
  // (n_tangents, n_fl, n_species, n_g) and (n_tangents, n_fl) of the inputs.
  Buffers<ffi::NativeType<DT>> b{};
  Grid grid;
  ffi::Error error = PrepareInputs(weighted_all, continuum_flat, g_gauss, weights_gauss, &b, &grid);
  if (error.failure()) {
    return error;
  }
  const auto tangent_dims = weighted_tangents.dimensions();
  if (tangent_dims.size() != 4 || tangent_dims[1] != b.n_fl ||
      tangent_dims[2] != b.n_species || tangent_dims[3] != b.n_g) {
    return ffi::Error::InvalidArgument("weighted_tangents must be (n_tangents, n_fl, n_species, n_g)");
  }
  b.n_tangents = static_cast<int>(tangent_dims[0]);
  if (continuum_tangents.dimensions().size() != 2 ||
      continuum_tangents.dimensions()[0] != b.n_tangents ||
      continuum_tangents.dimensions()[1] != b.n_fl) {
    return ffi::Error::InvalidArgument("continuum_tangents must be (n_tangents, n_fl)");
  }
  if (output_tangents->dimensions().size() != 3 ||
      output_tangents->dimensions()[0] != b.n_tangents ||
      output_tangents->dimensions()[1] != b.n_fl ||
      output_tangents->dimensions()[2] != b.n_g) {
    return ffi::Error::InvalidArgument("output_tangents must be (n_tangents, n_fl, n_g)");
  }
  b.tangents = weighted_tangents.typed_data();
  b.continuum_tangents = continuum_tangents.typed_data();
  b.output = output_tangents->typed_data();
  RunMixCells<Mode::kForward>(thread_pool, b, grid);
  return ffi::Error::Success();
}

// Reverse mode derivative
template <ffi::DataType DT>
ffi::Error PrtCombineCkMixVjpImpl(ffi::ThreadPool thread_pool,
                                  ffi::Buffer<DT> weighted_all,
                                  ffi::Buffer<DT> continuum_flat,
                                  ffi::Buffer<DT> g_gauss,
                                  ffi::Buffer<DT> weights_gauss,
                                  ffi::Buffer<DT> cotangents,
                                  ffi::ResultBuffer<DT> weighted_cotangents,
                                  ffi::ResultBuffer<DT> continuum_cotangents) {
  // Cotangents (n_tangents, n_fl, n_species, n_g) and (n_tangents, n_fl) of
  // the inputs, from cotangents (n_tangents, n_fl, n_g) of the mixed opacities.
  Buffers<ffi::NativeType<DT>> b{};
  Grid grid;
  ffi::Error error = PrepareInputs(weighted_all, continuum_flat, g_gauss, weights_gauss, &b, &grid);
  if (error.failure()) {
    return error;
  }
  const auto cotangent_dims = cotangents.dimensions();
  if (cotangent_dims.size() != 3 || cotangent_dims[1] != b.n_fl || cotangent_dims[2] != b.n_g) {
    return ffi::Error::InvalidArgument("cotangents must be (n_tangents, n_fl, n_g)");
  }
  b.n_tangents = static_cast<int>(cotangent_dims[0]);
  const auto weighted_dims = weighted_cotangents->dimensions();
  if (weighted_dims.size() != 4 || weighted_dims[0] != b.n_tangents ||
      weighted_dims[1] != b.n_fl || weighted_dims[2] != b.n_species ||
      weighted_dims[3] != b.n_g) {
    return ffi::Error::InvalidArgument(
        "weighted_cotangents must be (n_tangents, n_fl, n_species, n_g)");
  }
  if (continuum_cotangents->dimensions().size() != 2 ||
      continuum_cotangents->dimensions()[0] != b.n_tangents ||
      continuum_cotangents->dimensions()[1] != b.n_fl) {
    return ffi::Error::InvalidArgument("continuum_cotangents must be (n_tangents, n_fl)");
  }
  b.cotangents = cotangents.typed_data();
  b.output = weighted_cotangents->typed_data();
  b.continuum_output = continuum_cotangents->typed_data();
  RunMixCells<Mode::kReverse>(thread_pool, b, grid);
  return ffi::Error::Success();
}

// ---------------------------------------------------------------------------
// Interface to ffi
// Need to define handlers for the values and derivatives
// for ffi to pass through to be accessed in Python.
// ---------------------------------------------------------------------------


#define PRT_CK_MIX_HANDLERS(SUFFIX, DT)                                              \
  ffi::Error PrtCombineCkMix##SUFFIX##Impl(                                          \
      ffi::ThreadPool thread_pool, ffi::Buffer<DT> weighted_all,                     \
      ffi::Buffer<DT> continuum_flat, ffi::Buffer<DT> g_gauss,                       \
      ffi::Buffer<DT> weights_gauss, ffi::ResultBuffer<DT> output) {                 \
    return PrtCombineCkMixImpl<DT>(thread_pool, weighted_all, continuum_flat,        \
                                   g_gauss, weights_gauss, output);                  \
  }                                                                                  \
  ffi::Error PrtCombineCkMixJvp##SUFFIX##Impl(                                       \
      ffi::ThreadPool thread_pool, ffi::Buffer<DT> weighted_all,                     \
      ffi::Buffer<DT> continuum_flat, ffi::Buffer<DT> g_gauss,                       \
      ffi::Buffer<DT> weights_gauss, ffi::Buffer<DT> weighted_tangents,              \
      ffi::Buffer<DT> continuum_tangents, ffi::ResultBuffer<DT> output_tangents) {   \
    return PrtCombineCkMixJvpImpl<DT>(thread_pool, weighted_all, continuum_flat,     \
                                      g_gauss, weights_gauss, weighted_tangents,     \
                                      continuum_tangents, output_tangents);          \
  }                                                                                  \
  ffi::Error PrtCombineCkMixVjp##SUFFIX##Impl(                                       \
      ffi::ThreadPool thread_pool, ffi::Buffer<DT> weighted_all,                     \
      ffi::Buffer<DT> continuum_flat, ffi::Buffer<DT> g_gauss,                       \
      ffi::Buffer<DT> weights_gauss, ffi::Buffer<DT> cotangents,                     \
      ffi::ResultBuffer<DT> weighted_cotangents,                                     \
      ffi::ResultBuffer<DT> continuum_cotangents) {                                  \
    return PrtCombineCkMixVjpImpl<DT>(thread_pool, weighted_all, continuum_flat,     \
                                      g_gauss, weights_gauss, cotangents,            \
                                      weighted_cotangents, continuum_cotangents);    \
  }

PRT_CK_MIX_HANDLERS(F32, ffi::F32)
PRT_CK_MIX_HANDLERS(F64, ffi::F64)

#undef PRT_CK_MIX_HANDLERS

#define PRT_CK_MIX_SYMBOLS(SUFFIX, DT)                                     \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(PrtCombineCkMix##SUFFIX,                   \
                                PrtCombineCkMix##SUFFIX##Impl,             \
                                ffi::Ffi::Bind()                           \
                                    .Ctx<ffi::ThreadPool>()                \
                                    .Arg<ffi::Buffer<DT>>()                \
                                    .Arg<ffi::Buffer<DT>>()                \
                                    .Arg<ffi::Buffer<DT>>()                \
                                    .Arg<ffi::Buffer<DT>>()                \
                                    .Ret<ffi::Buffer<DT>>());              \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(PrtCombineCkMixJvp##SUFFIX,                \
                                PrtCombineCkMixJvp##SUFFIX##Impl,          \
                                ffi::Ffi::Bind()                           \
                                    .Ctx<ffi::ThreadPool>()                \
                                    .Arg<ffi::Buffer<DT>>()                \
                                    .Arg<ffi::Buffer<DT>>()                \
                                    .Arg<ffi::Buffer<DT>>()                \
                                    .Arg<ffi::Buffer<DT>>()                \
                                    .Arg<ffi::Buffer<DT>>()                \
                                    .Arg<ffi::Buffer<DT>>()                \
                                    .Ret<ffi::Buffer<DT>>());              \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(PrtCombineCkMixVjp##SUFFIX,                \
                                PrtCombineCkMixVjp##SUFFIX##Impl,          \
                                ffi::Ffi::Bind()                           \
                                    .Ctx<ffi::ThreadPool>()                \
                                    .Arg<ffi::Buffer<DT>>()                \
                                    .Arg<ffi::Buffer<DT>>()                \
                                    .Arg<ffi::Buffer<DT>>()                \
                                    .Arg<ffi::Buffer<DT>>()                \
                                    .Arg<ffi::Buffer<DT>>()                \
                                    .Ret<ffi::Buffer<DT>>()                \
                                    .Ret<ffi::Buffer<DT>>());

PRT_CK_MIX_SYMBOLS(F32, ffi::F32)
PRT_CK_MIX_SYMBOLS(F64, ffi::F64)

#undef PRT_CK_MIX_SYMBOLS

// ---------------------------------------------------------------------------
// Python module: expose the handlers as PyCapsules for
// jax.ffi.register_ffi_target.
// ---------------------------------------------------------------------------

PyObject* NewCapsule(void* pointer) {
  return PyCapsule_New(pointer, nullptr, nullptr);
}

#define PRT_CK_MIX_CAPSULE(NAME, SYMBOL)                    \
  PyObject* NAME(PyObject*, PyObject*) {                    \
    return NewCapsule(reinterpret_cast<void*>(SYMBOL));     \
  }

PRT_CK_MIX_CAPSULE(CombineCkMixF32Capsule, PrtCombineCkMixF32)
PRT_CK_MIX_CAPSULE(CombineCkMixF64Capsule, PrtCombineCkMixF64)
PRT_CK_MIX_CAPSULE(CombineCkMixJvpF32Capsule, PrtCombineCkMixJvpF32)
PRT_CK_MIX_CAPSULE(CombineCkMixJvpF64Capsule, PrtCombineCkMixJvpF64)
PRT_CK_MIX_CAPSULE(CombineCkMixVjpF32Capsule, PrtCombineCkMixVjpF32)
PRT_CK_MIX_CAPSULE(CombineCkMixVjpF64Capsule, PrtCombineCkMixVjpF64)

#undef PRT_CK_MIX_CAPSULE

PyMethodDef kMethods[] = {
    {"combine_ck_mix_f32_capsule", CombineCkMixF32Capsule, METH_NOARGS, nullptr},
    {"combine_ck_mix_f64_capsule", CombineCkMixF64Capsule, METH_NOARGS, nullptr},
    {"combine_ck_mix_jvp_f32_capsule", CombineCkMixJvpF32Capsule, METH_NOARGS, nullptr},
    {"combine_ck_mix_jvp_f64_capsule", CombineCkMixJvpF64Capsule, METH_NOARGS, nullptr},
    {"combine_ck_mix_vjp_f32_capsule", CombineCkMixVjpF32Capsule, METH_NOARGS, nullptr},
    {"combine_ck_mix_vjp_f64_capsule", CombineCkMixVjpF64Capsule, METH_NOARGS, nullptr},
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
