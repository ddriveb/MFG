// Cached-layout C++17 float64 fixed-budget Mirror-Prox for Ticket07.
// The opaque runtime owns immutable unit incidence/layout and reusable work
// buffers. Each call supplies current reduced demands and fresh NN initials.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(_WIN32)
#define MFG_FAST_EXPORT __declspec(dllexport)
#else
#define MFG_FAST_EXPORT __attribute__((visibility("default")))
#endif

namespace {

struct Runtime {
  std::size_t row_count = 0;
  std::size_t variable_count = 0;
  std::size_t block_count = 0;
  std::size_t resource_count = 0;
  std::size_t incidence_count = 0;
  std::size_t original_row_count = 0;
  std::size_t original_slot_count = 0;
  double fixed_startup_ms = 0.0;
  std::vector<std::uint64_t> row_offsets;
  std::vector<std::uint64_t> dual_offsets;
  std::vector<std::uint64_t> column_offsets;
  std::vector<std::uint64_t> resource_indices;
  std::vector<std::uint64_t> original_row_offsets;
  std::vector<std::uint64_t> original_row_groups;
  std::vector<std::uint64_t> original_slot_to_group_slot;
  std::vector<std::uint64_t> round_floors;
  std::vector<std::uint64_t> demand_sums;
  std::vector<std::size_t> round_order;
  std::vector<double> round_fractions;
  std::vector<double> numerators;
  std::vector<double> denominators;
  std::vector<std::size_t> row_of_variable;
  std::vector<double> coefficients;

  std::vector<double> p_current;
  std::vector<double> p_trial;
  std::vector<double> p_next;
  std::vector<double> p_average;
  std::vector<double> gradient;
  std::vector<double> gradient_trial;
  std::vector<double> dual_current;
  std::vector<double> dual_trial;
  std::vector<double> dual_next;
  std::vector<double> dual_update;
  std::vector<double> dual_average;
  std::vector<double> loads_current;
  std::vector<double> loads_trial;
  std::vector<double> resource_sums;
  std::vector<double> projection_scratch;
  std::vector<double> uniform_shares;
};

bool scalar_is_finite(double value) { return std::isfinite(value); }

void copy_error(char* output, std::size_t capacity, const std::string& message) {
  if (!output || capacity == 0) return;
  const std::size_t count = std::min(capacity - 1, message.size());
  std::memcpy(output, message.data(), count);
  output[count] = '\0';
}

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void validate_simplexes(const Runtime& runtime, const double* primal,
                        const double* dual) {
  for (std::size_t row = 0; row < runtime.row_count; ++row) {
    const auto begin = static_cast<std::size_t>(runtime.row_offsets[row]);
    const auto end = static_cast<std::size_t>(runtime.row_offsets[row + 1]);
    double total = 0.0;
    for (std::size_t i = begin; i < end; ++i) {
      require(scalar_is_finite(primal[i]), "initial primal contains a non-finite value");
      require(primal[i] >= -1e-12, "initial primal contains a negative value");
      total += primal[i];
    }
    require(std::abs(total - 1.0) <= 1e-9,
            "each initial primal row must sum to one");
  }
  for (std::size_t block = 0; block < runtime.block_count; ++block) {
    const auto begin = static_cast<std::size_t>(runtime.dual_offsets[block]);
    const auto end = static_cast<std::size_t>(runtime.dual_offsets[block + 1]);
    double total = 0.0;
    for (std::size_t i = begin; i < end; ++i) {
      require(scalar_is_finite(dual[i]), "initial dual contains a non-finite value");
      require(dual[i] >= -1e-12, "initial dual contains a negative value");
      total += dual[i];
    }
    require(std::abs(total - 1.0) <= 1e-9,
            "each initial dual block must sum to one");
  }
}

void apply_operator(Runtime& runtime, const std::vector<double>& shares,
                    std::vector<double>& loads) {
  std::fill(loads.begin(), loads.end(), 0.0);
  for (std::size_t block = 0; block < runtime.block_count; ++block) {
    const std::size_t column_base = block * (runtime.variable_count + 1);
    const std::size_t resource_base =
        static_cast<std::size_t>(runtime.dual_offsets[block]);
    for (std::size_t variable = 0; variable < runtime.variable_count; ++variable) {
      const double share = shares[variable];
      if (share == 0.0) continue;
      const auto begin = static_cast<std::size_t>(
          runtime.column_offsets[column_base + variable]);
      const auto end = static_cast<std::size_t>(
          runtime.column_offsets[column_base + variable + 1]);
      for (std::size_t incidence = begin; incidence < end; ++incidence) {
        const std::size_t resource =
            resource_base + static_cast<std::size_t>(runtime.resource_indices[incidence]);
        loads[resource] += runtime.coefficients[incidence] * share;
      }
    }
  }
}

void apply_adjoint(Runtime& runtime, const std::vector<double>& weights,
                   std::vector<double>& gradients) {
  std::fill(gradients.begin(), gradients.end(), 0.0);
  for (std::size_t block = 0; block < runtime.block_count; ++block) {
    const std::size_t column_base = block * (runtime.variable_count + 1);
    const std::size_t resource_base =
        static_cast<std::size_t>(runtime.dual_offsets[block]);
    for (std::size_t variable = 0; variable < runtime.variable_count; ++variable) {
      double column_value = 0.0;
      const auto begin = static_cast<std::size_t>(
          runtime.column_offsets[column_base + variable]);
      const auto end = static_cast<std::size_t>(
          runtime.column_offsets[column_base + variable + 1]);
      for (std::size_t incidence = begin; incidence < end; ++incidence) {
        const std::size_t resource =
            resource_base + static_cast<std::size_t>(runtime.resource_indices[incidence]);
        column_value += runtime.coefficients[incidence] * weights[resource];
      }
      gradients[variable] += column_value;
    }
  }
}

void project_primal(Runtime& runtime, const std::vector<double>& input,
                    std::vector<double>& output) {
  for (std::size_t row = 0; row < runtime.row_count; ++row) {
    const std::size_t begin =
        static_cast<std::size_t>(runtime.row_offsets[row]);
    const double left =
        std::clamp((input[begin] - input[begin + 1] + 1.0) * 0.5, 0.0, 1.0);
    output[begin] = left;
    output[begin + 1] = 1.0 - left;
  }
}

void project_simplex(Runtime& runtime, const std::vector<double>& input,
                     std::size_t input_offset, std::vector<double>& output,
                     std::size_t output_offset, std::size_t width) {
  if (width == 1) {
    output[output_offset] = 1.0;
    return;
  }
  auto& ordered = runtime.projection_scratch;
  std::copy(input.begin() + input_offset,
            input.begin() + input_offset + width, ordered.begin());
  std::sort(ordered.begin(), ordered.begin() + width, std::greater<double>());
  double cumulative = 0.0;
  double threshold = 0.0;
  std::size_t rho = 0;
  for (std::size_t position = 1; position <= width; ++position) {
    cumulative += ordered[position - 1];
    const double candidate =
        (cumulative - 1.0) / static_cast<double>(position);
    if (ordered[position - 1] - candidate > 0.0) {
      rho = position;
      threshold = candidate;
    }
  }
  require(rho > 0, "dual simplex projection found no active coordinate");
  double total = 0.0;
  for (std::size_t i = 0; i < width; ++i) {
    output[output_offset + i] =
        std::max(input[input_offset + i] - threshold, 0.0);
    total += output[output_offset + i];
  }
  require(total > 0.0 && scalar_is_finite(total),
          "dual simplex projection produced an invalid sum");
  for (std::size_t i = 0; i < width; ++i) {
    output[output_offset + i] /= total;
  }
}

void project_dual(Runtime& runtime, const std::vector<double>& input,
                  std::vector<double>& output) {
  for (std::size_t block = 0; block < runtime.block_count; ++block) {
    const auto begin = static_cast<std::size_t>(runtime.dual_offsets[block]);
    const auto end = static_cast<std::size_t>(runtime.dual_offsets[block + 1]);
    project_simplex(runtime, input, begin, output, begin, end - begin);
  }
}

double compute_uniform_reference(Runtime& runtime) {
  apply_operator(runtime, runtime.uniform_shares, runtime.loads_current);
  double reference = runtime.fixed_startup_ms;
  for (std::size_t block = 0; block < runtime.block_count; ++block) {
    const auto begin = static_cast<std::size_t>(runtime.dual_offsets[block]);
    const auto end = static_cast<std::size_t>(runtime.dual_offsets[block + 1]);
    double maximum = runtime.loads_current[begin];
    for (std::size_t i = begin + 1; i < end; ++i) {
      maximum = std::max(maximum, runtime.loads_current[i]);
    }
    reference += maximum;
  }
  if (!(reference > 0.0)) return 1.0;
  require(scalar_is_finite(reference), "uniform objective reference is non-finite");
  return reference;
}

double compute_norm_bound(Runtime& runtime, double reference) {
  double max_column_sum = 0.0;
  for (std::size_t variable = 0; variable < runtime.variable_count; ++variable) {
    double total = 0.0;
    for (std::size_t block = 0; block < runtime.block_count; ++block) {
      const std::size_t column_base = block * (runtime.variable_count + 1);
      const auto begin = static_cast<std::size_t>(
          runtime.column_offsets[column_base + variable]);
      const auto end = static_cast<std::size_t>(
          runtime.column_offsets[column_base + variable + 1]);
      for (std::size_t incidence = begin; incidence < end; ++incidence) {
        total += runtime.coefficients[incidence] / reference;
      }
    }
    max_column_sum = std::max(max_column_sum, total);
  }

  double max_row_sum = 0.0;
  for (std::size_t block = 0; block < runtime.block_count; ++block) {
    const auto begin = static_cast<std::size_t>(runtime.dual_offsets[block]);
    const auto end = static_cast<std::size_t>(runtime.dual_offsets[block + 1]);
    std::fill(runtime.resource_sums.begin() + begin,
              runtime.resource_sums.begin() + end, 0.0);
    const std::size_t column_base = block * (runtime.variable_count + 1);
    for (std::size_t variable = 0; variable < runtime.variable_count; ++variable) {
      const auto col_begin = static_cast<std::size_t>(
          runtime.column_offsets[column_base + variable]);
      const auto col_end = static_cast<std::size_t>(
          runtime.column_offsets[column_base + variable + 1]);
      for (std::size_t incidence = col_begin; incidence < col_end; ++incidence) {
        const std::size_t resource =
            begin + static_cast<std::size_t>(runtime.resource_indices[incidence]);
        runtime.resource_sums[resource] +=
            runtime.coefficients[incidence] / reference;
      }
    }
    double block_maximum = runtime.resource_sums[begin];
    for (std::size_t resource = begin + 1; resource < end; ++resource) {
      block_maximum = std::max(block_maximum, runtime.resource_sums[resource]);
    }
    max_row_sum = std::max(max_row_sum, block_maximum);
  }
  const double product = max_column_sum * max_row_sum;
  require(scalar_is_finite(product) && product >= 0.0, "safe operator norm is invalid");
  return std::sqrt(product);
}

void snapshot(const Runtime& runtime, std::size_t iteration,
              double* output_primal, double* output_dual) {
  const double divisor = static_cast<double>(iteration);
  for (std::size_t i = 0; i < runtime.variable_count; ++i) {
    output_primal[i] = runtime.p_average[i] / divisor;
  }
  for (std::size_t i = 0; i < runtime.resource_count; ++i) {
    output_dual[i] = runtime.dual_average[i] / divisor;
  }
}

void integerize_snapshot(Runtime& runtime, const double* reduced_shares,
                         const std::uint64_t* original_demands,
                         std::int64_t* output_quotas) {
  for (std::size_t row = 0; row < runtime.original_row_count; ++row) {
    const auto begin = static_cast<std::size_t>(runtime.original_row_offsets[row]);
    const auto end = static_cast<std::size_t>(runtime.original_row_offsets[row + 1]);
    const auto width = end - begin;
    const std::uint64_t demand = original_demands[row];
    std::uint64_t floor_sum = 0;
    const std::size_t group = static_cast<std::size_t>(runtime.original_row_groups[row]);
    const std::size_t group_begin =
        static_cast<std::size_t>(runtime.row_offsets[group]);
    for (std::size_t slot = 0; slot < width; ++slot) {
      const std::size_t group_slot = static_cast<std::size_t>(
          runtime.original_slot_to_group_slot[begin + slot]);
      const double share = reduced_shares[group_begin + group_slot];
      const double exact = std::max(0.0, share) * static_cast<double>(demand);
      require(scalar_is_finite(exact) && exact >= 0.0,
              "original-row integerization produced a non-finite share");
      const double floored = std::floor(exact);
      require(floored <= static_cast<double>(demand),
              "original-row integerization floor exceeds row demand");
      runtime.round_floors[begin + slot] =
          static_cast<std::uint64_t>(floored);
      runtime.round_fractions[begin + slot] = exact - floored;
      floor_sum += runtime.round_floors[begin + slot];
      runtime.round_order[slot] = slot;
    }
    require(floor_sum <= demand,
            "original-row integerization has a negative remainder");
    const std::uint64_t remaining = demand - floor_sum;
    require(remaining <= width,
            "original-row integerization remainder exceeds row width");
    std::sort(runtime.round_order.begin(), runtime.round_order.begin() + width,
              [&](std::size_t left, std::size_t right) {
                const double left_fraction = runtime.round_fractions[begin + left];
                const double right_fraction = runtime.round_fractions[begin + right];
                if (left_fraction != right_fraction) return left_fraction > right_fraction;
                return left < right;
              });
    for (std::uint64_t rank = 0; rank < remaining; ++rank) {
      ++runtime.round_floors[begin + runtime.round_order[static_cast<std::size_t>(rank)]];
    }
    for (std::size_t slot = 0; slot < width; ++slot) {
      require(runtime.round_floors[begin + slot] <=
                  static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()),
              "integer quota exceeds signed output range");
      output_quotas[begin + slot] =
          static_cast<std::int64_t>(runtime.round_floors[begin + slot]);
    }
  }
}

}  // namespace

extern "C" MFG_FAST_EXPORT void* mfg_fast_create_v1(
    std::uint64_t row_count, std::uint64_t variable_count,
    std::uint64_t block_count, const std::uint64_t* row_offsets,
    const std::uint64_t* dual_offsets, const std::uint64_t* column_offsets,
    const std::uint64_t* resource_indices, const double* numerators,
    const double* denominators, std::uint64_t incidence_count,
    double fixed_startup_ms, std::uint64_t original_row_count,
    std::uint64_t original_slot_count,
    const std::uint64_t* original_row_offsets,
    const std::uint64_t* original_row_groups,
    const std::uint64_t* original_slot_to_group_slot,
    char* error, std::size_t error_capacity) noexcept {
  try {
    require(row_count > 0 && variable_count > 0 && block_count > 0,
            "native layout dimensions must be positive");
    require(row_offsets && dual_offsets && column_offsets &&
                original_row_offsets && original_row_groups &&
                original_slot_to_group_slot,
            "native layout offset or expansion buffers are null");
    require((incidence_count == 0) ||
                (resource_indices && numerators && denominators),
            "native incidence buffers are null");
    require(scalar_is_finite(fixed_startup_ms) && fixed_startup_ms >= 0.0,
            "fixed startup must be finite and non-negative");
    require(variable_count <= std::numeric_limits<std::size_t>::max() &&
                row_count <= std::numeric_limits<std::size_t>::max() &&
                block_count <= std::numeric_limits<std::size_t>::max() &&
                incidence_count <= std::numeric_limits<std::size_t>::max() &&
                original_row_count <= std::numeric_limits<std::size_t>::max() &&
                original_slot_count <= std::numeric_limits<std::size_t>::max(),
            "native layout exceeds addressable size");
    require(variable_count < std::numeric_limits<std::uint64_t>::max() &&
                block_count <= std::numeric_limits<std::uint64_t>::max() /
                                   (variable_count + 1),
            "native column-offset length overflows");
    require(row_count < std::numeric_limits<std::uint64_t>::max() &&
                block_count < std::numeric_limits<std::uint64_t>::max() &&
                original_row_count < std::numeric_limits<std::uint64_t>::max(),
            "native offset length overflows");
    require(original_row_count > 0 && original_slot_count == original_row_count * 2,
            "native original rows must each have two slots");

    auto runtime = std::make_unique<Runtime>();
    runtime->row_count = static_cast<std::size_t>(row_count);
    runtime->variable_count = static_cast<std::size_t>(variable_count);
    runtime->block_count = static_cast<std::size_t>(block_count);
    runtime->incidence_count = static_cast<std::size_t>(incidence_count);
    runtime->original_row_count = static_cast<std::size_t>(original_row_count);
    runtime->original_slot_count = static_cast<std::size_t>(original_slot_count);
    runtime->fixed_startup_ms = fixed_startup_ms;
    runtime->row_offsets.assign(row_offsets, row_offsets + row_count + 1);
    runtime->dual_offsets.assign(dual_offsets, dual_offsets + block_count + 1);
    runtime->column_offsets.assign(
        column_offsets, column_offsets + block_count * (variable_count + 1));
    runtime->original_row_offsets.assign(
        original_row_offsets, original_row_offsets + original_row_count + 1);
    runtime->original_row_groups.assign(
        original_row_groups, original_row_groups + original_row_count);
    runtime->original_slot_to_group_slot.assign(
        original_slot_to_group_slot,
        original_slot_to_group_slot + original_slot_count);
    if (incidence_count > 0) {
      runtime->resource_indices.assign(resource_indices,
                                       resource_indices + incidence_count);
      runtime->numerators.assign(numerators, numerators + incidence_count);
      runtime->denominators.assign(denominators, denominators + incidence_count);
    }
    require(runtime->dual_offsets.back() <=
                std::numeric_limits<std::size_t>::max(),
            "native resource layout exceeds addressable size");
    runtime->resource_count = static_cast<std::size_t>(
        runtime->dual_offsets.back());
    require(runtime->resource_count > 0,
            "native dual layout has no resources");
    require(runtime->row_offsets.front() == 0 &&
                runtime->row_offsets.back() == variable_count,
            "row offsets do not span the primal layout");
    std::uint64_t expected_variable = 0;
    for (std::size_t row = 0; row < runtime->row_count; ++row) {
      const auto begin = runtime->row_offsets[row];
      const auto end = runtime->row_offsets[row + 1];
      require(begin == expected_variable && end > begin && end <= variable_count &&
                  end - begin == 2,
              "reduced G6 rows must be contiguous two-slot simplexes");
      expected_variable = end;
    }
    require(runtime->dual_offsets.front() == 0 &&
                runtime->resource_count > 0,
            "dual offsets do not define resource blocks");
    for (std::size_t block = 0; block < runtime->block_count; ++block) {
      require(runtime->dual_offsets[block + 1] > runtime->dual_offsets[block],
              "every dual block must contain a resource");
    }
    std::uint64_t expected_offset = 0;
    for (std::size_t block = 0; block < runtime->block_count; ++block) {
      const std::size_t column_base = block * (runtime->variable_count + 1);
      require(runtime->column_offsets[column_base] == expected_offset,
              "column offsets are not contiguous between blocks");
      for (std::size_t variable = 0; variable < runtime->variable_count; ++variable) {
        require(runtime->column_offsets[column_base + variable + 1] >=
                    runtime->column_offsets[column_base + variable],
                "column offsets are not monotone");
      }
      expected_offset = runtime->column_offsets[column_base + runtime->variable_count];
      require(expected_offset <= incidence_count,
              "column offsets exceed the incidence buffer");
      const std::uint64_t local_resources =
          runtime->dual_offsets[block + 1] - runtime->dual_offsets[block];
      for (std::uint64_t i = runtime->column_offsets[column_base];
           i < expected_offset; ++i) {
        require(runtime->resource_indices[static_cast<std::size_t>(i)] <
                    local_resources,
                "incidence names an unknown resource");
        require(scalar_is_finite(runtime->numerators[static_cast<std::size_t>(i)]) &&
                    runtime->numerators[static_cast<std::size_t>(i)] >= 0.0 &&
                    scalar_is_finite(runtime->denominators[static_cast<std::size_t>(i)]) &&
                    runtime->denominators[static_cast<std::size_t>(i)] > 0.0,
                "unit incidence factors must be finite and valid");
      }
    }
    require(expected_offset == incidence_count,
            "column offsets do not span the incidence buffer");

    runtime->row_of_variable.resize(runtime->variable_count);
    for (std::size_t row = 0; row < runtime->row_count; ++row) {
      for (std::size_t variable =
               static_cast<std::size_t>(runtime->row_offsets[row]);
           variable <
               static_cast<std::size_t>(runtime->row_offsets[row + 1]);
           ++variable) {
        runtime->row_of_variable[variable] = row;
      }
    }
    require(runtime->original_row_offsets.front() == 0 &&
                runtime->original_row_offsets.back() == original_slot_count,
            "original row offsets do not span the slot layout");
    for (std::size_t row = 0; row < runtime->original_row_count; ++row) {
      require(runtime->original_row_offsets[row] == row * 2 &&
                  runtime->original_row_offsets[row + 1] == (row + 1) * 2,
              "original G6 rows must have contiguous two-slot masks");
      const auto group = runtime->original_row_groups[row];
      require(group < row_count, "original row maps to an unknown reduced group");
      const auto group_width =
          runtime->row_offsets[group + 1] - runtime->row_offsets[group];
      require(group_width == 2,
              "original row must map to a two-slot reduced group");
      const auto first_mapped =
          runtime->original_slot_to_group_slot[row * 2];
      const auto second_mapped =
          runtime->original_slot_to_group_slot[row * 2 + 1];
      require(first_mapped < group_width && second_mapped < group_width &&
                  first_mapped != second_mapped,
              "original slot map must be a two-slot permutation");
    }
    runtime->round_floors.resize(runtime->original_slot_count);
    runtime->round_fractions.resize(runtime->original_slot_count);
    runtime->round_order.resize(2);
    runtime->demand_sums.resize(runtime->row_count);
    runtime->coefficients.resize(runtime->incidence_count);
    runtime->p_current.resize(runtime->variable_count);
    runtime->p_trial.resize(runtime->variable_count);
    runtime->p_next.resize(runtime->variable_count);
    runtime->p_average.resize(runtime->variable_count);
    runtime->gradient.resize(runtime->variable_count);
    runtime->gradient_trial.resize(runtime->variable_count);
    runtime->dual_current.resize(runtime->resource_count);
    runtime->dual_trial.resize(runtime->resource_count);
    runtime->dual_next.resize(runtime->resource_count);
    runtime->dual_update.resize(runtime->resource_count);
    runtime->dual_average.resize(runtime->resource_count);
    runtime->loads_current.resize(runtime->resource_count);
    runtime->loads_trial.resize(runtime->resource_count);
    runtime->resource_sums.resize(runtime->resource_count);
    std::size_t max_width = 1;
    for (std::size_t row = 0; row < runtime->row_count; ++row) {
      max_width = std::max(max_width, static_cast<std::size_t>(
          runtime->row_offsets[row + 1] - runtime->row_offsets[row]));
    }
    for (std::size_t block = 0; block < runtime->block_count; ++block) {
      max_width = std::max(max_width, static_cast<std::size_t>(
          runtime->dual_offsets[block + 1] - runtime->dual_offsets[block]));
    }
    runtime->projection_scratch.resize(max_width);
    runtime->uniform_shares.resize(runtime->variable_count);
    for (std::size_t row = 0; row < runtime->row_count; ++row) {
      const std::size_t begin =
          static_cast<std::size_t>(runtime->row_offsets[row]);
      const std::size_t width = static_cast<std::size_t>(
          runtime->row_offsets[row + 1] - runtime->row_offsets[row]);
      const double uniform = 1.0 / static_cast<double>(width);
      std::fill(runtime->uniform_shares.begin() + begin,
                runtime->uniform_shares.begin() + begin + width, uniform);
    }
    copy_error(error, error_capacity, "");
    return runtime.release();
  } catch (const std::exception& exc) {
    copy_error(error, error_capacity, exc.what());
    return nullptr;
  } catch (...) {
    copy_error(error, error_capacity, "unknown native layout error");
    return nullptr;
  }
}

extern "C" MFG_FAST_EXPORT void mfg_fast_destroy_v1(void* opaque) noexcept {
  delete static_cast<Runtime*>(opaque);
}

extern "C" MFG_FAST_EXPORT int mfg_fast_run_v1(
    void* opaque, const double* reduced_demands,
    const std::uint64_t* original_demands, const double* initial_primal,
    const double* initial_dual, const std::uint64_t* step_counts,
    std::uint64_t step_count, double* output_primal, double* output_dual,
    std::int64_t* output_quotas, double* output_reference_ms,
    double* output_norm_bound, char* error, std::size_t error_capacity) noexcept {
  try {
    auto* runtime = static_cast<Runtime*>(opaque);
    require(runtime != nullptr, "native runtime is closed");
    require(reduced_demands && original_demands && initial_primal &&
                initial_dual && step_counts && output_primal && output_dual &&
                output_quotas && output_reference_ms && output_norm_bound,
            "native call buffer is null");
    require(step_count > 0 &&
                step_count <= std::numeric_limits<std::size_t>::max(),
            "native call needs one or more step budgets");
    std::uint64_t previous_step = 0;
    for (std::size_t i = 0; i < static_cast<std::size_t>(step_count); ++i) {
      require(i == 0 || step_counts[i] > previous_step,
              "step budgets must be strictly increasing");
      require(step_counts[i] <= 1000000,
              "fixed step budget exceeds the native safety limit");
      previous_step = step_counts[i];
    }
    constexpr std::uint64_t max_safe_count = (std::uint64_t{1} << 53) - 1;
    std::fill(runtime->demand_sums.begin(), runtime->demand_sums.end(), 0);
    std::uint64_t global_total = 0;
    for (std::size_t row = 0; row < runtime->original_row_count; ++row) {
      const std::uint64_t demand = original_demands[row];
      require(demand <= max_safe_count &&
                  global_total <= max_safe_count - demand,
              "original demand exceeds the frozen exact-count range");
      global_total += demand;
      const auto group = static_cast<std::size_t>(
          runtime->original_row_groups[row]);
      require(runtime->demand_sums[group] <= max_safe_count - demand,
              "reduced group demand exceeds the frozen exact-count range");
      runtime->demand_sums[group] += demand;
    }
    for (std::size_t row = 0; row < runtime->row_count; ++row) {
      require(scalar_is_finite(reduced_demands[row]) && reduced_demands[row] >= 0.0 &&
                  reduced_demands[row] ==
                      static_cast<double>(runtime->demand_sums[row]),
              "reduced demands do not match the current original-row counts");
    }
    validate_simplexes(*runtime, initial_primal, initial_dual);

    for (std::size_t block = 0; block < runtime->block_count; ++block) {
      const std::size_t column_base = block * (runtime->variable_count + 1);
      for (std::size_t current_variable = 0;
           current_variable < runtime->variable_count; ++current_variable) {
        const auto begin = static_cast<std::size_t>(
            runtime->column_offsets[column_base + current_variable]);
        const auto end = static_cast<std::size_t>(
            runtime->column_offsets[column_base + current_variable + 1]);
        for (std::size_t incidence = begin; incidence < end; ++incidence) {
          const double demand =
              reduced_demands[runtime->row_of_variable[current_variable]];
          runtime->coefficients[incidence] =
              demand * runtime->numerators[incidence] /
              runtime->denominators[incidence];
        }
      }
    }

    const double reference = compute_uniform_reference(*runtime);
    const double norm_bound = compute_norm_bound(*runtime, reference);
    require(scalar_is_finite(norm_bound), "safe operator norm is non-finite");
    const double gamma = norm_bound == 0.0 ? 0.0 : 0.5 / norm_bound;
    const double step_scale = gamma / reference;
    require(scalar_is_finite(step_scale), "fixed Mirror-Prox step is non-finite");
    *output_reference_ms = reference;
    *output_norm_bound = norm_bound;

    std::copy(initial_primal, initial_primal + runtime->variable_count,
              runtime->p_current.begin());
    std::copy(initial_dual, initial_dual + runtime->resource_count,
              runtime->dual_current.begin());
    std::fill(runtime->p_average.begin(), runtime->p_average.end(), 0.0);
    std::fill(runtime->dual_average.begin(), runtime->dual_average.end(), 0.0);

    std::size_t next_snapshot = 0;
    if (step_counts[0] == 0) {
      double* out_p = output_primal;
      double* out_dual = output_dual;
      std::copy(runtime->p_current.begin(), runtime->p_current.end(), out_p);
      std::copy(runtime->dual_current.begin(), runtime->dual_current.end(), out_dual);
      integerize_snapshot(
          *runtime, out_p, original_demands,
          output_quotas);
      ++next_snapshot;
    }
    const std::uint64_t maximum_step = step_counts[step_count - 1];
    for (std::uint64_t iteration = 1; iteration <= maximum_step; ++iteration) {
      apply_adjoint(*runtime, runtime->dual_current, runtime->gradient);
      for (std::size_t i = 0; i < runtime->variable_count; ++i) {
        runtime->p_trial[i] =
            runtime->p_current[i] - step_scale * runtime->gradient[i];
      }
      project_primal(*runtime, runtime->p_trial, runtime->p_trial);

      apply_operator(*runtime, runtime->p_current, runtime->loads_current);
      for (std::size_t i = 0; i < runtime->resource_count; ++i) {
        runtime->dual_update[i] =
            runtime->dual_current[i] + step_scale * runtime->loads_current[i];
      }
      project_dual(*runtime, runtime->dual_update, runtime->dual_trial);

      apply_adjoint(*runtime, runtime->dual_trial, runtime->gradient_trial);
      for (std::size_t i = 0; i < runtime->variable_count; ++i) {
        runtime->p_next[i] =
            runtime->p_current[i] - step_scale * runtime->gradient_trial[i];
      }
      project_primal(*runtime, runtime->p_next, runtime->p_next);

      apply_operator(*runtime, runtime->p_trial, runtime->loads_trial);
      for (std::size_t i = 0; i < runtime->resource_count; ++i) {
        runtime->dual_update[i] =
            runtime->dual_current[i] + step_scale * runtime->loads_trial[i];
      }
      project_dual(*runtime, runtime->dual_update, runtime->dual_next);

      for (std::size_t i = 0; i < runtime->variable_count; ++i) {
        runtime->p_average[i] += runtime->p_trial[i];
      }
      for (std::size_t i = 0; i < runtime->resource_count; ++i) {
        runtime->dual_average[i] += runtime->dual_trial[i];
      }
      runtime->p_current.swap(runtime->p_next);
      runtime->dual_current.swap(runtime->dual_next);

      if (next_snapshot < static_cast<std::size_t>(step_count) &&
          iteration == step_counts[next_snapshot]) {
        const std::size_t output_offset =
            next_snapshot * runtime->variable_count;
        const std::size_t dual_output_offset =
            next_snapshot * runtime->resource_count;
        snapshot(*runtime, static_cast<std::size_t>(iteration),
                 output_primal + output_offset,
                 output_dual + dual_output_offset);
        integerize_snapshot(
            *runtime, output_primal + output_offset, original_demands,
            output_quotas + next_snapshot * runtime->original_slot_count);
        ++next_snapshot;
      }
    }
    require(next_snapshot == static_cast<std::size_t>(step_count),
            "not every fixed budget produced an output");
    copy_error(error, error_capacity, "");
    return 0;
  } catch (const std::exception& exc) {
    copy_error(error, error_capacity, exc.what());
    return 1;
  } catch (...) {
    copy_error(error, error_capacity, "unknown native Mirror-Prox error");
    return 2;
  }
}
