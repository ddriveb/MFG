// Float64 native implementation of the frozen v2 projected Mirror-Prox loop.
// The Python model owns validation, sparse construction, normalization and
// independent certificate replay. This C ABI accepts one packed snapshot.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(_WIN32)
#define MFG_EXPORT __declspec(dllexport)
#else
#define MFG_EXPORT __attribute__((visibility("default")))
#endif

namespace {

enum class Status : int {
  converged = 0,
  iteration_cap = 1,
  certificate_error = 2,
  line_search_exhausted = 3
};

struct ProblemView {
  std::uint64_t row_count;
  std::uint64_t variable_count;
  std::uint64_t block_count;
  const std::uint64_t* row_offsets;
  const std::uint64_t* dual_offsets;
  const std::uint64_t* column_offsets;
  const std::uint64_t* resource_indices;
  const double* coefficients;
  double normalization_ms;
  double fixed_startup_ms;
};

std::vector<double> project_simplex(const std::vector<double>& values) {
  if (values.empty()) {
    throw std::runtime_error("simplex projection requires at least one coordinate");
  }
  if (values.size() == 1) {
    return {1.0};
  }

  std::vector<double> ordered(values);
  std::sort(ordered.begin(), ordered.end(), std::greater<double>());
  double cumulative = 0.0;
  double threshold = 0.0;
  std::size_t rho = 0;
  for (std::size_t position = 1; position <= ordered.size(); ++position) {
    cumulative += ordered[position - 1];
    const double candidate = (cumulative - 1.0) / static_cast<double>(position);
    if (ordered[position - 1] - candidate > 0.0) {
      rho = position;
      threshold = candidate;
    }
  }
  if (rho == 0) {
    throw std::runtime_error("simplex projection failed to find an active coordinate");
  }

  std::vector<double> projected(values.size(), 0.0);
  double total = 0.0;
  for (std::size_t index = 0; index < values.size(); ++index) {
    projected[index] = std::max(values[index] - threshold, 0.0);
    total += projected[index];
  }
  if (total <= 0.0) {
    const double uniform = 1.0 / static_cast<double>(values.size());
    std::fill(projected.begin(), projected.end(), uniform);
  } else {
    for (double& value : projected) {
      value /= total;
    }
  }
  return projected;
}

std::vector<double> project_product_simplexes(const ProblemView& problem,
                                              const std::vector<double>& values) {
  std::vector<double> projected(problem.variable_count, 0.0);
  for (std::uint64_t row = 0; row < problem.row_count; ++row) {
    const auto start = problem.row_offsets[row];
    const auto end = problem.row_offsets[row + 1];
    std::vector<double> row_values(values.begin() + start, values.begin() + end);
    const auto row_projection = project_simplex(row_values);
    std::copy(row_projection.begin(), row_projection.end(), projected.begin() + start);
  }
  return projected;
}

std::vector<double> apply_operator(const ProblemView& problem,
                                   const std::vector<double>& shares) {
  const auto resource_count = problem.dual_offsets[problem.block_count];
  std::vector<double> loads(resource_count, 0.0);
  for (std::uint64_t block = 0; block < problem.block_count; ++block) {
    const auto column_base = block * (problem.variable_count + 1);
    const auto resource_base = problem.dual_offsets[block];
    for (std::uint64_t variable = 0; variable < problem.variable_count; ++variable) {
      const double share = shares[variable];
      if (share == 0.0) {
        continue;
      }
      const auto start = problem.column_offsets[column_base + variable];
      const auto end = problem.column_offsets[column_base + variable + 1];
      for (auto incidence = start; incidence < end; ++incidence) {
        const auto resource = resource_base + problem.resource_indices[incidence];
        loads[resource] += problem.coefficients[incidence] * share;
      }
    }
  }
  return loads;
}

std::vector<double> apply_adjoint(const ProblemView& problem,
                                  const std::vector<double>& weights) {
  std::vector<double> gradients(problem.variable_count, 0.0);
  for (std::uint64_t block = 0; block < problem.block_count; ++block) {
    const auto column_base = block * (problem.variable_count + 1);
    const auto resource_base = problem.dual_offsets[block];
    for (std::uint64_t variable = 0; variable < problem.variable_count; ++variable) {
      double column_value = 0.0;
      const auto start = problem.column_offsets[column_base + variable];
      const auto end = problem.column_offsets[column_base + variable + 1];
      for (auto incidence = start; incidence < end; ++incidence) {
        const auto resource = resource_base + problem.resource_indices[incidence];
        column_value += problem.coefficients[incidence] * weights[resource];
      }
      gradients[variable] += column_value;
    }
  }
  return gradients;
}

std::vector<double> initialize_dual(const ProblemView& problem,
                                    const std::vector<double>& initial_shares,
                                    int initialization_mode) {
  const auto loads = apply_operator(problem, initial_shares);
  std::vector<double> weights(problem.dual_offsets[problem.block_count], 0.0);
  for (std::uint64_t block = 0; block < problem.block_count; ++block) {
    const auto start = problem.dual_offsets[block];
    const auto end = problem.dual_offsets[block + 1];
    const auto maximum = *std::max_element(loads.begin() + start, loads.begin() + end);
    if (initialization_mode == 0) {
      for (auto resource = start; resource < end; ++resource) {
        if (loads[resource] == maximum) {
          weights[resource] = 1.0;
          break;
        }
      }
    } else {
      std::uint64_t ties = 0;
      for (auto resource = start; resource < end; ++resource) {
        if (loads[resource] == maximum) {
          ++ties;
        }
      }
      if (ties == 0) {
        throw std::runtime_error("tied maximum initialization found no maximum resource");
      }
      const double weight = 1.0 / static_cast<double>(ties);
      for (auto resource = start; resource < end; ++resource) {
        if (loads[resource] == maximum) {
          weights[resource] = weight;
        }
      }
    }
  }
  return weights;
}

std::vector<double> project_dual(const ProblemView& problem,
                                 const std::vector<double>& values) {
  std::vector<double> projected(values.size(), 0.0);
  for (std::uint64_t block = 0; block < problem.block_count; ++block) {
    const auto start = problem.dual_offsets[block];
    const auto end = problem.dual_offsets[block + 1];
    std::vector<double> block_values(values.begin() + start, values.begin() + end);
    const auto block_projection = project_simplex(block_values);
    std::copy(block_projection.begin(), block_projection.end(), projected.begin() + start);
  }
  return projected;
}

double objective_value(const ProblemView& problem, const std::vector<double>& shares) {
  const auto loads = apply_operator(problem, shares);
  double total = problem.fixed_startup_ms;
  for (std::uint64_t block = 0; block < problem.block_count; ++block) {
    const auto start = problem.dual_offsets[block];
    const auto end = problem.dual_offsets[block + 1];
    total += *std::max_element(loads.begin() + start, loads.begin() + end);
  }
  return total;
}

double dual_lower_bound(const ProblemView& problem, const std::vector<double>& weights) {
  const auto gradients = apply_adjoint(problem, weights);
  double bound = problem.fixed_startup_ms;
  for (std::uint64_t row = 0; row < problem.row_count; ++row) {
    const auto start = problem.row_offsets[row];
    const auto end = problem.row_offsets[row + 1];
    bound += *std::min_element(gradients.begin() + start, gradients.begin() + end);
  }
  return bound;
}

struct Workspace {
  std::vector<double> current_loads;
  std::vector<double> trial_loads;
  std::vector<double> candidate_loads;
  std::vector<double> current_gradient;
  std::vector<double> trial_gradient;
  std::vector<double> candidate_gradient;
  std::vector<double> trial_share_values;
  std::vector<double> trial_shares;
  std::vector<double> trial_dual_values;
  std::vector<double> trial_dual;
  std::vector<double> next_share_values;
  std::vector<double> next_shares;
  std::vector<double> next_dual_values;
  std::vector<double> next_dual;
  std::vector<double> average_share_sums;
  std::vector<double> average_dual_sums;
  std::vector<double> averaged_shares;
  std::vector<double> averaged_dual;
  std::vector<double> sorted_simplex;

  Workspace(const ProblemView& problem, std::uint64_t total_resources)
      : current_loads(total_resources, 0.0),
        trial_loads(total_resources, 0.0),
        candidate_loads(total_resources, 0.0),
        current_gradient(problem.variable_count, 0.0),
        trial_gradient(problem.variable_count, 0.0),
        candidate_gradient(problem.variable_count, 0.0),
        trial_share_values(problem.variable_count, 0.0),
        trial_shares(problem.variable_count, 0.0),
        trial_dual_values(total_resources, 0.0),
        trial_dual(total_resources, 0.0),
        next_share_values(problem.variable_count, 0.0),
        next_shares(problem.variable_count, 0.0),
        next_dual_values(total_resources, 0.0),
        next_dual(total_resources, 0.0),
        average_share_sums(problem.variable_count, 0.0),
        average_dual_sums(total_resources, 0.0),
        averaged_shares(problem.variable_count, 0.0),
        averaged_dual(total_resources, 0.0),
        sorted_simplex(max_simplex_width(problem), 0.0) {}

  static std::uint64_t max_simplex_width(const ProblemView& problem) {
    std::uint64_t width = 1;
    for (std::uint64_t row = 0; row < problem.row_count; ++row) {
      width = std::max(width, problem.row_offsets[row + 1] - problem.row_offsets[row]);
    }
    for (std::uint64_t block = 0; block < problem.block_count; ++block) {
      width = std::max(width, problem.dual_offsets[block + 1] - problem.dual_offsets[block]);
    }
    return width;
  }
};

void apply_operator_into(const ProblemView& problem,
                         const std::vector<double>& shares,
                         std::vector<double>& loads) {
  std::fill(loads.begin(), loads.end(), 0.0);
  for (std::uint64_t block = 0; block < problem.block_count; ++block) {
    const auto column_base = block * (problem.variable_count + 1);
    const auto resource_base = problem.dual_offsets[block];
    for (std::uint64_t variable = 0; variable < problem.variable_count; ++variable) {
      const double share = shares[variable];
      if (share == 0.0) {
        continue;
      }
      const auto start = problem.column_offsets[column_base + variable];
      const auto end = problem.column_offsets[column_base + variable + 1];
      for (auto incidence = start; incidence < end; ++incidence) {
        loads[resource_base + problem.resource_indices[incidence]] +=
            problem.coefficients[incidence] * share;
      }
    }
  }
}

void apply_adjoint_into(const ProblemView& problem,
                        const std::vector<double>& weights,
                        std::vector<double>& gradients) {
  std::fill(gradients.begin(), gradients.end(), 0.0);
  for (std::uint64_t block = 0; block < problem.block_count; ++block) {
    const auto column_base = block * (problem.variable_count + 1);
    const auto resource_base = problem.dual_offsets[block];
    for (std::uint64_t variable = 0; variable < problem.variable_count; ++variable) {
      double column_value = 0.0;
      const auto start = problem.column_offsets[column_base + variable];
      const auto end = problem.column_offsets[column_base + variable + 1];
      for (auto incidence = start; incidence < end; ++incidence) {
        column_value += problem.coefficients[incidence] *
            weights[resource_base + problem.resource_indices[incidence]];
      }
      gradients[variable] += column_value;
    }
  }
}

void project_simplex_segment_into(const std::vector<double>& values,
                                  std::uint64_t value_start,
                                  std::uint64_t width,
                                  std::vector<double>& output,
                                  std::uint64_t output_start,
                                  std::vector<double>& sorted) {
  if (width == 0 || width > sorted.size()) {
    throw std::runtime_error("invalid simplex projection width");
  }
  if (width == 1) {
    output[output_start] = 1.0;
    return;
  }
  std::copy(values.begin() + value_start, values.begin() + value_start + width,
            sorted.begin());
  std::sort(sorted.begin(), sorted.begin() + width, std::greater<double>());
  double cumulative = 0.0;
  double threshold = 0.0;
  std::uint64_t rho = 0;
  for (std::uint64_t position = 1; position <= width; ++position) {
    cumulative += sorted[position - 1];
    const double candidate = (cumulative - 1.0) / static_cast<double>(position);
    if (sorted[position - 1] - candidate > 0.0) {
      rho = position;
      threshold = candidate;
    }
  }
  if (rho == 0) {
    throw std::runtime_error("simplex projection failed to find an active coordinate");
  }
  double total = 0.0;
  for (std::uint64_t index = 0; index < width; ++index) {
    const double projected = std::max(values[value_start + index] - threshold, 0.0);
    output[output_start + index] = projected;
    total += projected;
  }
  if (total <= 0.0) {
    const double uniform = 1.0 / static_cast<double>(width);
    for (std::uint64_t index = 0; index < width; ++index) {
      output[output_start + index] = uniform;
    }
  } else {
    for (std::uint64_t index = 0; index < width; ++index) {
      output[output_start + index] /= total;
    }
  }
}

void project_product_simplexes_into(const ProblemView& problem,
                                    const std::vector<double>& values,
                                    std::vector<double>& output,
                                    std::vector<double>& sorted) {
  for (std::uint64_t row = 0; row < problem.row_count; ++row) {
    const auto start = problem.row_offsets[row];
    const auto width = problem.row_offsets[row + 1] - start;
    project_simplex_segment_into(values, start, width, output, start, sorted);
  }
}

void project_dual_into(const ProblemView& problem,
                       const std::vector<double>& values,
                       std::vector<double>& output,
                       std::vector<double>& sorted) {
  for (std::uint64_t block = 0; block < problem.block_count; ++block) {
    const auto start = problem.dual_offsets[block];
    const auto width = problem.dual_offsets[block + 1] - start;
    project_simplex_segment_into(values, start, width, output, start, sorted);
  }
}

double objective_value_into(const ProblemView& problem,
                            const std::vector<double>& shares,
                            Workspace& workspace) {
  apply_operator_into(problem, shares, workspace.candidate_loads);
  double total = problem.fixed_startup_ms;
  for (std::uint64_t block = 0; block < problem.block_count; ++block) {
    const auto start = problem.dual_offsets[block];
    const auto end = problem.dual_offsets[block + 1];
    total += *std::max_element(workspace.candidate_loads.begin() + start,
                               workspace.candidate_loads.begin() + end);
  }
  return total;
}

double dual_lower_bound_into(const ProblemView& problem,
                             const std::vector<double>& weights,
                             Workspace& workspace) {
  apply_adjoint_into(problem, weights, workspace.candidate_gradient);
  double bound = problem.fixed_startup_ms;
  for (std::uint64_t row = 0; row < problem.row_count; ++row) {
    const auto start = problem.row_offsets[row];
    const auto end = problem.row_offsets[row + 1];
    bound += *std::min_element(workspace.candidate_gradient.begin() + start,
                               workspace.candidate_gradient.begin() + end);
  }
  return bound;
}

void initialize_dual_into(const ProblemView& problem,
                           const std::vector<double>& initial_shares,
                           std::vector<double>& dual,
                           std::vector<double>& loads,
                           int initialization_mode) {
  apply_operator_into(problem, initial_shares, loads);
  std::fill(dual.begin(), dual.end(), 0.0);
  for (std::uint64_t block = 0; block < problem.block_count; ++block) {
    const auto start = problem.dual_offsets[block];
    const auto end = problem.dual_offsets[block + 1];
    const double maximum = *std::max_element(loads.begin() + start, loads.begin() + end);
    if (initialization_mode == 0) {
      for (auto resource = start; resource < end; ++resource) {
        if (loads[resource] == maximum) {
          dual[resource] = 1.0;
          break;
        }
      }
    } else {
      std::uint64_t tie_count = 0;
      for (auto resource = start; resource < end; ++resource) {
        tie_count += loads[resource] == maximum ? 1 : 0;
      }
      const double weight = 1.0 / static_cast<double>(tie_count);
      for (auto resource = start; resource < end; ++resource) {
        if (loads[resource] == maximum) {
          dual[resource] = weight;
        }
      }
    }
  }
}

double local_condition_delta(const ProblemView& problem,
                             const std::vector<double>& current_shares,
                             const std::vector<double>& current_dual,
                             const std::vector<double>& trial_shares,
                             const std::vector<double>& trial_dual,
                             const std::vector<double>& next_shares,
                             const std::vector<double>& next_dual,
                             const std::vector<double>& current_gradient,
                             const std::vector<double>& trial_gradient,
                             const std::vector<double>& current_loads,
                             const std::vector<double>& trial_loads,
                             double normalization_ms,
                             double gamma,
                             double* out_slack) {
  double inner_product = 0.0;
  double trial_current_norm_squared = 0.0;
  double trial_next_norm_squared = 0.0;
  for (std::uint64_t variable = 0; variable < problem.variable_count; ++variable) {
    const double f_difference =
        (trial_gradient[variable] - current_gradient[variable]) / normalization_ms;
    const double trial_current = trial_shares[variable] - current_shares[variable];
    const double trial_next = trial_shares[variable] - next_shares[variable];
    inner_product += f_difference * trial_next;
    trial_current_norm_squared += trial_current * trial_current;
    trial_next_norm_squared += trial_next * trial_next;
  }
  for (std::uint64_t block = 0; block < problem.block_count; ++block) {
    const auto start = problem.dual_offsets[block];
    const auto end = problem.dual_offsets[block + 1];
    for (auto resource = start; resource < end; ++resource) {
      const double f_difference =
          -(trial_loads[resource] - current_loads[resource]) / normalization_ms;
      const double trial_current = trial_dual[resource] - current_dual[resource];
      const double trial_next = trial_dual[resource] - next_dual[resource];
      inner_product += f_difference * trial_next;
      trial_current_norm_squared += trial_current * trial_current;
      trial_next_norm_squared += trial_next * trial_next;
    }
  }
  const double first_half = 0.5 * trial_current_norm_squared;
  const double second_half = 0.5 * trial_next_norm_squared;
  const double scaled_inner = gamma * inner_product;
  const double delta = scaled_inner - first_half - second_half;
  const double magnitude = std::max(1.0, std::abs(scaled_inner) + first_half + second_half);
  const auto dimension = std::max<std::uint64_t>(
      1, problem.variable_count + problem.dual_offsets[problem.block_count]);
  *out_slack = 64.0 * std::numeric_limits<double>::epsilon() *
      static_cast<double>(dimension) * magnitude;
  return delta;
}

double gap_threshold(double absolute_gap_ms, double relative_gap, double upper,
                     double lower) {
  return absolute_gap_ms + relative_gap * std::max(std::abs(upper), std::abs(lower));
}

void copy_error(const std::string& message, char* error_buffer,
                std::size_t error_capacity) {
  if (error_buffer == nullptr || error_capacity == 0) {
    return;
  }
  const auto count = std::min(message.size(), error_capacity - 1);
  std::memcpy(error_buffer, message.data(), count);
  error_buffer[count] = '\0';
}

}  // namespace

extern "C" MFG_EXPORT int mfg_mirror_prox_solve_v1(
    std::uint64_t row_count, std::uint64_t variable_count,
    std::uint64_t block_count, const std::uint64_t* row_offsets,
    const std::uint64_t* dual_offsets, const std::uint64_t* column_offsets,
    const std::uint64_t* resource_indices, const double* coefficients,
    const double* initial_shares, double normalization_ms,
    double fixed_startup_ms, double operator_norm_bound,
    std::uint64_t max_iterations, double absolute_gap_ms, double relative_gap,
    int initialization_mode, int initial_certificate_check,
    double* output_shares, double* output_dual_weights, int* output_status,
    std::uint64_t* output_iterations, double* output_upper_bound_ms,
    double* output_lower_bound_ms, double* output_gap_ms, double* output_step_size,
    double* output_operator_norm_bound, double* output_elapsed_ms,
    double* output_initial_gap_ms, int* output_initial_certificate_passed,
    char* error_buffer, std::size_t error_capacity) {
  try {
    if (row_count == 0 || variable_count == 0 || block_count == 0 ||
        row_offsets == nullptr || dual_offsets == nullptr || column_offsets == nullptr ||
        initial_shares == nullptr || output_shares == nullptr ||
        output_dual_weights == nullptr || output_status == nullptr ||
        output_iterations == nullptr || output_upper_bound_ms == nullptr ||
        output_lower_bound_ms == nullptr || output_gap_ms == nullptr ||
        output_step_size == nullptr || output_operator_norm_bound == nullptr ||
        output_elapsed_ms == nullptr || output_initial_gap_ms == nullptr ||
        output_initial_certificate_passed == nullptr) {
      throw std::runtime_error("native solver received a null required pointer or empty shape");
    }
    if (normalization_ms <= 0.0 || !std::isfinite(normalization_ms) ||
        !std::isfinite(fixed_startup_ms) || operator_norm_bound < 0.0 ||
        !std::isfinite(operator_norm_bound) || max_iterations == 0 ||
        absolute_gap_ms < 0.0 || !std::isfinite(absolute_gap_ms) ||
        relative_gap < 0.0 || !std::isfinite(relative_gap) ||
        (initialization_mode != 0 && initialization_mode != 1)) {
      throw std::runtime_error("native solver received invalid scalar parameters");
    }
    const auto total_resources = dual_offsets[block_count];
    if (total_resources == 0) {
      throw std::runtime_error("native solver requires at least one dual resource");
    }
    for (std::uint64_t row = 0; row < row_count; ++row) {
      if (row_offsets[row] >= row_offsets[row + 1] || row_offsets[row + 1] > variable_count) {
        throw std::runtime_error("invalid row simplex offsets");
      }
    }
    for (std::uint64_t block = 0; block < block_count; ++block) {
      if (dual_offsets[block] >= dual_offsets[block + 1]) {
        throw std::runtime_error("invalid dual simplex offsets");
      }
    }

    ProblemView problem{row_count, variable_count, block_count, row_offsets,
                        dual_offsets, column_offsets, resource_indices, coefficients,
                        normalization_ms, fixed_startup_ms};
    std::vector<double> primal(initial_shares, initial_shares + variable_count);
    auto current_dual = initialize_dual(problem, primal, initialization_mode);
    const double step_size = operator_norm_bound == 0.0 ? 0.0 : 0.5 / operator_norm_bound;
    *output_step_size = step_size;
    *output_operator_norm_bound = operator_norm_bound;
    *output_initial_certificate_passed = 0;

    if (operator_norm_bound == 0.0) {
      const double upper = objective_value(problem, primal);
      const double lower = dual_lower_bound(problem, current_dual);
      *output_status = static_cast<int>(Status::converged);
      *output_iterations = 0;
      *output_upper_bound_ms = upper;
      *output_lower_bound_ms = lower;
      *output_gap_ms = upper - lower;
      *output_elapsed_ms = 0.0;
      *output_initial_gap_ms = upper - lower;
      std::copy(primal.begin(), primal.end(), output_shares);
      std::copy(current_dual.begin(), current_dual.end(), output_dual_weights);
      return 0;
    }

    const auto start_time = std::chrono::steady_clock::now();
    std::vector<double> average_share_sums(variable_count, 0.0);
    std::vector<double> average_dual_sums(total_resources, 0.0);
    std::vector<double> best_shares = primal;
    double best_upper = objective_value(problem, best_shares);
    std::vector<double> best_dual = current_dual;
    double best_lower = dual_lower_bound(problem, best_dual);
    double initial_gap = best_upper - best_lower;
    bool initial_passed = false;
    bool certificate_error = false;
    Status status = Status::iteration_cap;
    std::uint64_t iterations = 0;
    const auto checkpoint_interval = std::max<std::uint64_t>(
        1, std::min<std::uint64_t>(50, max_iterations / 100));

    if (initial_certificate_check != 0 &&
        initial_gap <= gap_threshold(absolute_gap_ms, relative_gap,
                                     best_upper, best_lower)) {
      initial_passed = true;
      status = Status::converged;
    } else {
      for (std::uint64_t iteration = 1; iteration <= max_iterations; ++iteration) {
        const auto current_gradient = apply_adjoint(problem, current_dual);
        std::vector<double> trial_share_values(variable_count, 0.0);
        for (std::uint64_t variable = 0; variable < variable_count; ++variable) {
          trial_share_values[variable] = primal[variable] -
              step_size * current_gradient[variable] / normalization_ms;
        }
        auto trial_shares = project_product_simplexes(problem, trial_share_values);

        const auto current_loads = apply_operator(problem, primal);
        std::vector<double> trial_dual_values(total_resources, 0.0);
        for (std::uint64_t index = 0; index < total_resources; ++index) {
          trial_dual_values[index] = current_dual[index] +
              step_size * current_loads[index] / normalization_ms;
        }
        auto trial_dual = project_dual(problem, trial_dual_values);

        const auto trial_gradient = apply_adjoint(problem, trial_dual);
        std::vector<double> next_share_values(variable_count, 0.0);
        for (std::uint64_t variable = 0; variable < variable_count; ++variable) {
          next_share_values[variable] = primal[variable] -
              step_size * trial_gradient[variable] / normalization_ms;
        }
        auto next_shares = project_product_simplexes(problem, next_share_values);

        const auto trial_loads = apply_operator(problem, trial_shares);
        std::vector<double> next_dual_values(total_resources, 0.0);
        for (std::uint64_t index = 0; index < total_resources; ++index) {
          next_dual_values[index] = current_dual[index] +
              step_size * trial_loads[index] / normalization_ms;
        }
        auto next_dual = project_dual(problem, next_dual_values);

        for (std::uint64_t index = 0; index < variable_count; ++index) {
          average_share_sums[index] += trial_shares[index];
        }
        for (std::uint64_t index = 0; index < total_resources; ++index) {
          average_dual_sums[index] += trial_dual[index];
        }
        primal = next_shares;
        current_dual = next_dual;
        iterations = iteration;

        if (iteration % checkpoint_interval == 0 || iteration == max_iterations) {
          std::vector<double> averaged_shares(variable_count, 0.0);
          for (std::uint64_t index = 0; index < variable_count; ++index) {
            averaged_shares[index] = average_share_sums[index] /
                                     static_cast<double>(iteration);
          }
          std::vector<double> averaged_dual(total_resources, 0.0);
          for (std::uint64_t index = 0; index < total_resources; ++index) {
            averaged_dual[index] = average_dual_sums[index] /
                                   static_cast<double>(iteration);
          }

          const std::vector<const std::vector<double>*> primal_candidates = {
              &primal, &trial_shares, &next_shares, &averaged_shares};
          for (const auto* candidate : primal_candidates) {
            const double candidate_upper = objective_value(problem, *candidate);
            if (candidate_upper < best_upper) {
              best_upper = candidate_upper;
              best_shares = *candidate;
            }
          }
          const std::vector<const std::vector<double>*> dual_candidates = {
              &current_dual, &trial_dual, &next_dual, &averaged_dual};
          for (const auto* candidate : dual_candidates) {
            const double candidate_lower = dual_lower_bound(problem, *candidate);
            if (candidate_lower > best_lower) {
              best_lower = candidate_lower;
              best_dual = *candidate;
            }
          }

          const double gap = best_upper - best_lower;
          const double numeric_tolerance = 1e-10 *
              std::max({1.0, std::abs(best_upper), std::abs(best_lower)});
          if (gap < -numeric_tolerance) {
            certificate_error = true;
            status = Status::certificate_error;
            break;
          }
          if (gap <= gap_threshold(absolute_gap_ms, relative_gap,
                                  best_upper, best_lower)) {
            status = Status::converged;
            break;
          }
        }
      }
      if (!certificate_error && status != Status::converged) {
        status = Status::iteration_cap;
      }
    }

    const auto end_time = std::chrono::steady_clock::now();
    const double elapsed_ms = std::chrono::duration<double, std::milli>(
        end_time - start_time).count();
    *output_status = static_cast<int>(status);
    *output_iterations = iterations;
    *output_upper_bound_ms = best_upper;
    *output_lower_bound_ms = best_lower;
    *output_gap_ms = best_upper - best_lower;
    *output_elapsed_ms = elapsed_ms;
    *output_initial_gap_ms = initial_gap;
    *output_initial_certificate_passed = initial_passed ? 1 : 0;
    std::copy(best_shares.begin(), best_shares.end(), output_shares);
    std::copy(best_dual.begin(), best_dual.end(), output_dual_weights);
    return 0;
  } catch (const std::exception& error) {
    copy_error(error.what(), error_buffer, error_capacity);
    return 1;
  } catch (...) {
    copy_error("unknown native Mirror-Prox failure", error_buffer, error_capacity);
    return 2;
  }
}

extern "C" MFG_EXPORT int mfg_mirror_prox_solve_adaptive_v1(
    std::uint64_t row_count, std::uint64_t variable_count,
    std::uint64_t block_count, const std::uint64_t* row_offsets,
    const std::uint64_t* dual_offsets, const std::uint64_t* column_offsets,
    const std::uint64_t* resource_indices, const double* coefficients,
    const double* initial_shares, double normalization_ms,
    double fixed_startup_ms, double operator_norm_bound,
    std::uint64_t max_iterations, double absolute_gap_ms, double relative_gap,
    int initialization_mode, int initial_certificate_check,
    int adaptive_line_search, double* output_shares, double* output_dual_weights,
    double* output_average_shares, double* output_average_dual_weights,
    int* output_status, std::uint64_t* output_iterations,
    std::uint64_t* output_accepted_trials, std::uint64_t* output_rejected_trials,
    double* output_upper_bound_ms, double* output_lower_bound_ms,
    double* output_gap_ms, double* output_step_size,
    double* output_operator_norm_bound, double* output_elapsed_ms,
    double* output_initial_gap_ms, double* output_total_weights,
    double* output_max_delta, double* output_max_accepted_delta,
    double* output_max_floating_slack, double* output_initial_step_size,
    double* output_final_step_size, double* output_max_step_size,
    int* output_initial_certificate_passed, char* error_buffer,
    std::size_t error_capacity) {
  try {
    if (row_count == 0 || variable_count == 0 || block_count == 0 ||
        row_offsets == nullptr || dual_offsets == nullptr || column_offsets == nullptr ||
        initial_shares == nullptr || output_shares == nullptr ||
        output_dual_weights == nullptr || output_average_shares == nullptr ||
        output_average_dual_weights == nullptr || output_status == nullptr ||
        output_iterations == nullptr || output_accepted_trials == nullptr ||
        output_rejected_trials == nullptr || output_upper_bound_ms == nullptr ||
        output_lower_bound_ms == nullptr || output_gap_ms == nullptr ||
        output_step_size == nullptr || output_operator_norm_bound == nullptr ||
        output_elapsed_ms == nullptr || output_initial_gap_ms == nullptr ||
        output_total_weights == nullptr || output_max_delta == nullptr ||
        output_max_accepted_delta == nullptr || output_max_floating_slack == nullptr ||
        output_initial_step_size == nullptr || output_final_step_size == nullptr ||
        output_max_step_size == nullptr || output_initial_certificate_passed == nullptr) {
      throw std::runtime_error("adaptive native solver received a null pointer or empty shape");
    }
    if (normalization_ms <= 0.0 || !std::isfinite(normalization_ms) ||
        !std::isfinite(fixed_startup_ms) || operator_norm_bound < 0.0 ||
        !std::isfinite(operator_norm_bound) || max_iterations == 0 ||
        absolute_gap_ms < 0.0 || !std::isfinite(absolute_gap_ms) ||
        relative_gap < 0.0 || !std::isfinite(relative_gap) ||
        (initialization_mode != 0 && initialization_mode != 1) ||
        (initial_certificate_check != 0 && initial_certificate_check != 1) ||
        (adaptive_line_search != 0 && adaptive_line_search != 1)) {
      throw std::runtime_error("adaptive native solver received invalid scalar parameters");
    }
    const auto total_resources = dual_offsets[block_count];
    if (total_resources == 0) {
      throw std::runtime_error("adaptive native solver requires a dual resource");
    }
    for (std::uint64_t row = 0; row < row_count; ++row) {
      if (row_offsets[row] >= row_offsets[row + 1] ||
          row_offsets[row + 1] > variable_count) {
        throw std::runtime_error("invalid row simplex offsets");
      }
    }
    for (std::uint64_t block = 0; block < block_count; ++block) {
      if (dual_offsets[block] >= dual_offsets[block + 1]) {
        throw std::runtime_error("invalid dual simplex offsets");
      }
    }

    ProblemView problem{row_count, variable_count, block_count, row_offsets,
                        dual_offsets, column_offsets, resource_indices, coefficients,
                        normalization_ms, fixed_startup_ms};
    const double initial_step = operator_norm_bound == 0.0
        ? 0.0 : 0.5 / operator_norm_bound;
    const double maximum_step = 1e4 * initial_step;
    double gamma_guess = initial_step;
    double last_step = initial_step;
    double maximum_step_used = 0.0;
    std::vector<double> current_shares(initial_shares,
                                       initial_shares + variable_count);
    std::vector<double> current_dual(total_resources, 0.0);
    std::vector<double> best_shares(variable_count, 0.0);
    std::vector<double> best_dual(total_resources, 0.0);
    Workspace workspace(problem, total_resources);
    initialize_dual_into(problem, current_shares, current_dual,
                         workspace.current_loads, initialization_mode);
    double best_upper = objective_value_into(problem, current_shares, workspace);
    double best_lower = dual_lower_bound_into(problem, current_dual, workspace);
    const double initial_gap = best_upper - best_lower;
    best_shares = current_shares;
    best_dual = current_dual;
    const double step_size = initial_step;
    *output_step_size = step_size;
    *output_operator_norm_bound = operator_norm_bound;
    *output_initial_step_size = initial_step;
    *output_final_step_size = initial_step;
    *output_max_step_size = 0.0;
    *output_initial_certificate_passed = 0;

    Status status = Status::iteration_cap;
    bool initial_passed = false;
    bool certificate_error = false;
    std::uint64_t accepted_trials = 0;
    std::uint64_t rejected_trials = 0;
    std::uint64_t iterations = 0;
    double total_weights = 0.0;
    double maximum_delta = -std::numeric_limits<double>::infinity();
    double maximum_accepted_delta = -std::numeric_limits<double>::infinity();
    double maximum_slack = 0.0;
    const auto checkpoint_interval = std::max<std::uint64_t>(
        1, std::min<std::uint64_t>(50, max_iterations / 100));

    const auto start_time = std::chrono::steady_clock::now();
    if (operator_norm_bound == 0.0) {
      if (initial_certificate_check != 0 &&
          initial_gap <= gap_threshold(absolute_gap_ms, relative_gap,
                                       best_upper, best_lower)) {
        initial_passed = true;
      }
      status = Status::converged;
    } else if (initial_certificate_check != 0 &&
               initial_gap <= gap_threshold(absolute_gap_ms, relative_gap,
                                            best_upper, best_lower)) {
      initial_passed = true;
      status = Status::converged;
    } else {
      bool exhausted = false;
      for (std::uint64_t iteration = 1; iteration <= max_iterations; ++iteration) {
        bool accepted = false;
        const std::uint64_t proposal_limit = adaptive_line_search != 0 ? 100 : 1;
        for (std::uint64_t proposal = 0; proposal < proposal_limit; ++proposal) {
          const double gamma = adaptive_line_search != 0
              ? std::min(gamma_guess, maximum_step) : initial_step;
          apply_adjoint_into(problem, current_dual, workspace.current_gradient);
          for (std::uint64_t variable = 0; variable < variable_count; ++variable) {
            workspace.trial_share_values[variable] = current_shares[variable] -
                gamma * workspace.current_gradient[variable] / normalization_ms;
          }
          project_product_simplexes_into(problem, workspace.trial_share_values,
                                          workspace.trial_shares,
                                          workspace.sorted_simplex);

          apply_operator_into(problem, current_shares, workspace.current_loads);
          for (std::uint64_t resource = 0; resource < total_resources; ++resource) {
            workspace.trial_dual_values[resource] = current_dual[resource] +
                gamma * workspace.current_loads[resource] / normalization_ms;
          }
          project_dual_into(problem, workspace.trial_dual_values,
                            workspace.trial_dual, workspace.sorted_simplex);

          apply_adjoint_into(problem, workspace.trial_dual, workspace.trial_gradient);
          for (std::uint64_t variable = 0; variable < variable_count; ++variable) {
            workspace.next_share_values[variable] = current_shares[variable] -
                gamma * workspace.trial_gradient[variable] / normalization_ms;
          }
          project_product_simplexes_into(problem, workspace.next_share_values,
                                          workspace.next_shares,
                                          workspace.sorted_simplex);

          apply_operator_into(problem, workspace.trial_shares, workspace.trial_loads);
          for (std::uint64_t resource = 0; resource < total_resources; ++resource) {
            workspace.next_dual_values[resource] = current_dual[resource] +
                gamma * workspace.trial_loads[resource] / normalization_ms;
          }
          project_dual_into(problem, workspace.next_dual_values,
                            workspace.next_dual, workspace.sorted_simplex);

          double delta = 0.0;
          double floating_slack = 0.0;
          if (adaptive_line_search != 0) {
            delta = local_condition_delta(
                problem, current_shares, current_dual, workspace.trial_shares,
                workspace.trial_dual, workspace.next_shares, workspace.next_dual,
                workspace.current_gradient, workspace.trial_gradient,
                workspace.current_loads, workspace.trial_loads,
                normalization_ms, gamma, &floating_slack);
            maximum_delta = std::max(maximum_delta, delta);
            maximum_slack = std::max(maximum_slack, floating_slack);
            if (delta > 0.0) {
              ++rejected_trials;
              gamma_guess = gamma * 0.8;
              if (!(gamma_guess > 0.0) || !std::isfinite(gamma_guess)) {
                exhausted = true;
                break;
              }
              continue;
            }
            maximum_accepted_delta = std::max(maximum_accepted_delta, delta);
          }

          accepted = true;
          ++accepted_trials;
          iterations = accepted_trials;
          last_step = gamma;
          maximum_step_used = std::max(maximum_step_used, gamma);
          total_weights += gamma;
          const double averaging_weight = adaptive_line_search != 0 ? gamma : 1.0;
          for (std::uint64_t index = 0; index < variable_count; ++index) {
            workspace.average_share_sums[index] +=
                averaging_weight * workspace.trial_shares[index];
          }
          for (std::uint64_t index = 0; index < total_resources; ++index) {
            workspace.average_dual_sums[index] +=
                averaging_weight * workspace.trial_dual[index];
          }
          current_shares.swap(workspace.next_shares);
          current_dual.swap(workspace.next_dual);
          gamma_guess = std::min(gamma * 1.2, maximum_step);
          break;
        }

        if (!accepted) {
          status = Status::line_search_exhausted;
          exhausted = true;
          break;
        }

        if (iterations % checkpoint_interval == 0 || iterations == max_iterations) {
          const double average_denominator = adaptive_line_search != 0
              ? total_weights : static_cast<double>(iterations);
          for (std::uint64_t index = 0; index < variable_count; ++index) {
            workspace.averaged_shares[index] =
                workspace.average_share_sums[index] / average_denominator;
          }
          for (std::uint64_t index = 0; index < total_resources; ++index) {
            workspace.averaged_dual[index] =
                workspace.average_dual_sums[index] / average_denominator;
          }

          double candidate_upper = objective_value_into(problem, current_shares, workspace);
          if (candidate_upper < best_upper) {
            best_upper = candidate_upper;
            best_shares = current_shares;
          }
          candidate_upper = objective_value_into(problem, workspace.trial_shares, workspace);
          if (candidate_upper < best_upper) {
            best_upper = candidate_upper;
            best_shares = workspace.trial_shares;
          }
          candidate_upper = objective_value_into(problem, current_shares, workspace);
          if (candidate_upper < best_upper) {
            best_upper = candidate_upper;
            best_shares = current_shares;
          }
          candidate_upper = objective_value_into(problem, workspace.averaged_shares, workspace);
          if (candidate_upper < best_upper) {
            best_upper = candidate_upper;
            best_shares = workspace.averaged_shares;
          }

          double candidate_lower = dual_lower_bound_into(problem, current_dual, workspace);
          if (candidate_lower > best_lower) {
            best_lower = candidate_lower;
            best_dual = current_dual;
          }
          candidate_lower = dual_lower_bound_into(problem, workspace.trial_dual, workspace);
          if (candidate_lower > best_lower) {
            best_lower = candidate_lower;
            best_dual = workspace.trial_dual;
          }
          candidate_lower = dual_lower_bound_into(problem, current_dual, workspace);
          if (candidate_lower > best_lower) {
            best_lower = candidate_lower;
            best_dual = current_dual;
          }
          candidate_lower = dual_lower_bound_into(problem, workspace.averaged_dual, workspace);
          if (candidate_lower > best_lower) {
            best_lower = candidate_lower;
            best_dual = workspace.averaged_dual;
          }

          const double gap = best_upper - best_lower;
          const double numeric_tolerance = 1e-10 *
              std::max({1.0, std::abs(best_upper), std::abs(best_lower)});
          if (gap < -numeric_tolerance) {
            certificate_error = true;
            status = Status::certificate_error;
            break;
          }
          if (gap <= gap_threshold(absolute_gap_ms, relative_gap,
                                  best_upper, best_lower)) {
            status = Status::converged;
            break;
          }
        }
      }
      if (certificate_error) {
        status = Status::certificate_error;
      } else if (!exhausted && status != Status::converged) {
        status = Status::iteration_cap;
      }
    }

    if (total_weights > 0.0) {
      const double denominator = adaptive_line_search != 0
          ? total_weights : static_cast<double>(iterations);
      for (std::uint64_t index = 0; index < variable_count; ++index) {
        workspace.averaged_shares[index] = workspace.average_share_sums[index] / denominator;
      }
      for (std::uint64_t index = 0; index < total_resources; ++index) {
        workspace.averaged_dual[index] = workspace.average_dual_sums[index] / denominator;
      }
    } else {
      workspace.averaged_shares = current_shares;
      workspace.averaged_dual = current_dual;
    }

    const auto end_time = std::chrono::steady_clock::now();
    const double elapsed_ms = std::chrono::duration<double, std::milli>(
        end_time - start_time).count();
    *output_status = static_cast<int>(status);
    *output_iterations = iterations;
    *output_accepted_trials = accepted_trials;
    *output_rejected_trials = rejected_trials;
    *output_upper_bound_ms = best_upper;
    *output_lower_bound_ms = best_lower;
    *output_gap_ms = best_upper - best_lower;
    *output_step_size = last_step;
    *output_operator_norm_bound = operator_norm_bound;
    *output_elapsed_ms = elapsed_ms;
    *output_initial_gap_ms = initial_gap;
    *output_total_weights = total_weights;
    *output_max_delta = std::isfinite(maximum_delta) ? maximum_delta : 0.0;
    *output_max_accepted_delta =
        std::isfinite(maximum_accepted_delta) ? maximum_accepted_delta : 0.0;
    *output_max_floating_slack = maximum_slack;
    *output_initial_step_size = initial_step;
    *output_final_step_size = iterations > 0 ? last_step : initial_step;
    *output_max_step_size = maximum_step_used;
    *output_initial_certificate_passed = initial_passed ? 1 : 0;
    std::copy(best_shares.begin(), best_shares.end(), output_shares);
    std::copy(best_dual.begin(), best_dual.end(), output_dual_weights);
    std::copy(workspace.averaged_shares.begin(), workspace.averaged_shares.end(),
              output_average_shares);
    std::copy(workspace.averaged_dual.begin(), workspace.averaged_dual.end(),
              output_average_dual_weights);
    return 0;
  } catch (const std::exception& error) {
    copy_error(error.what(), error_buffer, error_capacity);
    return 1;
  } catch (...) {
    copy_error("unknown adaptive native Mirror-Prox failure", error_buffer,
               error_capacity);
    return 2;
  }
}
