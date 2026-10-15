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

enum class Status : int { converged = 0, iteration_cap = 1, certificate_error = 2 };

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
