#include <algorithm>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <string>
#include <vector>

#define main granite_scaling_frozen_cli_main
#include "granite_scaling_sinkhorn_relaxed_v2.cpp"
#undef main

#if defined(_WIN32)
#define MFG_EXPORT __declspec(dllexport)
#else
#define MFG_EXPORT __attribute__((visibility("default")))
#endif

namespace {
constexpr std::size_t kDiagnosticCount = 17;
constexpr std::size_t kMetadataCount = 4;
constexpr std::size_t kPolicyStatusCapacity = 64;

int copy_error(char* output, std::size_t capacity, int code, const char* message) noexcept {
  if (output && capacity) {
    const std::size_t n = std::min(capacity - 1, std::strlen(message));
    std::memcpy(output, message, n);
    output[n] = '\0';
  }
  return code;
}

const char* policy_name(int32_t policy) {
  switch (policy) {
    case 0: return "structured";
    case 1: return "lplb";
    case 2: return "sinkhorn_relaxed";
    default: return nullptr;
  }
}

void fill_diagnostics(const granite_scaling::PolicyResult& result,
                      double solve_us, double call_us,
                      double* d) {
  double continuous_peak = 0.0;
  double integer_peak = 0.0;
  for (double x : result.continuous_loads) continuous_peak = std::max(continuous_peak, x / 10.0);
  for (double x : result.integer_loads) integer_peak = std::max(integer_peak, x / 10.0);
  d[0] = solve_us;
  d[1] = call_us;
  d[2] = result.row_residual;
  d[3] = result.column_residual;
  d[4] = result.kkt_residual;
  d[5] = continuous_peak;
  d[6] = integer_peak;
  d[7] = static_cast<double>(result.f_tau);
  d[8] = static_cast<double>(result.fw_gap);
  d[9] = result.relaxed_epsilon;
  d[10] = result.relaxed_rho;
  d[11] = result.relaxed_theta;
  d[12] = result.relaxed_objective;
  d[13] = result.relaxed_update_residual;
  d[14] = result.relaxed_kkt_score_range;
  d[15] = result.relaxed_uniform_column_deviation_routes;
  d[16] = result.relaxed_probability_response_deviation;
}
}  // namespace

extern "C" MFG_EXPORT int granite_training_policy_plan_v1(
    const int64_t* counts,
    int32_t expert_count,
    int32_t gpus,
    int32_t policy,
    std::size_t output_length,
    double* shares_left,
    int64_t* quotas_left,
    int32_t* metadata,
    std::size_t metadata_length,
    double* diagnostics,
    std::size_t diagnostics_length,
    char* policy_status,
    std::size_t policy_status_capacity,
    char* error_message,
    std::size_t error_capacity) noexcept {
  using namespace granite_scaling;
  if (error_message && error_capacity) error_message[0] = '\0';
  if (!error_message || error_capacity == 0) return 1;
  if (!counts || !shares_left || !quotas_left || !metadata || !diagnostics || !policy_status) {
    return copy_error(error_message, error_capacity, 1, "all count and output pointers must be non-null");
  }
  if (expert_count <= 0 || expert_count > 4096) {
    return copy_error(error_message, error_capacity, 1, "expert_count must be in 1..4096");
  }
  if (output_length < static_cast<std::size_t>(expert_count)) {
    return copy_error(error_message, error_capacity, 1, "output_length is shorter than expert_count");
  }
  if (metadata_length < kMetadataCount) {
    return copy_error(error_message, error_capacity, 1, "metadata_length must be at least 4");
  }
  if (diagnostics_length < kDiagnosticCount) {
    return copy_error(error_message, error_capacity, 1, "diagnostics_length must be at least 17");
  }
  if (policy_status_capacity < kPolicyStatusCapacity) {
    return copy_error(error_message, error_capacity, 1, "policy_status_capacity must be at least 64");
  }
  if (!(gpus == 2 || (gpus >= 4 && gpus <= 512))) {
    return copy_error(error_message, error_capacity, 1, "gpus must be 2 or in 4..512");
  }
  const char* strategy = policy_name(policy);
  if (!strategy) {
    return copy_error(error_message, error_capacity, 1, "policy must be 0 (structured), 1 (lplb), or 2 (sinkhorn_relaxed)");
  }

  const auto call_begin = Clock::now();
  try {
    std::vector<int64_t> input(counts, counts + expert_count);
    int64_t total = 0;
    for (int64_t count : input) {
      if (count < 0) {
        return copy_error(error_message, error_capacity, 1, "counts must be nonnegative int64 values");
      }
      if (count > std::numeric_limits<int64_t>::max() - total) {
        return copy_error(error_message, error_capacity, 2, "route total overflows int64");
      }
      total += count;
      if (total > kMaxTotalRoutes) {
        return copy_error(error_message, error_capacity, 2, "route total exceeds exact binary64 integer range");
      }
    }

    Config cfg;
    cfg.g = gpus;
    cfg.e = expert_count;
    cfg.group_size = 8;
    cfg.topology = "ring";
    cfg.strategy = strategy;
    cfg.mode = "counts";
    cfg.repeats = 1;
    cfg.declared_counts = input;

    int32_t components = 1;
    int32_t ring_size = gpus;
    std::vector<Placement> placements;
    if (gpus == 2) {
      placements.resize(static_cast<std::size_t>(expert_count));
      for (int32_t e = 0; e < expert_count; ++e) {
        const int32_t left = e % 2;
        placements[static_cast<std::size_t>(e)] = {left, (left + 1) % 2, 0, left};
      }
    } else {
      placements = make_placements(cfg, components, ring_size);
    }

    const auto solve_begin = Clock::now();
    PolicyResult result = solve_policy(cfg, input, placements, components, ring_size, strategy);
    const auto solve_end = Clock::now();
    for (int32_t e = 0; e < expert_count; ++e) {
      const std::size_t index = static_cast<std::size_t>(e);
      if (!std::isfinite(result.shares[index][0]) ||
          result.shares[index][0] < 0.0 || result.shares[index][0] > 1.0 ||
          result.quotas[index][0] < 0 || result.quotas[index][0] > input[index] ||
          result.quotas[index][0] + result.quotas[index][1] != input[index]) {
        return copy_error(error_message, error_capacity, 3, "native planner returned an invalid share or quota");
      }
    }

    const auto call_end = Clock::now();
    const double solve_us = micros(solve_begin, solve_end);
    const double call_us = micros(call_begin, call_end);
    for (int32_t e = 0; e < expert_count; ++e) {
      const std::size_t index = static_cast<std::size_t>(e);
      shares_left[index] = result.shares[index][0];
      quotas_left[index] = result.quotas[index][0];
    }
    metadata[0] = result.iterations;
    metadata[1] = result.boundary_flags;
    metadata[2] = result.active_tie_guard_count;
    metadata[3] = result.relaxed_active_component_count;
    fill_diagnostics(result, solve_us, call_us, diagnostics);
    const std::size_t status_n = std::min(policy_status_capacity - 1, result.status.size());
    std::memcpy(policy_status, result.status.data(), status_n);
    policy_status[status_n] = '\0';
    return 0;
  } catch (const std::exception& ex) {
    return copy_error(error_message, error_capacity, 3, ex.what());
  } catch (...) {
    return copy_error(error_message, error_capacity, 4, "unknown native planner failure");
  }
}
