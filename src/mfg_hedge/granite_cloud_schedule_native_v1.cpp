#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace {
constexpr int kExperts = 32;
constexpr int kReplicas = 2;

struct TokenSlot {
  int64_t token;
  int64_t slot;
  bool operator==(const TokenSlot& other) const noexcept {
    return token == other.token && slot == other.slot;
  }
};
struct TokenSlotHash {
  size_t operator()(const TokenSlot& value) const noexcept {
    const size_t a = std::hash<int64_t>{}(value.token);
    const size_t b = std::hash<int64_t>{}(value.slot);
    return a ^ (b + static_cast<size_t>(0x9e3779b97f4a7c15ULL) + (a << 6) + (a >> 2));
  }
};

void set_error(char* out, int64_t capacity, const std::string& message) noexcept {
  if (out == nullptr || capacity <= 0) return;
  const size_t n = std::min(static_cast<size_t>(capacity - 1), message.size());
  std::memcpy(out, message.data(), n);
  out[n] = '\0';
}

int fail(char* error, int64_t capacity, const std::string& message) noexcept {
  set_error(error, capacity, message);
  return 1;
}
}  // namespace

extern "C" int granite_cloud_build_schedule_v1(
    int64_t route_count,
    const int64_t* expert_ids,
    const int64_t* token_ids,
    const int64_t* route_slots,
    int32_t device_count,
    int32_t replica_count,
    const int32_t* placement_gpu_by_expert_replica,
    const int64_t* quotas_by_expert_replica,
    int64_t* route_order_out,
    int64_t* counts_by_gpu_expert_out,
    int32_t* replica_by_route_out,
    char* error_out,
    int64_t error_capacity) noexcept {
  try {
    if (route_count < 0) return fail(error_out, error_capacity, "route_count must be nonnegative");
    if (device_count != 4 && device_count != 6 && device_count != 8)
      return fail(error_out, error_capacity, "device_count must be 4, 6, or 8");
    if (replica_count != kReplicas)
      return fail(error_out, error_capacity, "the frozen ring requires exactly two replicas per Expert");
    if (placement_gpu_by_expert_replica == nullptr || quotas_by_expert_replica == nullptr ||
        counts_by_gpu_expert_out == nullptr)
      return fail(error_out, error_capacity, "placement, quota, and count buffers are required");
    if (route_count > 0 && (expert_ids == nullptr || token_ids == nullptr || route_slots == nullptr ||
                            route_order_out == nullptr || replica_by_route_out == nullptr))
      return fail(error_out, error_capacity, "route and output buffers are required for nonempty input");
    if (static_cast<uint64_t>(route_count) > static_cast<uint64_t>(std::numeric_limits<size_t>::max()))
      return fail(error_out, error_capacity, "route_count exceeds host address space");

    std::array<int64_t, kExperts> expert_route_counts{};
    for (int expert = 0; expert < kExperts; ++expert) {
      const int32_t gpu0 = placement_gpu_by_expert_replica[expert * kReplicas];
      const int32_t gpu1 = placement_gpu_by_expert_replica[expert * kReplicas + 1];
      if (gpu0 < 0 || gpu0 >= device_count || gpu1 < 0 || gpu1 >= device_count)
        return fail(error_out, error_capacity, "placement GPU is outside the logical ring");
      if (gpu0 == gpu1)
        return fail(error_out, error_capacity, "two replicas of one Expert must use distinct GPUs");
    }

    std::unordered_set<TokenSlot, TokenSlotHash> unique_routes;
    unique_routes.reserve(static_cast<size_t>(route_count));
    for (int64_t route = 0; route < route_count; ++route) {
      const int64_t expert = expert_ids[route];
      if (expert < 0 || expert >= kExperts) return fail(error_out, error_capacity, "Expert id outside 0..31");
      if (token_ids[route] < 0 || route_slots[route] < 0)
        return fail(error_out, error_capacity, "token index and route slot must be nonnegative");
      if (!unique_routes.insert(TokenSlot{token_ids[route], route_slots[route]}).second)
        return fail(error_out, error_capacity, "duplicate (token, route slot)");
      ++expert_route_counts[static_cast<size_t>(expert)];
    }

    for (int expert = 0; expert < kExperts; ++expert) {
      const int64_t q0 = quotas_by_expert_replica[expert * kReplicas];
      const int64_t q1 = quotas_by_expert_replica[expert * kReplicas + 1];
      const int64_t demand = expert_route_counts[static_cast<size_t>(expert)];
      if (q0 < 0 || q1 < 0 || q0 > demand || q1 > demand || q0 + q1 != demand)
        return fail(error_out, error_capacity, "Expert replica quotas must be nonnegative and sum to its route count");
    }

    std::array<std::vector<int64_t>, kExperts> routes_by_expert;
    for (int64_t route = 0; route < route_count; ++route)
      routes_by_expert[static_cast<size_t>(expert_ids[route])].push_back(route);
    std::vector<int32_t> owners(static_cast<size_t>(route_count), -1);
    for (int expert = 0; expert < kExperts; ++expert) {
      auto& routes = routes_by_expert[static_cast<size_t>(expert)];
      std::sort(routes.begin(), routes.end(), [&](int64_t lhs, int64_t rhs) {
        if (token_ids[lhs] != token_ids[rhs]) return token_ids[lhs] < token_ids[rhs];
        if (route_slots[lhs] != route_slots[rhs]) return route_slots[lhs] < route_slots[rhs];
        return lhs < rhs;
      });
      const int64_t first_quota = quotas_by_expert_replica[expert * kReplicas];
      for (int64_t i = 0; i < static_cast<int64_t>(routes.size()); ++i)
        owners[static_cast<size_t>(routes[static_cast<size_t>(i)])] = (i < first_quota) ? 0 : 1;
    }

    std::vector<std::vector<int64_t>> groups(static_cast<size_t>(device_count * kExperts));
    for (int64_t route = 0; route < route_count; ++route) {
      const int32_t expert = static_cast<int32_t>(expert_ids[route]);
      const int32_t replica = owners[static_cast<size_t>(route)];
      if (replica < 0 || replica >= kReplicas)
        return fail(error_out, error_capacity, "a route did not receive exactly one replica assignment");
      if (replica_by_route_out != nullptr) replica_by_route_out[route] = replica;
      const int32_t gpu = placement_gpu_by_expert_replica[expert * kReplicas + replica];
      groups[static_cast<size_t>(gpu * kExperts + expert)].push_back(route);
    }

    int64_t cursor = 0;
    int64_t conserved = 0;
    for (int gpu = 0; gpu < device_count; ++gpu) {
      for (int expert = 0; expert < kExperts; ++expert) {
        const auto& routes = groups[static_cast<size_t>(gpu * kExperts + expert)];
        counts_by_gpu_expert_out[gpu * kExperts + expert] = static_cast<int64_t>(routes.size());
        for (const int64_t route : routes) {
          route_order_out[cursor++] = route;
          ++conserved;
        }
      }
    }
    if (cursor != route_count || conserved != route_count)
      return fail(error_out, error_capacity, "GPU/Expert schedule did not conserve every route");
    set_error(error_out, error_capacity, "");
    return 0;
  } catch (const std::exception& exc) {
    return fail(error_out, error_capacity, std::string("native schedule exception: ") + exc.what());
  } catch (...) {
    return fail(error_out, error_capacity, "unknown native schedule exception");
  }
}
