#pragma once

#include <set>
#include <string>
#include <utility>

namespace vkmotion {

// MoveIt2's Allowed Collision Matrix: explicit disable-pairs beyond the default same/adjacent-link exclusion.
class AllowedCollisionMatrix {
 public:
  void disable_collision(const std::string& a, const std::string& b) { pairs_.insert(normalize(a, b)); }
  bool is_allowed(const std::string& a, const std::string& b) const { return pairs_.count(normalize(a, b)) > 0; }

 private:
  static std::pair<std::string, std::string> normalize(const std::string& a, const std::string& b) {
    return a < b ? std::make_pair(a, b) : std::make_pair(b, a);
  }
  std::set<std::pair<std::string, std::string>> pairs_;
};

}  // namespace vkmotion
