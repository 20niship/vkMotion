#pragma once

#include <concepts>
#include <cutil/quaternion.hpp>
#include <cutil/vec.hpp>
#include <variant>
#include <vector>

#include "robot_config.hpp"
#include "types.hpp"

namespace vkmotion {

// MoveIt2のpath constraints(position/orientation/joint)相当。
template <typename T>
concept Constraint = requires(const T& c, const JointState& js, const RobotConfig& robot) {
  { c.is_satisfied(js, robot) } -> std::convertible_to<bool>;
  { c.violation(js, robot) } -> std::convertible_to<float>;
};

// URDFのjoint limitより厳しい一時的な範囲を課す。
struct JointConstraint {
  int joint_index = 0;
  float lower     = 0.0f;
  float upper     = 0.0f;

  bool is_satisfied(const JointState& js, const RobotConfig& robot) const { return violation(js, robot) <= 0.0f; }
  float violation(const JointState& js, const RobotConfig&) const {
    float v = js.position[static_cast<size_t>(joint_index)];
    return std::max(lower - v, v - upper);
  }
};

// エンドエフェクタ(RobotConfig::link_names()の末端)の位置をtargetの周り±toleranceのボックス内に保つ。
struct PositionConstraint {
  cutil::Vec3f target{};
  cutil::Vec3f tolerance{};

  bool is_satisfied(const JointState& js, const RobotConfig& robot) const { return violation(js, robot) <= 0.0f; }
  float violation(const JointState& js, const RobotConfig& robot) const {
    const cutil::Mat4f& t = robot.forward_kinematics(js).back();
    cutil::Vec3f pos(t(0, 3), t(1, 3), t(2, 3));
    float v = -1e30f;
    for(int i = 0; i < 3; i++) v = std::max(v, std::abs(pos[i] - target[i]) - tolerance[i]);
    return v;
  }
};

// エンドエフェクタの姿勢をtargetからtolerance_rad以内に保つ。
struct OrientationConstraint {
  cutil::Quatf target = cutil::Quatf::identity();
  float tolerance_rad  = 0.0f;

  bool is_satisfied(const JointState& js, const RobotConfig& robot) const { return violation(js, robot) <= 0.0f; }
  float violation(const JointState& js, const RobotConfig& robot) const;
};

// ヘテロな制約集合。実行時に組み立てる(型消去はstd::variant+std::visitで素朴に行う)。
class ConstraintSet {
 public:
  template <Constraint T>
  void add(T c) {
    constraints_.emplace_back(std::move(c));
  }

  bool is_satisfied(const JointState& js, const RobotConfig& robot) const {
    for(const auto& c : constraints_) {
      bool ok = std::visit([&](const auto& constraint) { return constraint.is_satisfied(js, robot); }, c);
      if(!ok) return false;
    }
    return true;
  }

  // 違反量の合計。満たしていれば0。repair時の勾配降下の目的関数として使う。
  float total_violation(const JointState& js, const RobotConfig& robot) const {
    float total = 0.0f;
    for(const auto& c : constraints_) {
      float v = std::visit([&](const auto& constraint) { return constraint.violation(js, robot); }, c);
      total += std::max(0.0f, v);
    }
    return total;
  }

  bool empty() const { return constraints_.empty(); }

 private:
  std::vector<std::variant<JointConstraint, PositionConstraint, OrientationConstraint>> constraints_;
};

}  // namespace vkmotion
