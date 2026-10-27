#pragma once

#include <cutil/mat.hpp>
#include <cutil/ref.hpp>
#include <memory>
#include <string>
#include <vector>

#include "fast_kinematics.hpp"
#include "types.hpp"
#include "urdf_loader.hpp"

namespace vkmotion {

class RobotConfig {
 public:
  static cutil::Ref<RobotConfig> from_file(const std::string& urdf_path, const std::string& collision_spheres_yaml_path = "");

  const std::vector<std::string>& joint_names() const { return joint_names_; }
  size_t dof() const { return joint_names_.size(); }
  const std::vector<float>& joint_position_limits_lower() const { return limits_lower_; }
  const std::vector<float>& joint_position_limits_upper() const { return limits_upper_; }
  const std::vector<float>& joint_velocity_limits() const { return limits_velocity_; }
  const std::vector<std::string>& link_names() const { return link_names_; }

  std::vector<cutil::Mat4f> forward_kinematics(const JointState& joint_state) const;
  std::vector<CollisionSphere> collision_spheres_world(const JointState& joint_state) const;
  const std::vector<CollisionSphere>& collision_spheres_local() const { return collision_spheres_; }

  const UrdfModel& urdf_model() const { return model_; }
  // ロード時にコンパイル済みのFK(hot path用)。forward_kinematics()もこれで計算する。
  const CompiledKinematics& compiled_kinematics() const { return *kin_; }

 private:
  UrdfModel model_;
  std::shared_ptr<CompiledKinematics> kin_;
  std::vector<std::string> joint_names_;
  std::vector<float> limits_lower_;
  std::vector<float> limits_upper_;
  std::vector<float> limits_velocity_;
  std::vector<std::string> link_names_;
  std::vector<CollisionSphere> collision_spheres_;
};

}  // namespace vkmotion
