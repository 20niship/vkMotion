#pragma once

#include <cutil/mat.hpp>
#include <vector>

#include "urdf_loader.hpp"

namespace vkmotion {

// Per-link world transforms (indexed like model.links), joint_positions ordered like active_joint_names(model).
std::vector<cutil::Mat4f> forward_kinematics(const UrdfModel& model, const std::vector<float>& joint_positions);

}  // namespace vkmotion
