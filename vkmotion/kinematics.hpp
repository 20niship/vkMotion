#pragma once

#include <cutil/mat.hpp>
#include <vector>

#include "urdf_loader.hpp"

namespace vkmotion {

// Per-link world transforms (indexed like model.links), joint_positions ordered like active_joint_names(model).
std::vector<cutil::Mat4f> forward_kinematics(const UrdfModel& model, const std::vector<float>& joint_positions);

// Root-to-tip joint ordering for a non-branching serial-chain robot (GPU FK needs a flat chain, not a tree).
struct SerialChain {
  std::vector<int> joint_order;  // indices into model.joints, kinematic (root->tip) order
  std::vector<int> link_depth;   // per model.links index: how many joint_order entries precede that link (root link = 0)
};

// Throws on branching (>1 child joint per link); IKSolver already assumes a single serial chain too.
SerialChain build_serial_chain(const UrdfModel& model);

}  // namespace vkmotion
