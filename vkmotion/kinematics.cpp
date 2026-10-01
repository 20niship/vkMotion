#include "kinematics.hpp"

#include <cutil/quaternion.hpp>
#include <functional>
#include <unordered_map>
#include <unordered_set>

namespace vkmotion {

namespace {

cutil::Mat4f joint_origin_transform(const UrdfJoint& j) {
  return cutil::mat4_translation(j.origin_xyz) * cutil::mat4_rotation(cutil::Quatf::from_euler(j.origin_rpy));
}

cutil::Mat4f joint_motion_transform(const UrdfJoint& j, float value) {
  switch(j.type) {
    case UrdfJointType::kRevolute:
    case UrdfJointType::kContinuous:
      return cutil::mat4_rotation(j.axis, static_cast<double>(value));
    case UrdfJointType::kPrismatic:
      return cutil::mat4_translation(j.axis * value);
    case UrdfJointType::kFixed:
    default:
      return cutil::Mat4f::identity();
  }
}

}  // namespace

std::vector<cutil::Mat4f> forward_kinematics(const UrdfModel& model, const std::vector<float>& joint_positions) {
  std::vector<cutil::Mat4f> link_world(model.links.size(), cutil::Mat4f::identity());

  std::unordered_map<std::string, std::vector<int>> children_of;
  std::unordered_set<std::string> child_links;
  for(size_t i = 0; i < model.joints.size(); i++) {
    children_of[model.joints[i].parent_link].push_back(static_cast<int>(i));
    child_links.insert(model.joints[i].child_link);
  }

  std::unordered_map<std::string, int> active_index;
  int next_index = 0;
  for(const auto& j : model.joints) {
    if(j.type != UrdfJointType::kFixed) active_index[j.name] = next_index++;
  }

  std::function<void(const std::string&, const cutil::Mat4f&)> visit = [&](const std::string& link_name, const cutil::Mat4f& world) {
    int li = model.find_link(link_name);
    if(li >= 0) link_world[static_cast<size_t>(li)] = world;

    auto it = children_of.find(link_name);
    if(it == children_of.end()) return;
    for(int ji : it->second) {
      const UrdfJoint& j = model.joints[static_cast<size_t>(ji)];
      float value         = 0.0f;
      auto ait             = active_index.find(j.name);
      if(ait != active_index.end() && ait->second < static_cast<int>(joint_positions.size())) {
        value = joint_positions[static_cast<size_t>(ait->second)];
      }
      cutil::Mat4f child_world = world * joint_origin_transform(j) * joint_motion_transform(j, value);
      visit(j.child_link, child_world);
    }
  };

  for(const auto& link : model.links) {
    if(child_links.find(link.name) == child_links.end()) {
      visit(link.name, cutil::Mat4f::identity());
    }
  }

  return link_world;
}

}  // namespace vkmotion
