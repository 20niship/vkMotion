#include "cartesian_path.hpp"

#include <cmath>

#include "ik_solver.hpp"
#include "robot_world.hpp"

namespace vkmotion {

namespace {

Pose ee_pose(const RobotConfig& robot, const JointState& js) {
  std::vector<cutil::Mat4f> link_world = robot.forward_kinematics(js);
  const cutil::Mat4f& t                = link_world.back();
  Pose p;
  p.position = cutil::Vec3f(t(0, 3), t(1, 3), t(2, 3));
  cutil::Mat<3, 3, float> r;
  for(unsigned i = 0; i < 3; i++)
    for(unsigned j = 0; j < 3; j++) r(i, j) = t(i, j);
  p.quaternion = cutil::Quatf::from_mat3(r);
  return p;
}

float quat_angle_between(const cutil::Quatf& a, const cutil::Quatf& b) {
  float d = std::abs(static_cast<float>(a.dot(b)));
  d       = std::max(-1.0f, std::min(1.0f, d));
  return 2.0f * std::acos(d);
}

Pose lerp_pose(const Pose& a, const Pose& b, float t) {
  Pose p;
  p.position   = a.position + (b.position - a.position) * t;
  p.quaternion = a.quaternion.slerp(b.quaternion, t);
  return p;
}

}  // namespace

CartesianPathPlanner::CartesianPathPlanner(cutil::Ref<RobotConfig> robot_config, cutil::Ref<WorldConfig> world_config, bool use_vulkan)
    : robot_config_(robot_config), world_config_(world_config), use_vulkan_(use_vulkan) {}

CartesianPathResult CartesianPathPlanner::compute_cartesian_path(const JointState& start_state, const std::vector<CartesianWaypoint>& waypoints,
                                                                   const CartesianPathConfig& config) {
  CartesianPathResult result;
  if(waypoints.empty()) {
    result.status = "no waypoints given";
    return result;
  }

  RobotWorld rw(RobotWorldConfig{robot_config_, world_config_, use_vulkan_});
  IKSolverConfig ik_cfg{robot_config_, world_config_, 32};
  IKSolver ik(ik_cfg);

  if(!rw.is_collision_free(start_state)) {
    result.status = "start state is in collision";
    return result;
  }

  std::vector<Pose> cartesian_pts;
  cartesian_pts.push_back(ee_pose(*robot_config_, start_state));
  for(const auto& wp : waypoints) cartesian_pts.push_back(wp.pose);

  float total_len = 0.0f;
  for(size_t i = 1; i < cartesian_pts.size(); i++) total_len += (cartesian_pts[i].position - cartesian_pts[i - 1].position).length();
  if(total_len < 1e-9f) total_len = 1e-9f;

  result.trajectory.push_back(start_state);
  JointState seed      = start_state;
  float achieved_len   = 0.0f;

  for(size_t seg = 1; seg < cartesian_pts.size(); seg++) {
    const Pose& a = cartesian_pts[seg - 1];
    const Pose& b = cartesian_pts[seg];
    float seg_len = (b.position - a.position).length();
    float seg_ang = quat_angle_between(a.quaternion, b.quaternion);
    int steps      = std::max(1, static_cast<int>(std::ceil(std::max(seg_len / config.max_step, seg_ang / config.max_angular_step))));

    bool segment_ok = true;
    for(int s = 1; s <= steps; s++) {
      float t          = static_cast<float>(s) / static_cast<float>(steps);
      Pose target       = lerp_pose(a, b, t);
      auto ik_result    = ik.solve_single(target, seed);
      if(!ik_result || !rw.is_collision_free(*ik_result)) {
        segment_ok = false;
        break;
      }
      seed = *ik_result;
      result.trajectory.push_back(seed);
      achieved_len += seg_len / static_cast<float>(steps);
    }
    if(!segment_ok) break;
  }

  result.fraction_achieved = std::min(1.0f, achieved_len / total_len);
  result.success           = result.fraction_achieved >= config.min_fraction;
  result.status            = result.success ? "success" : "incomplete: IK or collision failure along the path";
  return result;
}

}  // namespace vkmotion
