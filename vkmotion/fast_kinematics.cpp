#include "fast_kinematics.hpp"

#include <algorithm>
#include <cmath>
#include <queue>
#include <stdexcept>
#include <unordered_map>

namespace vkmotion {

namespace {

// URDFのrpy: R = Rz(yaw) * Ry(pitch) * Rx(roll)
void rpy_to_matrix(const cutil::Vec3f& rpy, float* r) {
  const float cr = std::cos(rpy[0]), sr = std::sin(rpy[0]);
  const float cp = std::cos(rpy[1]), sp = std::sin(rpy[1]);
  const float cy = std::cos(rpy[2]), sy = std::sin(rpy[2]);
  r[0] = cy * cp;
  r[1] = cy * sp * sr - sy * cr;
  r[2] = cy * sp * cr + sy * sr;
  r[3] = sy * cp;
  r[4] = sy * sp * sr + cy * cr;
  r[5] = sy * sp * cr - cy * sr;
  r[6] = -sp;
  r[7] = cp * sr;
  r[8] = cp * cr;
}

inline void mul33(const float* a, const float* b, float* c) {
  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 3; j++) c[i * 3 + j] = a[i * 3 + 0] * b[0 + j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j];
}

inline void rot_axis_angle(const float* a, float s, float c, float* r) {
  const float t = 1.0f - c;
  r[0] = c + a[0] * a[0] * t;
  r[1] = a[0] * a[1] * t - a[2] * s;
  r[2] = a[0] * a[2] * t + a[1] * s;
  r[3] = a[1] * a[0] * t + a[2] * s;
  r[4] = c + a[1] * a[1] * t;
  r[5] = a[1] * a[2] * t - a[0] * s;
  r[6] = a[2] * a[0] * t - a[1] * s;
  r[7] = a[2] * a[1] * t + a[0] * s;
  r[8] = c + a[2] * a[2] * t;
}

}  // namespace

CompiledKinematics::CompiledKinematics(const UrdfModel& model) {
  num_links_ = model.links.size();
  link_names_.reserve(num_links_);
  std::unordered_map<std::string, int> link_idx;
  for(size_t i = 0; i < model.links.size(); i++) {
    link_names_.push_back(model.links[i].name);
    link_idx[model.links[i].name] = static_cast<int>(i);
  }

  std::unordered_map<std::string, int> active_idx;
  for(const auto& j : model.joints)
    if(j.type != UrdfJointType::kFixed) active_idx[j.name] = static_cast<int>(dof_++);

  std::vector<std::vector<int>> children(num_links_);
  std::vector<bool> is_child(num_links_, false);
  for(size_t i = 0; i < model.joints.size(); i++) {
    const auto& j = model.joints[i];
    auto p = link_idx.find(j.parent_link);
    auto c = link_idx.find(j.child_link);
    if(p == link_idx.end() || c == link_idx.end()) throw std::runtime_error("vkmotion: joint '" + j.name + "' refers to an unknown link");
    children[static_cast<size_t>(p->second)].push_back(static_cast<int>(i));
    is_child[static_cast<size_t>(c->second)] = true;
  }

  step_of_link_.assign(num_links_, -1);
  chain_joints_.assign(num_links_, {});
  std::queue<int> bfs;
  for(size_t l = 0; l < num_links_; l++) {
    if(!is_child[l]) {
      root_links_.push_back(static_cast<int>(l));
      bfs.push(static_cast<int>(l));
    }
  }
  while(!bfs.empty()) {
    int l = bfs.front();
    bfs.pop();
    for(int ji : children[static_cast<size_t>(l)]) {
      const UrdfJoint& j = model.joints[static_cast<size_t>(ji)];
      Step s{};
      s.parent = l;
      s.link   = link_idx[j.child_link];
      rpy_to_matrix(j.origin_rpy, s.ro);
      for(int k = 0; k < 3; k++) s.to[k] = j.origin_xyz[k];
      float n = std::sqrt(j.axis[0] * j.axis[0] + j.axis[1] * j.axis[1] + j.axis[2] * j.axis[2]);
      if(n < 1e-9f) n = 1.0f;
      for(int k = 0; k < 3; k++) s.axis[k] = j.axis[k] / n;
      switch(j.type) {
        case UrdfJointType::kRevolute:
        case UrdfJointType::kContinuous: s.type = 1; break;
        case UrdfJointType::kPrismatic: s.type = 2; break;
        default: s.type = 0; break;
      }
      s.active = s.type == 0 ? -1 : active_idx[j.name];
      if(s.type == 1) {
        // Rot(a,th) = c*I + s*[a]x + (1-c)*a a^T  =>  ro*Rot = ro*(a a^T) + c*ro*(I - a a^T) + s*ro*[a]x
        const float* a = s.axis;
        float aat[9]   = {a[0] * a[0], a[0] * a[1], a[0] * a[2], a[1] * a[0], a[1] * a[1], a[1] * a[2], a[2] * a[0], a[2] * a[1], a[2] * a[2]};
        float k[9]     = {0, -a[2], a[1], a[2], 0, -a[0], -a[1], a[0], 0};
        float ima[9];
        for(int i = 0; i < 9; i++) ima[i] = (i % 4 == 0 ? 1.0f : 0.0f) - aat[i];
        mul33(s.ro, aat, s.a0);
        mul33(s.ro, ima, s.ac);
        mul33(s.ro, k, s.as);
      }

      step_of_link_[static_cast<size_t>(s.link)] = static_cast<int>(steps_.size());
      chain_joints_[static_cast<size_t>(s.link)] = chain_joints_[static_cast<size_t>(l)];
      if(s.active >= 0) chain_joints_[static_cast<size_t>(s.link)].push_back(s.active);
      steps_.push_back(s);
      bfs.push(s.link);
    }
  }
}

int CompiledKinematics::link_index(const std::string& name) const {
  for(size_t i = 0; i < link_names_.size(); i++)
    if(link_names_[i] == name) return static_cast<int>(i);
  return -1;
}

void CompiledKinematics::forward(const float* q, Frame* out) const {
  for(int root : root_links_) {
    Frame& f = out[root];
    f.r[0] = f.r[4] = f.r[8] = 1.0f;
    f.r[1] = f.r[2] = f.r[3] = f.r[5] = f.r[6] = f.r[7] = 0.0f;
    f.p[0] = f.p[1] = f.p[2] = 0.0f;
  }
  for(const Step& s : steps_) {
    const Frame& par = out[s.parent];
    Frame& ch        = out[s.link];
    float m[9];
    float t[3] = {s.to[0], s.to[1], s.to[2]};
    if(s.type == 1) {
      float sn, cs;
      sn = std::sin(q[s.active]);
      cs = std::cos(q[s.active]);
      float rj[9];
      rot_axis_angle(s.axis, sn, cs, rj);
      mul33(s.ro, rj, m);
    } else {
      for(int i = 0; i < 9; i++) m[i] = s.ro[i];
      if(s.type == 2) {
        for(int k = 0; k < 3; k++) t[k] += s.axis[k] * q[s.active];
      }
    }
    mul33(par.r, m, ch.r);
    for(int i = 0; i < 3; i++) ch.p[i] = par.p[i] + par.r[i * 3] * t[0] + par.r[i * 3 + 1] * t[1] + par.r[i * 3 + 2] * t[2];
  }
}

void CompiledKinematics::forward8(const v8f* sin_q, const v8f* cos_q, const v8f* q, Frame8* out) const {
  for(int root : root_links_) {
    Frame8& f = out[root];
    for(int i = 0; i < 9; i++) f.r[i] = v8f{} + ((i % 4 == 0) ? 1.0f : 0.0f);
    for(int i = 0; i < 3; i++) f.p[i] = v8f{};
  }
  for(const Step& s : steps_) {
    const Frame8& par = out[s.parent];
    Frame8& ch        = out[s.link];
    v8f m[9];
    v8f t[3] = {v8f{} + s.to[0], v8f{} + s.to[1], v8f{} + s.to[2]};
    if(s.type == 1) {
      const v8f& sn = sin_q[s.active];
      const v8f& cs = cos_q[s.active];
      for(int i = 0; i < 9; i++) m[i] = s.a0[i] + cs * s.ac[i] + sn * s.as[i];
    } else {
      for(int i = 0; i < 9; i++) m[i] = v8f{} + s.ro[i];
      if(s.type == 2) {
        for(int k = 0; k < 3; k++) t[k] += q[s.active] * s.axis[k];
      }
    }
    for(int i = 0; i < 3; i++) {
      for(int j = 0; j < 3; j++) ch.r[i * 3 + j] = par.r[i * 3] * m[j] + par.r[i * 3 + 1] * m[3 + j] + par.r[i * 3 + 2] * m[6 + j];
      ch.p[i] = par.p[i] + par.r[i * 3] * t[0] + par.r[i * 3 + 1] * t[1] + par.r[i * 3 + 2] * t[2];
    }
  }
}

void CompiledKinematics::export_steps(std::vector<int32_t>& step_meta, std::vector<float>& step_data, std::vector<int32_t>& roots) const {
  step_meta.clear();
  step_data.clear();
  roots.assign(root_links_.begin(), root_links_.end());
  for(const Step& s : steps_) {
    step_meta.insert(step_meta.end(), {s.parent, s.link, static_cast<int32_t>(s.type), s.active});
    step_data.insert(step_data.end(), {s.axis[0], s.axis[1], s.axis[2], 0.0f});
    for(int r = 0; r < 3; r++) step_data.insert(step_data.end(), {s.ro[r * 3 + 0], s.ro[r * 3 + 1], s.ro[r * 3 + 2], s.to[r]});
  }
}

void CompiledKinematics::reach_bound(int link, float local_norm, float* per_joint) const {
  float acc = local_norm;
  int cur   = link;
  while(step_of_link_[static_cast<size_t>(cur)] >= 0) {
    const Step& s = steps_[static_cast<size_t>(step_of_link_[static_cast<size_t>(cur)])];
    if(s.active >= 0) per_joint[s.active] = std::max(per_joint[s.active], s.type == 2 ? 1.0f : acc);
    acc += std::sqrt(s.to[0] * s.to[0] + s.to[1] * s.to[1] + s.to[2] * s.to[2]);
    cur = s.parent;
  }
}

void CompiledKinematics::jacobian(const Frame* frames, int tip_link, float* J) const {
  std::fill(J, J + 6 * dof_, 0.0f);
  const Frame& tip = frames[tip_link];
  for(int ji : chain_joints_[static_cast<size_t>(tip_link)]) {
    // ji番目のactive jointを持つstepを探す(dofは小さいので線形で十分。ホットパス用途ではIKが1反復あたり1回呼ぶだけ)
    for(const Step& s : steps_) {
      if(s.active != ji) continue;
      const Frame& jf = frames[s.link];
      float z[3];
      for(int i = 0; i < 3; i++) z[i] = jf.r[i * 3] * s.axis[0] + jf.r[i * 3 + 1] * s.axis[1] + jf.r[i * 3 + 2] * s.axis[2];
      if(s.type == 1) {
        float d[3] = {tip.p[0] - jf.p[0], tip.p[1] - jf.p[1], tip.p[2] - jf.p[2]};
        J[0 * dof_ + ji] = z[1] * d[2] - z[2] * d[1];
        J[1 * dof_ + ji] = z[2] * d[0] - z[0] * d[2];
        J[2 * dof_ + ji] = z[0] * d[1] - z[1] * d[0];
        J[3 * dof_ + ji] = z[0];
        J[4 * dof_ + ji] = z[1];
        J[5 * dof_ + ji] = z[2];
      } else {
        J[0 * dof_ + ji] = z[0];
        J[1 * dof_ + ji] = z[1];
        J[2 * dof_ + ji] = z[2];
      }
      break;
    }
  }
}

}  // namespace vkmotion
