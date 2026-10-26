#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "urdf_loader.hpp"

namespace vkmotion {

// 8コンフィグを同時に扱うSIMDベクトル(GCC/Clang vector extension。AVX2が無ければコンパイラが分割して実行)。
typedef float v8f __attribute__((vector_size(32), aligned(32)));
typedef int v8i __attribute__((vector_size(32), aligned(32)));
constexpr int kSimdWidth = 8;

// 行優先3x3回転 + 並進。FKの内側ループでヒープ確保や文字列比較をしないための素のPOD。
struct Frame {
  float r[9];
  float p[3];
};

// UrdfModelをロード時に1回だけ「トポロジカル順のステップ列」へコンパイルしたFK/ヤコビアン。
// 既存のvkmotion::forward_kinematicsは呼び出しごとにhash mapを作る参照実装なので、ホットパスはこちらを使う。
struct Frame8 {
  v8f r[9];
  v8f p[3];
};

class CompiledKinematics {
 public:
  explicit CompiledKinematics(const UrdfModel& model);

  size_t dof() const { return dof_; }
  size_t num_links() const { return num_links_; }
  int link_index(const std::string& name) const;  // -1: 無し

  // out[num_links]。q[dof]はactive_joint_names(model)の順。
  void forward(const float* q, Frame* out) const;

  // 8コンフィグ分を一度に計算する。sin_q/cos_q/q は active joint index ごとに8レーン分(prismaticはqを使う)。out[num_links]
  void forward8(const v8f* sin_q, const v8f* cos_q, const v8f* q, Frame8* out) const;

  // tip_linkのフレーム原点に対する幾何ヤコビアン(6 x dof, 行優先; 上3行=並進, 下3行=回転)。framesはforward()の出力。
  void jacobian(const Frame* frames, int tip_link, float* J) const;

  // link上でoriginからlocal_norm離れた点が、各active jointの回転(または並進)1単位あたりに動く距離の上界を per_joint に max累積する。
  void reach_bound(int link, float local_norm, float* per_joint) const;

  // GPU用のフラットな書き出し。step_meta: 4 int/step (parent, link, type[0=fixed,1=revolute,2=prismatic], active)。
  // step_data: 16 float/step = axis(xyz,0), ro row0(xyz)+to.x, ro row1(xyz)+to.y, ro row2(xyz)+to.z。
  void export_steps(std::vector<int32_t>& step_meta, std::vector<float>& step_data, std::vector<int32_t>& roots) const;

  // tip_linkまでの経路上にあるactive joint index(昇順)。
  const std::vector<int>& chain_joints(int tip_link) const { return chain_joints_[static_cast<size_t>(tip_link)]; }

 private:
  struct Step {
    int parent;
    int link;
    float ro[9];  // joint originの回転
    float to[3];  // joint originの並進
    float axis[3];
    float a0[9], ac[9], as[9];  // revolute: ro * Rot(axis,theta) = a0 + cos*ac + sin*as
    uint8_t type;  // 0=fixed 1=revolute 2=prismatic
    int active;    // q内のindex、fixedは-1
  };
  size_t dof_       = 0;
  size_t num_links_ = 0;
  std::vector<Step> steps_;
  std::vector<std::string> link_names_;
  std::vector<int> step_of_link_;                  // linkを子とするstepのindex、rootは-1
  std::vector<std::vector<int>> chain_joints_;     // link -> active joint indices
  std::vector<int> root_links_;
};

}  // namespace vkmotion
