#include "fast_collision.hpp"
#include "gpu_codegen.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <immintrin.h>
#include <thread>

namespace vkmotion {

thread_local uint64_t g_check_count = 0;
thread_local uint64_t g_block_count = 0;
thread_local uint64_t g_edge_count  = 0;
uint64_t FastCollision::debug_check_count() { return g_check_count; }
uint64_t FastCollision::debug_block_count() { return g_block_count; }
uint64_t FastCollision::debug_edge_count() { return g_edge_count; }

struct FastCollision::Scratch {
  std::vector<Frame> frames;
  std::vector<float> sx, sy, sz;
  std::vector<float> bx, by, bz;  // 境界球中心(world)
  std::vector<int> queue;         // motion_free用
  std::vector<int> order;
  std::vector<float> block;
};

struct FastCollision::Scratch8 {
  std::vector<Frame8> frames;
  std::vector<v8f> sx, sy, sz, bx, by, bz, gx, gy, gz, sin_q, cos_q, q;
};

FastCollision::Scratch8& FastCollision::scratch8(size_t num_links, size_t num_spheres, size_t num_link_spheres, size_t dof) {
  thread_local Scratch8 sc;
  if(sc.frames.size() < num_links) sc.frames.resize(num_links);
  if(sc.sx.size() < num_spheres) {
    sc.sx.resize(num_spheres);
    sc.sy.resize(num_spheres);
    sc.sz.resize(num_spheres);
  }
  if(sc.bx.size() < num_link_spheres) {
    sc.bx.resize(num_link_spheres);
    sc.by.resize(num_link_spheres);
    sc.bz.resize(num_link_spheres);
  }
  if(sc.gx.size() < num_links) {
    sc.gx.resize(num_links);
    sc.gy.resize(num_links);
    sc.gz.resize(num_links);
  }
  if(sc.sin_q.size() < dof) {
    sc.sin_q.resize(dof);
    sc.cos_q.resize(dof);
    sc.q.resize(dof);
  }
  return sc;
}

FastCollision::Scratch& FastCollision::scratch(size_t num_links, size_t num_spheres) {
  thread_local Scratch sc;
  if(sc.frames.size() < num_links) sc.frames.resize(num_links);
  if(sc.sx.size() < num_spheres) {
    sc.sx.resize(num_spheres);
    sc.sy.resize(num_spheres);
    sc.sz.resize(num_spheres);
  }
  return sc;
}

namespace {

void quat_to_rot_transposed(const cutil::Quatf& q, float* rt) {
  // world->local = R^T。R: 正規化したクォータニオンの回転行列
  float n = std::sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
  if(n < 1e-12f) n = 1.0f;
  const float w = q.w / n, x = q.x / n, y = q.y / n, z = q.z / n;
  float r[9] = {1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w),
                2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)};
  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 3; j++) rt[i * 3 + j] = r[j * 3 + i];
}

inline float dist_box(const float* c, const float* rt, const float* h, float x, float y, float z) {
  const float dx = x - c[0], dy = y - c[1], dz = z - c[2];
  const float lx = rt[0] * dx + rt[1] * dy + rt[2] * dz;
  const float ly = rt[3] * dx + rt[4] * dy + rt[5] * dz;
  const float lz = rt[6] * dx + rt[7] * dy + rt[8] * dz;
  const float qx = std::fabs(lx) - h[0], qy = std::fabs(ly) - h[1], qz = std::fabs(lz) - h[2];
  const float ox = std::max(qx, 0.0f), oy = std::max(qy, 0.0f), oz = std::max(qz, 0.0f);
  return std::sqrt(ox * ox + oy * oy + oz * oz) + std::min(std::max(qx, std::max(qy, qz)), 0.0f);
}

inline float dist_ball(const float* c, float r, float x, float y, float z) {
  const float dx = x - c[0], dy = y - c[1], dz = z - c[2];
  return std::sqrt(dx * dx + dy * dy + dz * dz) - r;
}

inline float dist_cyl(const float* c, const float* rt, float r, float half_h, float x, float y, float z) {
  const float dx = x - c[0], dy = y - c[1], dz = z - c[2];
  const float lx = rt[0] * dx + rt[1] * dy + rt[2] * dz;
  const float ly = rt[3] * dx + rt[4] * dy + rt[5] * dz;
  const float lz = rt[6] * dx + rt[7] * dy + rt[8] * dz;
  const float radial   = std::sqrt(lx * lx + ly * ly) - r;
  const float vertical = std::fabs(lz) - half_h;
  const float a = std::max(radial, 0.0f), b = std::max(vertical, 0.0f);
  return std::sqrt(a * a + b * b) + std::min(std::max(radial, vertical), 0.0f);
}

inline float dist_cap(const float* c, const float* rt, const float* base, const float* tip, float r, float x, float y, float z) {
  const float dx = x - c[0], dy = y - c[1], dz = z - c[2];
  const float p[3] = {rt[0] * dx + rt[1] * dy + rt[2] * dz, rt[3] * dx + rt[4] * dy + rt[5] * dz, rt[6] * dx + rt[7] * dy + rt[8] * dz};
  const float ab[3] = {tip[0] - base[0], tip[1] - base[1], tip[2] - base[2]};
  const float len2  = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
  float t = 0.0f;
  if(len2 > 1e-12f) t = std::clamp(((p[0] - base[0]) * ab[0] + (p[1] - base[1]) * ab[1] + (p[2] - base[2]) * ab[2]) / len2, 0.0f, 1.0f);
  const float ex = p[0] - (base[0] + ab[0] * t), ey = p[1] - (base[1] + ab[1] * t), ez = p[2] - (base[2] + ab[2] * t);
  return std::sqrt(ex * ex + ey * ey + ez * ez) - r;
}

inline float dist_plane(const float* c, const float* n, float x, float y, float z) {
  return (x - c[0]) * n[0] + (y - c[1]) * n[1] + (z - c[2]) * n[2];
}

}  // namespace

FastCollision::FastCollision(const RobotConfig& robot, const AllowedCollisionMatrix& acm) : kin_(robot.urdf_model()), acm_(acm) {
  const UrdfModel& model = robot.urdf_model();

  // 球をリンクごとにまとめる
  std::vector<std::vector<size_t>> by_link(kin_.num_links());
  const auto& spheres = robot.collision_spheres_local();
  for(size_t i = 0; i < spheres.size(); i++) {
    int li = kin_.link_index(spheres[i].link_name);
    if(li >= 0) by_link[static_cast<size_t>(li)].push_back(i);
  }
  // 球を最大kMaxCluster個ずつの空間クラスタに分け、クラスタごとに境界球を持つ(広いリンクでも境界球が小さく保たれ、足切りが効く)。
  constexpr size_t kMaxCluster = 4;
  std::function<void(size_t, std::vector<size_t>)> add_cluster = [&](size_t l, std::vector<size_t> idx) {
    if(idx.size() > kMaxCluster) {
      float lo[3] = {1e30f, 1e30f, 1e30f}, hi[3] = {-1e30f, -1e30f, -1e30f};
      for(size_t si : idx)
        for(int k = 0; k < 3; k++) {
          lo[k] = std::min(lo[k], spheres[si].center[k]);
          hi[k] = std::max(hi[k], spheres[si].center[k]);
        }
      int axis = 0;
      for(int k = 1; k < 3; k++)
        if(hi[k] - lo[k] > hi[axis] - lo[axis]) axis = k;
      std::sort(idx.begin(), idx.end(), [&](size_t x, size_t y) { return spheres[x].center[axis] < spheres[y].center[axis]; });
      const size_t half = idx.size() / 2;
      add_cluster(l, std::vector<size_t>(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(half)));
      add_cluster(l, std::vector<size_t>(idx.begin() + static_cast<std::ptrdiff_t>(half), idx.end()));
      return;
    }
    LinkSpheres ls{};
    ls.link     = static_cast<int>(l);
    ls.begin    = static_cast<int>(lx_.size());
    float lo[3] = {1e30f, 1e30f, 1e30f}, hi[3] = {-1e30f, -1e30f, -1e30f};
    for(size_t si : idx) {
      const auto& sp = spheres[si];
      lx_.push_back(sp.center[0]);
      ly_.push_back(sp.center[1]);
      lz_.push_back(sp.center[2]);
      lr_.push_back(sp.radius);
      for(int k = 0; k < 3; k++) {
        lo[k] = std::min(lo[k], sp.center[k]);
        hi[k] = std::max(hi[k], sp.center[k]);
      }
    }
    ls.count = static_cast<int>(idx.size());
    for(int k = 0; k < 3; k++) ls.bs_c[k] = 0.5f * (lo[k] + hi[k]);
    float rad = 0.0f;
    for(int i = ls.begin; i < ls.begin + ls.count; i++) {
      float dx = lx_[i] - ls.bs_c[0], dy = ly_[i] - ls.bs_c[1], dz = lz_[i] - ls.bs_c[2];
      rad      = std::max(rad, std::sqrt(dx * dx + dy * dy + dz * dz) + lr_[i]);
    }
    ls.bs_r = rad * 1.0001f;
    links_.push_back(ls);
    link_names_.push_back(model.links[l].name);
  };
  for(size_t l = 0; l < by_link.size(); l++)
    if(!by_link[l].empty()) add_cluster(l, by_link[l]);
  for(size_t i = 0; i < links_.size();) {
    Group g{};
    g.first = static_cast<int>(i);
    g.link  = links_[i].link;
    size_t j = i;
    float lo[3] = {1e30f, 1e30f, 1e30f}, hi[3] = {-1e30f, -1e30f, -1e30f};
    while(j < links_.size() && links_[j].link == g.link) {
      for(int k = links_[j].begin; k < links_[j].begin + links_[j].count; k++) {
        const float c[3] = {lx_[k], ly_[k], lz_[k]};
        for(int a = 0; a < 3; a++) {
          lo[a] = std::min(lo[a], c[a]);
          hi[a] = std::max(hi[a], c[a]);
        }
      }
      j++;
    }
    g.count = static_cast<int>(j - i);
    for(int a = 0; a < 3; a++) g.bs_c[a] = 0.5f * (lo[a] + hi[a]);
    float rad = 0.0f;
    for(size_t c = i; c < j; c++)
      for(int k = links_[c].begin; k < links_[c].begin + links_[c].count; k++) {
        float dx = lx_[k] - g.bs_c[0], dy = ly_[k] - g.bs_c[1], dz = lz_[k] - g.bs_c[2];
        rad      = std::max(rad, std::sqrt(dx * dx + dy * dy + dz * dz) + lr_[k]);
      }
    g.bs_r = rad * 1.0001f;
    groups_.push_back(g);
    i = j;
  }
  n_spheres_ = lx_.size();
  r_min_     = n_spheres_ ? *std::min_element(lr_.begin(), lr_.end()) : 0.05f;

  // 自己衝突ペア: 別リンク・隣接(共有jointを持つ)でない・ACM許可でない
  auto adjacent = [&](const std::string& a, const std::string& b) {
    for(const auto& j : model.joints)
      if((j.parent_link == a && j.child_link == b) || (j.parent_link == b && j.child_link == a)) return true;
    return false;
  };
  for(size_t a = 0; a < links_.size(); a++) {
    for(size_t b = a + 1; b < links_.size(); b++) {
      if(link_names_[a] == link_names_[b]) continue;
      if(adjacent(link_names_[a], link_names_[b])) continue;
      if(acm_.is_allowed(link_names_[a], link_names_[b])) continue;
      pairs_.emplace_back(static_cast<int>(a), static_cast<int>(b));
    }
  }

  // 各active jointについて、そのjoint軸まわりの回転で球中心が動く距離の上界(rad->m換算)
  joint_reach_.assign(kin_.dof(), 0.0f);
  for(const auto& ls : links_) {
    for(int i = ls.begin; i < ls.begin + ls.count; i++) {
      float norm = std::sqrt(lx_[i] * lx_[i] + ly_[i] * ly_[i] + lz_[i] * lz_[i]);
      std::vector<float> b(kin_.dof(), 0.0f);
      kin_.reach_bound(ls.link, norm, b.data());
      for(size_t j = 0; j < b.size(); j++) joint_reach_[j] = std::max(joint_reach_[j], b[j]);
    }
  }

  // ロボット専用のSIMDカーネルを生成・コンパイルする(初回は数秒、以降はディスクのキャッシュ)。使えなければ汎用のmask8_implのまま。
  GpuRobotData rd;
  if(pack_gpu_robot(rd)) jit_ = load_cpu_kernel(rd, kin_.dof());
}

bool FastCollision::pack_gpu_robot(GpuRobotData& o) const {
  if(kin_.dof() > 16 || kin_.num_links() > 32 || groups_.size() > 16 || n_spheres_ > 96 || links_.size() > 40) return false;
  o               = GpuRobotData{};
  o.dof           = static_cast<int>(kin_.dof());
  o.num_links     = static_cast<int>(kin_.num_links());
  o.num_clusters  = static_cast<int>(links_.size());
  o.num_groups    = static_cast<int>(groups_.size());
  o.num_spheres   = static_cast<int>(n_spheres_);
  kin_.export_steps(o.step_meta, o.step_data, o.roots);
  {
    // GPU用: ルートリンクを「単位姿勢で開始」するtype=3のステップとして先頭に足す。各ステップのaxis.wに、そのリンクのグループ番号(無ければ-1)をビット列で入れる
    // (シェーダはFKで姿勢が確定した直後に、そのリンクの境界球・衝突球の世界座標を計算する)。
    std::vector<int> group_of_link(kin_.num_links(), -1);
    for(size_t g = 0; g < groups_.size(); g++) group_of_link[static_cast<size_t>(groups_[g].link)] = static_cast<int>(g);
    auto bits = [](int v) {
      float f;
      std::memcpy(&f, &v, sizeof(f));
      return f;
    };
    std::vector<int32_t> meta;
    std::vector<float> data;
    for(int r : o.roots) {
      meta.insert(meta.end(), {r, r, 3, -1});
      data.insert(data.end(), {0.0f, 0.0f, 0.0f, bits(group_of_link[static_cast<size_t>(r)]), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
    }
    meta.insert(meta.end(), o.step_meta.begin(), o.step_meta.end());
    data.insert(data.end(), o.step_data.begin(), o.step_data.end());
    for(size_t s = 0; s < meta.size() / 4; s++) {
      const int link = meta[s * 4 + 1];
      if(meta[s * 4 + 2] != 3) data[s * 16 + 3] = bits(group_of_link[static_cast<size_t>(link)]);
    }
    o.step_meta = std::move(meta);
    o.step_data = std::move(data);
  }
  o.num_steps = static_cast<int>(o.step_meta.size() / 4);
  for(size_t i = 0; i < n_spheres_; i++) o.spheres.insert(o.spheres.end(), {lx_[i], ly_[i], lz_[i], lr_[i]});
  std::vector<int> group_of(links_.size(), 0);
  for(size_t g = 0; g < groups_.size(); g++)
    for(int c = groups_[g].first; c < groups_[g].first + groups_[g].count; c++) group_of[static_cast<size_t>(c)] = static_cast<int>(g);
  for(size_t c = 0; c < links_.size(); c++) {
    const LinkSpheres& l = links_[c];
    o.cluster_meta.insert(o.cluster_meta.end(), {l.link, l.begin, l.count, group_of[c]});
    o.cluster_bs.insert(o.cluster_bs.end(), {l.bs_c[0], l.bs_c[1], l.bs_c[2], l.bs_r});
  }
  for(const Group& g : groups_) {
    o.group_meta.insert(o.group_meta.end(), {g.first, g.count, g.link, 0});
    o.group_bs.insert(o.group_bs.end(), {g.bs_c[0], g.bs_c[1], g.bs_c[2], g.bs_r});
  }
  // 自己衝突はリンク単位(pairs_はあるリンクペアのクラスタ全組み合わせ)なので、グループごとのビットマスクで渡す
  o.pairs.assign(groups_.size(), 0);
  for(size_t k = 0; k < pairs_.size(); k++) {
    const int ga = group_of[static_cast<size_t>(pairs_[k].first)], gb = group_of[static_cast<size_t>(pairs_[k].second)];
    o.pairs[static_cast<size_t>(std::min(ga, gb))] |= 1 << std::max(ga, gb);
    o.cluster_pairs.insert(o.cluster_pairs.end(), {pairs_[k].first, pairs_[k].second});
  }
  o.num_pairs = static_cast<int>(groups_.size());
  return o.num_steps <= 24;
}

void FastCollision::pack_gpu_world(GpuWorldData& o) const {
  o = GpuWorldData{};
  auto mask_of = [&](size_t obs) {
    uint32_t m = 0;
    for(size_t g = 0; g < groups_.size(); g++)
      if(allowed_[obs][static_cast<size_t>(groups_[g].first)]) m |= 1u << g;
    float f;
    std::memcpy(&f, &m, sizeof(f));
    return f;
  };
  // 箱はロボットのベース(原点)からの最短距離の昇順に並べる。球の原点からの距離+半径が箱の最短距離より小さければ、それ以降の箱は遠すぎて当たらない(VAMPと同じ打ち切り)。
  // 最短距離(保守的な下界)は rt行0 の w に入れる。
  auto min_dist = [&](const Box& b) {
    const float c  = std::sqrt(b.c[0] * b.c[0] + b.c[1] * b.c[1] + b.c[2] * b.c[2]);
    const float h  = std::sqrt(b.half[0] * b.half[0] + b.half[1] * b.half[1] + b.half[2] * b.half[2]);
    return std::max(0.0f, c - h);
  };
  std::vector<size_t> order(boxes_.size());
  for(size_t i = 0; i < order.size(); i++) order[i] = i;
  std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return min_dist(boxes_[a]) < min_dist(boxes_[b]); });
  for(size_t k : order) {
    const Box& b = boxes_[k];
    o.boxes.insert(o.boxes.end(), {b.c[0], b.c[1], b.c[2], mask_of(k)});
    // rt行1のwは「world軸に揃った箱(回転が単位行列)」のフラグ。専用カーネルはその箱の距離計算から回転を省く
    bool aligned = true;
    for(int k = 0; k < 9; k++) aligned = aligned && std::fabs(b.rot[k] - ((k % 4 == 0) ? 1.0f : 0.0f)) < 1e-6f;
    o.boxes.insert(o.boxes.end(), {b.rot[0], b.rot[1], b.rot[2], min_dist(b)});
    o.boxes.insert(o.boxes.end(), {b.rot[3], b.rot[4], b.rot[5], aligned ? 1.0f : 0.0f});
    o.boxes.insert(o.boxes.end(), {b.rot[6], b.rot[7], b.rot[8], 0.0f});
    o.boxes.insert(o.boxes.end(), {b.half[0], b.half[1], b.half[2], 0.0f});
    o.boxes.insert(o.boxes.end(), {b.aabb_half[0], b.aabb_half[1], b.aabb_half[2], 0.0f});
  }
  o.num_boxes = static_cast<int>(boxes_.size());
  size_t obs  = boxes_.size();
  for(const Ball& b : balls_) {
    o.balls.insert(o.balls.end(), {b.c[0], b.c[1], b.c[2], b.r});
    o.balls.insert(o.balls.end(), {mask_of(obs++), 0.0f, 0.0f, 0.0f});
  }
  o.num_balls = static_cast<int>(balls_.size());
}

bool FastCollision::set_world(const WorldConfig& world) {
  boxes_.clear();
  balls_.clear();
  cyls_.clear();
  caps_.clear();
  planes_.clear();
  allowed_.clear();
  if(!world.convex_mesh().empty() || !world.sdf_grid().empty() || !world.attached_objects().empty()) return false;

  auto push_allowed = [&](const std::string& name) {
    std::vector<uint8_t> row(links_.size());
    for(size_t i = 0; i < links_.size(); i++) row[i] = acm_.is_allowed(link_names_[i], name) ? 1 : 0;
    allowed_.push_back(std::move(row));
  };

  for(const auto& o : world.cuboid()) {
    Box b{};
    for(int k = 0; k < 3; k++) {
      b.c[k]    = o.pose.position[k];
      b.half[k] = 0.5f * o.dims[k];
    }
    quat_to_rot_transposed(o.pose.quaternion, b.rot);
    // 回転後のworld軸AABBの半径 = |R| * half  (b.rotはworld->localなので転置の絶対値を使う)
    for(int i = 0; i < 3; i++) b.aabb_half[i] = std::fabs(b.rot[0 * 3 + i]) * b.half[0] + std::fabs(b.rot[1 * 3 + i]) * b.half[1] + std::fabs(b.rot[2 * 3 + i]) * b.half[2];
    boxes_.push_back(b);
    push_allowed(o.name);
  }
  for(const auto& o : world.sphere()) {
    Ball b{};
    for(int k = 0; k < 3; k++) b.c[k] = o.pose.position[k];
    b.r = o.radius;
    balls_.push_back(b);
    push_allowed(o.name);
  }
  for(const auto& o : world.cylinder()) {
    Cyl c{};
    for(int k = 0; k < 3; k++) c.c[k] = o.pose.position[k];
    quat_to_rot_transposed(o.pose.quaternion, c.rot);
    c.r      = o.radius;
    c.half_h = 0.5f * o.height;
    cyls_.push_back(c);
    push_allowed(o.name);
  }
  for(const auto& o : world.capsule()) {
    Cap c{};
    for(int k = 0; k < 3; k++) {
      c.c[k]    = o.pose.position[k];
      c.base[k] = o.base[k];
      c.tip[k]  = o.tip[k];
    }
    quat_to_rot_transposed(o.pose.quaternion, c.rot);
    c.r = o.radius;
    caps_.push_back(c);
    push_allowed(o.name);
  }
  for(const auto& o : world.plane()) {
    Pln p{};
    for(int k = 0; k < 3; k++) p.c[k] = o.pose.position[k];
    float rt[9];
    quat_to_rot_transposed(o.pose.quaternion, rt);
    // 法線 = R * (0,0,1) = R^Tの転置の第3列 = rtの第3行
    p.n[0] = rt[6];
    p.n[1] = rt[7];
    p.n[2] = rt[8];
    planes_.push_back(p);
    push_allowed(o.name);
  }
  {
    GpuWorldData wd;  // ロボット専用カーネルに渡す箱・球の配列(GPUと同じ並び)
    pack_gpu_world(wd);
    jit_boxes_     = std::move(wd.boxes);
    jit_balls_     = std::move(wd.balls);
    jit_num_boxes_ = wd.num_boxes;
    jit_num_balls_ = wd.num_balls;
  }
  return true;
}

void FastCollision::world_positions(const float* q, Scratch& sc) const {
  kin_.forward(q, sc.frames.data());
  if(sc.bx.size() < links_.size()) {
    sc.bx.resize(links_.size());
    sc.by.resize(links_.size());
    sc.bz.resize(links_.size());
  }
  for(size_t l = 0; l < links_.size(); l++) {
    const LinkSpheres& ls = links_[l];
    const Frame& f        = sc.frames[static_cast<size_t>(ls.link)];
    for(int i = ls.begin; i < ls.begin + ls.count; i++) {
      sc.sx[i] = f.r[0] * lx_[i] + f.r[1] * ly_[i] + f.r[2] * lz_[i] + f.p[0];
      sc.sy[i] = f.r[3] * lx_[i] + f.r[4] * ly_[i] + f.r[5] * lz_[i] + f.p[1];
      sc.sz[i] = f.r[6] * lx_[i] + f.r[7] * ly_[i] + f.r[8] * lz_[i] + f.p[2];
    }
    sc.bx[l] = f.r[0] * ls.bs_c[0] + f.r[1] * ls.bs_c[1] + f.r[2] * ls.bs_c[2] + f.p[0];
    sc.by[l] = f.r[3] * ls.bs_c[0] + f.r[4] * ls.bs_c[1] + f.r[5] * ls.bs_c[2] + f.p[1];
    sc.bz[l] = f.r[6] * ls.bs_c[0] + f.r[7] * ls.bs_c[1] + f.r[8] * ls.bs_c[2] + f.p[2];
  }
}

bool FastCollision::world_free(const Scratch& sc, float margin) const {
  size_t obs = 0;
  // d_fn(x,y,z): 障害物表面までの符号付き距離(球半径を引く前)
  auto check = [&](size_t obs_idx, auto&& d_fn) -> bool {
    const std::vector<uint8_t>& allow = allowed_[obs_idx];
    for(size_t l = 0; l < links_.size(); l++) {
      if(allow[l]) continue;
      const LinkSpheres& ls = links_[l];
      if(d_fn(sc.bx[l], sc.by[l], sc.bz[l]) - ls.bs_r - margin > 0.0f) continue;  // リンク全体が障害物から離れている
      for(int i = ls.begin; i < ls.begin + ls.count; i++)
        if(d_fn(sc.sx[i], sc.sy[i], sc.sz[i]) - lr_[i] - margin <= 0.0f) return false;
    }
    return true;
  };
  for(const Box& b : boxes_) {
    if(!check(obs++, [&](float x, float y, float z) { return dist_box(b.c, b.rot, b.half, x, y, z); })) return false;
  }
  for(const Ball& b : balls_) {
    if(!check(obs++, [&](float x, float y, float z) { return dist_ball(b.c, b.r, x, y, z); })) return false;
  }
  for(const Cyl& c : cyls_) {
    if(!check(obs++, [&](float x, float y, float z) { return dist_cyl(c.c, c.rot, c.r, c.half_h, x, y, z); })) return false;
  }
  for(const Cap& c : caps_) {
    if(!check(obs++, [&](float x, float y, float z) { return dist_cap(c.c, c.rot, c.base, c.tip, c.r, x, y, z); })) return false;
  }
  for(const Pln& p : planes_) {
    if(!check(obs++, [&](float x, float y, float z) { return dist_plane(p.c, p.n, x, y, z); })) return false;
  }
  return true;
}

bool FastCollision::self_free(const Scratch& sc, float margin) const {
  for(const auto& pr : pairs_) {
    const LinkSpheres& a = links_[static_cast<size_t>(pr.first)];
    const LinkSpheres& b = links_[static_cast<size_t>(pr.second)];
    float dx = sc.bx[pr.first] - sc.bx[pr.second], dy = sc.by[pr.first] - sc.by[pr.second], dz = sc.bz[pr.first] - sc.bz[pr.second];
    float rr = a.bs_r + b.bs_r + 2.0f * margin;
    if(dx * dx + dy * dy + dz * dz > rr * rr) continue;
    for(int i = a.begin; i < a.begin + a.count; i++) {
      for(int j = b.begin; j < b.begin + b.count; j++) {
        float ex = sc.sx[i] - sc.sx[j], ey = sc.sy[i] - sc.sy[j], ez = sc.sz[i] - sc.sz[j];
        float rs = lr_[i] + lr_[j] + 2.0f * margin;
        if(ex * ex + ey * ey + ez * ez <= rs * rs) return false;
      }
    }
  }
  return true;
}

bool FastCollision::free_impl(const float* q, float margin) const {
  g_check_count++;
  if(simd_supported()) return (mask8_impl(q, margin, true, 1u) & 1u) != 0;
  return free_scalar(q, margin);
}

bool FastCollision::free_scalar(const float* q, float margin) const {
  Scratch& sc = scratch(kin_.num_links(), n_spheres_);
  world_positions(q, sc);
  return world_free(sc, margin) && self_free(sc, margin);
}

bool FastCollision::is_free(const float* q) const { return free_impl(q, 0.0f); }

bool FastCollision::is_free_world_only(const float* q) const {
  Scratch& sc = scratch(kin_.num_links(), n_spheres_);
  world_positions(q, sc);
  return world_free(sc, 0.0f);
}

int FastCollision::motion_steps(const float* a, const float* b, float step_override) const {
  float disp = 0.0f;
  for(size_t j = 0; j < kin_.dof(); j++) disp += std::fabs(b[j] - a[j]) * joint_reach_[j];
  const float step = std::max(step_override > 0.0f ? step_override : motion_step_, 1e-3f);
  return std::max(1, static_cast<int>(std::ceil(disp / step)));
}

namespace {

// 箱(中心c, world->local回転rt, 半径h)までの距離^2(外側のみ。内部なら0)
inline v8f box_sumsq(const float* c, const float* rt, const float* h, v8f x, v8f y, v8f z) {
  const v8f dx = x - c[0], dy = y - c[1], dz = z - c[2];
  const v8f lx = rt[0] * dx + rt[1] * dy + rt[2] * dz;
  const v8f ly = rt[3] * dx + rt[4] * dy + rt[5] * dz;
  const v8f lz = rt[6] * dx + rt[7] * dy + rt[8] * dz;
  const v8i abs_mask = {0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff};
  const v8f ax = reinterpret_cast<v8f>(reinterpret_cast<v8i>(lx) & abs_mask) - h[0];
  const v8f ay = reinterpret_cast<v8f>(reinterpret_cast<v8i>(ly) & abs_mask) - h[1];
  const v8f az = reinterpret_cast<v8f>(reinterpret_cast<v8i>(lz) & abs_mask) - h[2];
  const v8f zero = {};
  const v8f ox = ax > zero ? ax : zero;
  const v8f oy = ay > zero ? ay : zero;
  const v8f oz = az > zero ? az : zero;
  return ox * ox + oy * oy + oz * oz;
}

}  // namespace

uint32_t FastCollision::mask8_impl(const float* q, float margin, bool broadcast, uint32_t required) const {
  g_block_count++;
  const size_t dof = kin_.dof();
  if(jit_ && simd_supported()) {
    float qb[8 * 16];
    const float* qq = q;
    if(broadcast) {
      for(size_t lane = 0; lane < 8; lane++)
        for(size_t j = 0; j < dof; j++) qb[lane * dof + j] = q[j];
      qq = qb;
    }
    return jit_(qq, margin, required, jit_boxes_.data(), jit_num_boxes_, jit_balls_.data(), jit_num_balls_);
  }
  if(!simd_supported()) {
    if(broadcast) return free_scalar(q, margin) ? 0xFFu : 0u;
    uint32_t m = 0;
    for(int lane = 0; lane < 8; lane++)
      if(free_scalar(q + static_cast<size_t>(lane) * dof, margin)) m |= 1u << lane;
    return m;
  }
  Scratch8& sc = scratch8(kin_.num_links(), n_spheres_, links_.size(), dof);
  if(broadcast) {
    for(size_t j = 0; j < dof; j++) {
      float sn, cs;
      sincosf(q[j], &sn, &cs);
      sc.q[j]     = v8f{} + q[j];
      sc.sin_q[j] = v8f{} + sn;
      sc.cos_q[j] = v8f{} + cs;
    }
  } else {
    for(size_t j = 0; j < dof; j++) {
      for(int lane = 0; lane < 8; lane++) sc.q[j][lane] = q[static_cast<size_t>(lane) * dof + j];
      sincos8(sc.q[j], &sc.sin_q[j], &sc.cos_q[j]);
    }
  }
  kin_.forward8(sc.sin_q.data(), sc.cos_q.data(), sc.q.data(), sc.frames.data());

  for(size_t l = 0; l < links_.size(); l++) {
    const LinkSpheres& ls = links_[l];
    const Frame8& f       = sc.frames[static_cast<size_t>(ls.link)];
    for(int i = ls.begin; i < ls.begin + ls.count; i++) {
      sc.sx[i] = f.r[0] * lx_[i] + f.r[1] * ly_[i] + f.r[2] * lz_[i] + f.p[0];
      sc.sy[i] = f.r[3] * lx_[i] + f.r[4] * ly_[i] + f.r[5] * lz_[i] + f.p[1];
      sc.sz[i] = f.r[6] * lx_[i] + f.r[7] * ly_[i] + f.r[8] * lz_[i] + f.p[2];
    }
    sc.bx[l] = f.r[0] * ls.bs_c[0] + f.r[1] * ls.bs_c[1] + f.r[2] * ls.bs_c[2] + f.p[0];
    sc.by[l] = f.r[3] * ls.bs_c[0] + f.r[4] * ls.bs_c[1] + f.r[5] * ls.bs_c[2] + f.p[1];
    sc.bz[l] = f.r[6] * ls.bs_c[0] + f.r[7] * ls.bs_c[1] + f.r[8] * ls.bs_c[2] + f.p[2];
  }

  for(size_t g = 0; g < groups_.size(); g++) {
    const Group& gr = groups_[g];
    const Frame8& f = sc.frames[static_cast<size_t>(gr.link)];
    sc.gx[g] = f.r[0] * gr.bs_c[0] + f.r[1] * gr.bs_c[1] + f.r[2] * gr.bs_c[2] + f.p[0];
    sc.gy[g] = f.r[3] * gr.bs_c[0] + f.r[4] * gr.bs_c[1] + f.r[5] * gr.bs_c[2] + f.p[1];
    sc.gz[g] = f.r[6] * gr.bs_c[0] + f.r[7] * gr.bs_c[1] + f.r[8] * gr.bs_c[2] + f.p[2];
  }

  uint32_t alive = 0xFFu;
  size_t obs     = 0;
  // requiredのレーンがひとつでも衝突した時点で呼び出し側の結果は決まる(辺の検査など)ので打ち切る。required=0なら全レーンが衝突したときだけ打ち切る。
  const auto dead = [required](uint32_t a) { return required ? (a & required) != required : a == 0; };
  const v8i abs_mask = {0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff};
  for(const Box& b : boxes_) {
    const std::vector<uint8_t>& allow = allowed_[obs++];
    for(size_t g = 0; g < groups_.size(); g++) {
      const Group& gr = groups_[g];
      if(allow[static_cast<size_t>(gr.first)]) continue;  // 同じリンクのクラスタは許可も共通
      // 1段目: リンク全体の境界球 vs 箱のworld軸AABB(保守的。軸平行な箱ならexact)
      const float rr = gr.bs_r + margin;
      const v8f ddx = reinterpret_cast<v8f>(reinterpret_cast<v8i>(sc.gx[g] - b.c[0]) & abs_mask);
      const v8f ddy = reinterpret_cast<v8f>(reinterpret_cast<v8i>(sc.gy[g] - b.c[1]) & abs_mask);
      const v8f ddz = reinterpret_cast<v8f>(reinterpret_cast<v8i>(sc.gz[g] - b.c[2]) & abs_mask);
      if(!(movemask((ddx <= b.aabb_half[0] + rr) & (ddy <= b.aabb_half[1] + rr) & (ddz <= b.aabb_half[2] + rr)) & alive)) continue;
      for(int c = gr.first; c < gr.first + gr.count; c++) {
        const LinkSpheres& ls = links_[static_cast<size_t>(c)];
        const float rc        = ls.bs_r + margin;
        // 2段目: クラスタの境界球 vs 箱(OBB)
        if(!(movemask(box_sumsq(b.c, b.rot, b.half, sc.bx[c], sc.by[c], sc.bz[c]) <= rc * rc) & alive)) continue;
        for(int i = ls.begin; i < ls.begin + ls.count; i++) {
          const float r = lr_[i] + margin;
          alive &= ~movemask(box_sumsq(b.c, b.rot, b.half, sc.sx[i], sc.sy[i], sc.sz[i]) <= r * r);
        }
        if(dead(alive)) return 0;
      }
    }
  }
  for(const Ball& b : balls_) {
    const std::vector<uint8_t>& allow = allowed_[obs++];
    for(size_t l = 0; l < links_.size(); l++) {
      if(allow[l]) continue;
      const LinkSpheres& ls = links_[l];
      for(int i = ls.begin; i < ls.begin + ls.count; i++) {
        const v8f dx = sc.sx[i] - b.c[0], dy = sc.sy[i] - b.c[1], dz = sc.sz[i] - b.c[2];
        const float r = lr_[i] + b.r + margin;
        alive &= ~movemask(dx * dx + dy * dy + dz * dz <= r * r);
      }
      if(dead(alive)) return 0;
    }
  }

  for(const auto& pr : pairs_) {
    const LinkSpheres& a = links_[static_cast<size_t>(pr.first)];
    const LinkSpheres& b = links_[static_cast<size_t>(pr.second)];
    const v8f dx = sc.bx[pr.first] - sc.bx[pr.second], dy = sc.by[pr.first] - sc.by[pr.second], dz = sc.bz[pr.first] - sc.bz[pr.second];
    const float rr = a.bs_r + b.bs_r + 2.0f * margin;
    if(!(movemask(dx * dx + dy * dy + dz * dz <= rr * rr) & alive)) continue;
    for(int i = a.begin; i < a.begin + a.count; i++) {
      for(int j = b.begin; j < b.begin + b.count; j++) {
        const v8f ex = sc.sx[i] - sc.sx[j], ey = sc.sy[i] - sc.sy[j], ez = sc.sz[i] - sc.sz[j];
        const float rs = lr_[i] + lr_[j] + 2.0f * margin;
        alive &= ~movemask(ex * ex + ey * ey + ez * ez <= rs * rs);
      }
    }
    if(dead(alive)) return 0;
  }
  return alive;
}

uint32_t FastCollision::free_mask8(const float* q, float margin) const { return mask8_impl(q, margin, false, 0); }

bool FastCollision::motion_free(const float* a, const float* b, float margin, bool include_end, float step) const {
  g_edge_count++;
  const size_t dof = kin_.dof();
  const int n      = motion_steps(a, b, step);
  if(n <= 1 && !include_end) return true;  // 端点は呼び出し側で検査済みの前提

  Scratch& sc = scratch(kin_.num_links(), n_spheres_);
  // 検査する内部点(1..n-1)を、粗い分割から順(区間の中点の幅優先)に並べる。衝突経路ほど先頭の数点で棄却できる。
  std::vector<int>& order = sc.order;
  order.clear();
  if(include_end) order.push_back(n);
  {
    std::vector<int>& q2 = sc.queue;  // (lo,hi)ペアのキュー
    q2.clear();
    q2.push_back(0);
    q2.push_back(n);
    for(size_t head = 0; head < q2.size();) {
      int lo = q2[head++], hi = q2[head++];
      int mid = (lo + hi) / 2;
      if(mid == lo || mid == hi) continue;
      order.push_back(mid);
      if(mid - lo > 1) {
        q2.push_back(lo);
        q2.push_back(mid);
      }
      if(hi - mid > 1) {
        q2.push_back(mid);
        q2.push_back(hi);
      }
    }
  }

  std::vector<float>& block = sc.block;
  if(block.size() < 8 * dof) block.resize(8 * dof);
  for(size_t k = 0; k < order.size(); k += 8) {
    const size_t cnt = std::min<size_t>(8, order.size() - k);
    for(size_t lane = 0; lane < 8; lane++) {
      const int idx = order[k + std::min(lane, cnt - 1)];
      const float t = static_cast<float>(idx) / static_cast<float>(n);
      for(size_t j = 0; j < dof; j++) block[lane * dof + j] = a[j] + t * (b[j] - a[j]);
    }
    const uint32_t real = (1u << cnt) - 1u;
    if((mask8_impl(block.data(), margin, false, real) & real) != real) return false;
  }
  return true;
}

void FastCollision::is_free_batch(const float* qs, size_t n, uint8_t* out, int threads) const {
  const size_t dof = kin_.dof();
  auto work = [&](size_t begin, size_t end) {
    std::vector<float> block(8 * dof);
    for(size_t i = begin; i < end; i += 8) {
      const size_t cnt = std::min<size_t>(8, end - i);
      for(size_t lane = 0; lane < 8; lane++) std::copy(qs + (i + std::min(lane, cnt - 1)) * dof, qs + (i + std::min(lane, cnt - 1)) * dof + dof, block.begin() + static_cast<std::ptrdiff_t>(lane * dof));
      const uint32_t m = free_mask8(block.data(), 0.0f);
      for(size_t lane = 0; lane < cnt; lane++) out[i + lane] = (m >> lane) & 1u;
    }
  };
  if(threads <= 1 || n < 64) {
    work(0, n);
    return;
  }
  std::vector<std::thread> pool;
  const size_t chunk = (n + static_cast<size_t>(threads) - 1) / static_cast<size_t>(threads);
  for(int t = 0; t < threads; t++) {
    size_t b = static_cast<size_t>(t) * chunk, e = std::min(n, b + chunk);
    if(b >= e) break;
    pool.emplace_back(work, b, e);
  }
  for(auto& th : pool) th.join();
}

}  // namespace vkmotion
