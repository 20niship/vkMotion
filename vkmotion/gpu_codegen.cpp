#include "gpu_codegen.hpp"

#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <dlfcn.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <vkmotion/vkmotion_config.hpp>

namespace vkmotion {

namespace {

std::string lit(float v) {
  if(v == 0.0f) return "0.0";  // -0.0 も 0.0 に(式の符号反転を避ける)
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.9g", static_cast<double>(v));
  std::string s = buf;
  if(s.find_first_of(".eEn") == std::string::npos) s += ".0";
  return s;
}

std::string vec3_lit(float x, float y, float z) { return "vec3(" + lit(x) + ", " + lit(y) + ", " + lit(z) + ")"; }

int bits_to_int(float f) {
  int v;
  std::memcpy(&v, &f, sizeof(v));
  return v;
}

std::string read_text(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

uint64_t fnv1a(const std::string& s, uint64_t h = 1469598103934665603ull) {
  for(unsigned char c : s) {
    h ^= c;
    h *= 1099511628211ull;
  }
  return h;
}

bool mkdir_p(const std::string& path) {
  std::string cur;
  for(size_t i = 0; i <= path.size(); i++) {
    if(i == path.size() || path[i] == '/') {
      if(!cur.empty() && ::mkdir(cur.c_str(), 0755) != 0 && errno != EEXIST) return false;
    }
    if(i < path.size()) cur += path[i];
  }
  return true;
}

std::string cache_dir() {
  if(const char* e = std::getenv("VKMOTION_SHADER_CACHE")) return e;
  if(const char* e = std::getenv("XDG_CACHE_HOME")) return std::string(e) + "/vkmotion";
  if(const char* e = std::getenv("HOME")) return std::string(e) + "/.cache/vkmotion";
  return "/tmp/vkmotion-cache";
}

bool file_exists(const std::string& p) {
  struct stat st;
  return ::stat(p.c_str(), &st) == 0;
}

}  // namespace

std::string generate_robot_glsl(const GpuRobotData& r) {
  std::ostringstream o;
  const int ns = r.num_steps;
  // 衝突球の世界座標の式(リンク姿勢はレジスタにあるので、必要なときだけその場で計算する)
  auto sphere_pos = [&](int cluster, int i) {
    const int link = r.cluster_meta[static_cast<size_t>(cluster * 4)];
    const float* sp = &r.spheres[static_cast<size_t>(i * 4)];
    return "rot_apply(R" + std::to_string(link) + ", " + vec3_lit(sp[0], sp[1], sp[2]) + ") + P" + std::to_string(link);
  };
  auto sphere_r = [&](int i) { return r.spheres[static_cast<size_t>(i * 4 + 3)]; };
  o << "// 自動生成(vkmotion/gpu_codegen.cpp)。手で編集しないこと。\n";
  o << "bool config_free() {\n";
  // 関節ごとのsin/cos(回転ジョイントだけ)
  std::set<int> revolute;
  for(int s = 0; s < ns; s++)
    if(r.step_meta[static_cast<size_t>(s * 4 + 2)] == 1) revolute.insert(r.step_meta[static_cast<size_t>(s * 4 + 3)]);
  for(int j : revolute) o << "  float sn" << j << ", cs" << j << ";\n  sincos_fast(Q[" << j << "], sn" << j << ", cs" << j << ");\n";
  // リンク姿勢と境界球
  std::set<int> links;
  for(int s = 0; s < ns; s++) links.insert(r.step_meta[static_cast<size_t>(s * 4 + 1)]);
  for(int l : links) o << "  float R" << l << "[9];\n  vec3 P" << l << ";\n";
  for(int g = 0; g < r.num_groups; g++) o << "  vec3 G" << g << ";\n";
  for(int c = 0; c < r.num_clusters; c++) o << "  vec3 C" << c << ";\n";

  for(int s = 0; s < ns; s++) {
    const int* m  = &r.step_meta[static_cast<size_t>(s * 4)];
    const float* d = &r.step_data[static_cast<size_t>(s * 16)];
    const int pa = m[0], ch = m[1], type = m[2], joint = m[3];
    const int grp = bits_to_int(d[3]);
    o << "  {\n";
    if(type == 3) {
      o << "    R" << ch << " = float[9](1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);\n    P" << ch << " = vec3(0.0);\n";
    } else {
      const float ro[9] = {d[4], d[5], d[6], d[8], d[9], d[10], d[12], d[13], d[14]};
      const float to[3] = {d[7], d[11], d[15]};
      o << "    float mm[9];\n";
      if(type == 1) {
        const float ax = d[0], ay = d[1], az = d[2];
        o << "    float tt = 1.0 - cs" << joint << ";\n";
        const std::string sn = "sn" + std::to_string(joint), cs = "cs" + std::to_string(joint);
        o << "    float rj[9] = float[9](" << cs << " + " << lit(ax * ax) << " * tt, " << lit(ax * ay) << " * tt - " << lit(az) << " * " << sn << ", " << lit(ax * az) << " * tt + " << lit(ay) << " * " << sn << ",\n"
          << "                           " << lit(ay * ax) << " * tt + " << lit(az) << " * " << sn << ", " << cs << " + " << lit(ay * ay) << " * tt, " << lit(ay * az) << " * tt - " << lit(ax) << " * " << sn << ",\n"
          << "                           " << lit(az * ax) << " * tt - " << lit(ay) << " * " << sn << ", " << lit(az * ay) << " * tt + " << lit(ax) << " * " << sn << ", " << cs << " + " << lit(az * az) << " * tt);\n";
        for(int i = 0; i < 3; i++)
          for(int j = 0; j < 3; j++)
            o << "    mm[" << i * 3 + j << "] = " << lit(ro[i * 3]) << " * rj[" << j << "] + " << lit(ro[i * 3 + 1]) << " * rj[" << 3 + j << "] + " << lit(ro[i * 3 + 2]) << " * rj[" << 6 + j << "];\n";
      } else {
        for(int i = 0; i < 9; i++) o << "    mm[" << i << "] = " << lit(ro[i]) << ";\n";
      }
      // 並進: type==2は軸方向にQ[joint]だけ動く
      std::string tx = lit(to[0]), ty = lit(to[1]), tz = lit(to[2]);
      if(type == 2) {
        tx += " + " + lit(d[0]) + " * Q[" + std::to_string(joint) + "]";
        ty += " + " + lit(d[1]) + " * Q[" + std::to_string(joint) + "]";
        tz += " + " + lit(d[2]) + " * Q[" + std::to_string(joint) + "]";
      }
      o << "    vec3 t = vec3(" << tx << ", " << ty << ", " << tz << ");\n";
      for(int i = 0; i < 3; i++)
        for(int j = 0; j < 3; j++)
          o << "    R" << ch << "[" << i * 3 + j << "] = R" << pa << "[" << i * 3 << "] * mm[" << j << "] + R" << pa << "[" << i * 3 + 1 << "] * mm[" << 3 + j << "] + R" << pa << "[" << i * 3 + 2 << "] * mm[" << 6 + j << "];\n";
      o << "    P" << ch << " = P" << pa << " + rot_apply(R" << pa << ", t);\n";
    }
    // このリンクのグループ(境界球・クラスタ・衝突球)の世界座標
    if(grp >= 0) {
      const int g = grp;
      const float* gb = &r.group_bs[static_cast<size_t>(g * 4)];
      o << "    G" << g << " = rot_apply(R" << ch << ", " << vec3_lit(gb[0], gb[1], gb[2]) << ") + P" << ch << ";\n";
      const int first = r.group_meta[static_cast<size_t>(g * 4)], count = r.group_meta[static_cast<size_t>(g * 4 + 1)];
      for(int c = first; c < first + count; c++) {
        const float* cb = &r.cluster_bs[static_cast<size_t>(c * 4)];
        o << "    C" << c << " = rot_apply(R" << ch << ", " << vec3_lit(cb[0], cb[1], cb[2]) << ") + P" << ch << ";\n";
      }
    }
    o << "  }\n";
  }

  // 直方体: グループの境界球とAABBで候補を絞り(静的に展開)、候補だけ詳細判定
  o << "  for(uint b = 0; b < NUM_BOXES; b++) {\n";
  o << "    const uint bo = BOX_BASE + b * 6u;\n    const uint allow = floatBitsToUint(s_boxes[bo].w);\n    const vec3 center = s_boxes[bo].xyz;\n    const vec3 aabb_half = s_boxes[bo + 5].xyz;\n    uint cand = 0u;\n";
  for(int g = 0; g < r.num_groups; g++) {
    const float rad = r.group_bs[static_cast<size_t>(g * 4 + 3)];
    o << "    if(((allow >> " << g << "u) & 1u) == 0u && all(lessThanEqual(abs(G" << g << " - center) - aabb_half - vec3(" << lit(rad) << " + MARGIN), vec3(0.0)))) cand |= " << (1u << g) << "u;\n";
  }
  for(int g = 0; g < r.num_groups; g++) {
    const int first = r.group_meta[static_cast<size_t>(g * 4)], count = r.group_meta[static_cast<size_t>(g * 4 + 1)];
    o << "    if((cand & " << (1u << g) << "u) != 0u) {\n";
    for(int c = first; c < first + count; c++) {
      const float rc = r.cluster_bs[static_cast<size_t>(c * 4 + 3)];
      o << "      { float rc = " << lit(rc) << " + MARGIN; if(box_sumsq(bo, C" << c << ") <= rc * rc) {\n";
      const int sb = r.cluster_meta[static_cast<size_t>(c * 4 + 1)], sc = r.cluster_meta[static_cast<size_t>(c * 4 + 2)];
      for(int i = sb; i < sb + sc; i++) o << "        { float q = " << lit(sphere_r(i)) << " + MARGIN; if(box_sumsq(bo, " << sphere_pos(c, i) << ") <= q * q) return false; }\n";
      o << "      } }\n";
    }
    o << "    }\n";
  }
  o << "  }\n";
  {
    // 球障害物(数が少ない)はsph[]を引く汎用判定を使うので、あるときだけ全球の世界座標を書き出す
    o << "  if(NUM_BALLS != 0u) {\n";
    for(int c = 0; c < r.num_clusters; c++) {
      const int sb = r.cluster_meta[static_cast<size_t>(c * 4 + 1)], sc = r.cluster_meta[static_cast<size_t>(c * 4 + 2)];
      for(int i = sb; i < sb + sc; i++) o << "    sph[" << i << "] = vec4(" << sphere_pos(c, i) << ", " << lit(sphere_r(i)) << ");\n";
    }
    o << "    if(balls_hit()) return false;\n  }\n";
  }
  // 自己衝突: クラスタペアは静的
  for(size_t k = 0; k + 1 < r.cluster_pairs.size(); k += 2) {
    const int ca = r.cluster_pairs[k], cb = r.cluster_pairs[k + 1];
    const float rc = r.cluster_bs[static_cast<size_t>(ca * 4 + 3)] + r.cluster_bs[static_cast<size_t>(cb * 4 + 3)];
    o << "  { vec3 e = C" << ca << " - C" << cb << "; float q = " << lit(rc) << " + 2.0 * MARGIN; if(dot(e, e) <= q * q) {\n";
    const int sba = r.cluster_meta[static_cast<size_t>(ca * 4 + 1)], sca = r.cluster_meta[static_cast<size_t>(ca * 4 + 2)];
    const int sbb = r.cluster_meta[static_cast<size_t>(cb * 4 + 1)], scb = r.cluster_meta[static_cast<size_t>(cb * 4 + 2)];
    for(int i = sba; i < sba + sca; i++) {
      o << "    { vec3 a = " << sphere_pos(ca, i) << ";\n";
      for(int jj = sbb; jj < sbb + scb; jj++)
        o << "      { vec3 b = " << sphere_pos(cb, jj) << "; vec3 w = a - b; float z = " << lit(sphere_r(i) + sphere_r(jj)) << " + 2.0 * MARGIN; if(dot(w, w) <= z * z) return false; }\n";
      o << "    }\n";
    }
    o << "  } }\n";
  }
  o << "  return true;\n}\n";
  return o.str();
}

std::string compile_specialized_shader(const std::string& entry_shader, const GpuRobotData& robot, const std::vector<std::string>& extra_defines) {
  if(std::getenv("VKMOTION_NO_CODEGEN")) return "";
  const std::string src_dir = VKMOTION_SHADER_SOURCE_DIR;
  const std::string entry   = src_dir + "/" + entry_shader;
  if(!file_exists(entry) || !file_exists(VKMOTION_GLSLC)) return "";

  const std::string gen = generate_robot_glsl(robot);
  uint64_t h            = fnv1a(gen);
  h                     = fnv1a(read_text(entry), h);
  h                     = fnv1a(read_text(src_dir + "/include/robot_collision_common.glsl"), h);
  h                     = fnv1a(read_text(src_dir + "/include/rrt_common.glsl"), h);
  for(const auto& d : extra_defines) h = fnv1a(d, h);
  char hex[32];
  std::snprintf(hex, sizeof(hex), "%016llx", static_cast<unsigned long long>(h));

  const std::string dir = cache_dir();
  if(!mkdir_p(dir)) return "";
  const std::string out = dir + "/" + entry_shader + "." + hex + ".spv";
  if(file_exists(out)) return out;

  const std::string gen_dir = dir + "/gen_" + hex;
  if(!mkdir_p(gen_dir)) return "";
  {
    std::ofstream f(gen_dir + "/robot_generated.glsl");
    f << gen;
    if(!f) return "";
  }
  const std::string tmp = out + ".tmp" + std::to_string(::getpid());
  std::string cmd       = std::string("\"") + VKMOTION_GLSLC + "\" -O --target-env=vulkan1.1 -DROBOT_GENERATED";
  for(const auto& d : extra_defines) cmd += " -D" + d;
  cmd += " -I\"" + gen_dir + "\" -I\"" + src_dir + "\" \"" + entry + "\" -o \"" + tmp + "\" 2>&1";
  FILE* p = ::popen(cmd.c_str(), "r");
  if(!p) return "";
  std::string log;
  char buf[512];
  while(std::fgets(buf, sizeof(buf), p)) log += buf;
  const int rc = ::pclose(p);
  if(rc != 0 || !file_exists(tmp)) {
    if(std::getenv("VKM_DEBUG")) std::fprintf(stderr, "glslc failed (%d): %s\n", rc, log.c_str());
    ::unlink(tmp.c_str());
    return "";
  }
  if(std::rename(tmp.c_str(), out.c_str()) != 0) {
    ::unlink(tmp.c_str());
    return file_exists(out) ? out : "";
  }
  return out;
}


// ---- CPU(8レーンSIMD)版 ----

std::string generate_robot_cpp(const GpuRobotData& r, size_t dof) {
  std::ostringstream o;
  auto litf = [](float v) { return lit(v) + "f"; };
  const int ns = r.num_steps;
  o << "// 自動生成(vkmotion/gpu_codegen.cpp generate_robot_cpp)。手で編集しないこと。\n";
  o << "#include <cstdint>\n#include <cstring>\n#include \"vkmotion/simd8.hpp\"\nusing namespace vkmotion;\nnamespace {\n";
  o << "inline v8f absf(v8f x) { const v8i m = {0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff}; return reinterpret_cast<v8f>(reinterpret_cast<v8i>(x) & m); }\n";
  o << "inline v8f maxz(v8f x) { const v8f z = {}; return x > z ? x : z; }\n";
  // 箱(pack_gpu_worldの24 float): center+allow, rt行0..2, half, aabb_half
  o << "inline v8f bsq(const float* b, v8f x, v8f y, v8f z) {\n"
       "  const v8f dx = x - b[0], dy = y - b[1], dz = z - b[2];\n"
       "  v8f lx, ly, lz;\n"
       "  if(b[11] != 0.0f) { lx = dx; ly = dy; lz = dz; }  // world軸に揃った箱(回転を省く)\n"
       "  else { lx = b[4] * dx + b[5] * dy + b[6] * dz; ly = b[8] * dx + b[9] * dy + b[10] * dz; lz = b[12] * dx + b[13] * dy + b[14] * dz; }\n"
       "  const v8f ax = absf(lx) - b[16], ay = absf(ly) - b[17], az = absf(lz) - b[18];\n"
       "  const v8f ox = maxz(ax), oy = maxz(ay), oz = maxz(az);\n  return ox * ox + oy * oy + oz * oz;\n}\n";
  o << "inline uint32_t bits(float f) { uint32_t v; std::memcpy(&v, &f, 4); return v; }\n}\n";
  o << "extern \"C\" uint32_t vkm_mask8(const float* q, float margin, uint32_t required, const float* boxes, int nboxes, const float* balls, int nballs) {\n";
  for(size_t j = 0; j < dof; j++) {
    o << "  const v8f q" << j << " = {";
    for(int lane = 0; lane < 8; lane++) o << (lane ? ", " : "") << "q[" << lane * static_cast<int>(dof) + static_cast<int>(j) << "]";
    o << "};\n";
  }
  std::set<int> revolute;
  for(int s = 0; s < ns; s++)
    if(r.step_meta[static_cast<size_t>(s * 4 + 2)] == 1) revolute.insert(r.step_meta[static_cast<size_t>(s * 4 + 3)]);
  for(int j : revolute) o << "  v8f sn" << j << ", cs" << j << ";\n  sincos8(q" << j << ", &sn" << j << ", &cs" << j << ");\n";
  std::set<int> links;
  for(int s = 0; s < ns; s++) links.insert(r.step_meta[static_cast<size_t>(s * 4 + 1)]);
  for(int l : links) o << "  v8f R" << l << "[9], P" << l << "[3];\n";
  for(int g = 0; g < r.num_groups; g++) o << "  v8f Gx" << g << ", Gy" << g << ", Gz" << g << ";\n";
  o << "  v8f SX[" << std::max(r.num_spheres, 1) << "], SY[" << std::max(r.num_spheres, 1) << "], SZ[" << std::max(r.num_spheres, 1) << "];\n";
  for(int c = 0; c < r.num_clusters; c++) o << "  v8f Cx" << c << ", Cy" << c << ", Cz" << c << ";\n";
  auto xf = [&](int link, float x, float y, float z, const std::string& ox, const std::string& oy, const std::string& oz) {
    const std::string R = "R" + std::to_string(link), P = "P" + std::to_string(link);
    return ox + " = " + R + "[0] * " + litf(x) + " + " + R + "[1] * " + litf(y) + " + " + R + "[2] * " + litf(z) + " + " + P + "[0];\n" +
           oy + " = " + R + "[3] * " + litf(x) + " + " + R + "[4] * " + litf(y) + " + " + R + "[5] * " + litf(z) + " + " + P + "[1];\n" +
           oz + " = " + R + "[6] * " + litf(x) + " + " + R + "[7] * " + litf(y) + " + " + R + "[8] * " + litf(z) + " + " + P + "[2];\n";
  };
  for(int s = 0; s < ns; s++) {
    const int* m   = &r.step_meta[static_cast<size_t>(s * 4)];
    const float* d = &r.step_data[static_cast<size_t>(s * 16)];
    const int pa = m[0], ch = m[1], type = m[2], joint = m[3];
    const int grp = bits_to_int(d[3]);
    o << "  {\n";
    if(type == 3) {
      for(int k = 0; k < 9; k++) o << "    R" << ch << "[" << k << "] = v8f{} + " << (k % 4 == 0 ? "1.0f" : "0.0f") << ";\n";
      for(int k = 0; k < 3; k++) o << "    P" << ch << "[" << k << "] = v8f{};\n";
    } else {
      const float ro[9] = {d[4], d[5], d[6], d[8], d[9], d[10], d[12], d[13], d[14]};
      const float to[3] = {d[7], d[11], d[15]};
      o << "    v8f mm[9];\n";
      if(type == 1) {
        const float ax = d[0], ay = d[1], az = d[2];
        const std::string sn = "sn" + std::to_string(joint), cs = "cs" + std::to_string(joint);
        o << "    const v8f tt = 1.0f - " << cs << ";\n    v8f rj[9];\n";
        const std::string rj[9] = {cs + " + tt * " + litf(ax * ax),         "tt * " + litf(ax * ay) + " - " + sn + " * " + litf(az), "tt * " + litf(ax * az) + " + " + sn + " * " + litf(ay),
                                   "tt * " + litf(ay * ax) + " + " + sn + " * " + litf(az), cs + " + tt * " + litf(ay * ay),         "tt * " + litf(ay * az) + " - " + sn + " * " + litf(ax),
                                   "tt * " + litf(az * ax) + " - " + sn + " * " + litf(ay), "tt * " + litf(az * ay) + " + " + sn + " * " + litf(ax), cs + " + tt * " + litf(az * az)};
        for(int k = 0; k < 9; k++) o << "    rj[" << k << "] = " << rj[k] << ";\n";
        for(int i = 0; i < 3; i++)
          for(int j = 0; j < 3; j++)
            o << "    mm[" << i * 3 + j << "] = rj[" << j << "] * " << litf(ro[i * 3]) << " + rj[" << 3 + j << "] * " << litf(ro[i * 3 + 1]) << " + rj[" << 6 + j << "] * " << litf(ro[i * 3 + 2]) << ";\n";
      } else {
        for(int i = 0; i < 9; i++) o << "    mm[" << i << "] = v8f{} + " << litf(ro[i]) << ";\n";
      }
      o << "    v8f t[3] = {v8f{} + " << litf(to[0]) << ", v8f{} + " << litf(to[1]) << ", v8f{} + " << litf(to[2]) << "};\n";
      if(type == 2)
        for(int k = 0; k < 3; k++) o << "    t[" << k << "] += q" << joint << " * " << litf(d[k]) << ";\n";
      for(int i = 0; i < 3; i++)
        for(int j = 0; j < 3; j++)
          o << "    R" << ch << "[" << i * 3 + j << "] = R" << pa << "[" << i * 3 << "] * mm[" << j << "] + R" << pa << "[" << i * 3 + 1 << "] * mm[" << 3 + j << "] + R" << pa << "[" << i * 3 + 2 << "] * mm[" << 6 + j << "];\n";
      for(int i = 0; i < 3; i++)
        o << "    P" << ch << "[" << i << "] = P" << pa << "[" << i << "] + R" << pa << "[" << i * 3 << "] * t[0] + R" << pa << "[" << i * 3 + 1 << "] * t[1] + R" << pa << "[" << i * 3 + 2 << "] * t[2];\n";
    }
    if(grp >= 0) {
      const float* gb = &r.group_bs[static_cast<size_t>(grp * 4)];
      o << xf(ch, gb[0], gb[1], gb[2], "    Gx" + std::to_string(grp), "    Gy" + std::to_string(grp), "    Gz" + std::to_string(grp));
      const int first = r.group_meta[static_cast<size_t>(grp * 4)], count = r.group_meta[static_cast<size_t>(grp * 4 + 1)];
      for(int c = first; c < first + count; c++) {
        const float* cb = &r.cluster_bs[static_cast<size_t>(c * 4)];
        o << xf(ch, cb[0], cb[1], cb[2], "    Cx" + std::to_string(c), "    Cy" + std::to_string(c), "    Cz" + std::to_string(c));
        const int sb2 = r.cluster_meta[static_cast<size_t>(c * 4 + 1)], sc2 = r.cluster_meta[static_cast<size_t>(c * 4 + 2)];
        for(int i = sb2; i < sb2 + sc2; i++) {
          const float* sp = &r.spheres[static_cast<size_t>(i * 4)];
          o << xf(ch, sp[0], sp[1], sp[2], "    SX[" + std::to_string(i) + "]", "    SY[" + std::to_string(i) + "]", "    SZ[" + std::to_string(i) + "]");
        }
      }
    }
    o << "  }\n";
  }

  // 球iの世界座標を宣言する(リンク姿勢はレジスタ/スタックにあるので必要な箇所でその場計算)
  auto sphere_decl = [&](int /*cluster*/, int i, const std::string& n) {
    return "const v8f& " + n + "x = SX[" + std::to_string(i) + "]; const v8f& " + n + "y = SY[" + std::to_string(i) + "]; const v8f& " + n + "z = SZ[" + std::to_string(i) + "];\n";
  };
  auto sphere_r = [&](int i) { return r.spheres[static_cast<size_t>(i * 4 + 3)]; };

  bool cstage2 = false;
  if(std::getenv("VKM_CPU_STAGE")) {
    const int cs = std::atoi(std::getenv("VKM_CPU_STAGE"));
    if(cs == 2) cstage2 = true;
    if(cs == 1) {
      o << "  v8f acc = {};\n  for(int i = 0; i < " << r.num_spheres << "; i++) acc += SX[i] + SY[i] * 2.0f + SZ[i] * 3.0f;\n";
      for(int c = 0; c < r.num_clusters; c++) o << "  acc += Cx" << c << " + Cy" << c << " + Cz" << c << ";\n";
      for(int g = 0; g < r.num_groups; g++) o << "  acc += Gx" << g << " + Gy" << g << " + Gz" << g << ";\n";
      o << "  return movemask(acc < 1e9f);\n}\n";
      return o.str();
    }
  }
  o << "  uint32_t alive = 0xFFu;\n";
  o << "  const auto dead = [required](uint32_t a) { return required ? (a & required) != required : a == 0u; };\n";
  // 直方体: 箱はベースからの最短距離の昇順(B[7])。グループ境界球の「原点からの距離+半径」より遠い箱に当たったらそのグループの残りの箱は飛ばす(VAMPと同じ)
  for(int g = 0; g < r.num_groups; g++) {
    const float rad = r.group_bs[static_cast<size_t>(g * 4 + 3)];
    o << "  {\n    const float rr = " << litf(rad) << " + margin;\n    const v8f ext = sqrt8(Gx" << g << " * Gx" << g << " + Gy" << g << " * Gy" << g << " + Gz" << g << " * Gz" << g << ") + rr;\n";
    o << "    for(int b = 0; b < nboxes; b++) {\n      const float* B = boxes + 24 * b;\n";
    o << "      if(!(movemask(ext >= B[7]) & alive)) break;\n      if((bits(B[3]) >> " << g << ") & 1u) continue;\n";
    o << "      if(movemask((absf(Gx" << g << " - B[0]) <= B[20] + rr) & (absf(Gy" << g << " - B[1]) <= B[21] + rr) & (absf(Gz" << g << " - B[2]) <= B[22] + rr)) & alive) {\n";
    const int first = r.group_meta[static_cast<size_t>(g * 4)], count = r.group_meta[static_cast<size_t>(g * 4 + 1)];
    for(int c = first; c < first + count; c++) {
      const float rc = r.cluster_bs[static_cast<size_t>(c * 4 + 3)];
      o << "        { const float rc = " << litf(rc) << " + margin; if(movemask(bsq(B, Cx" << c << ", Cy" << c << ", Cz" << c << ") <= rc * rc) & alive) {\n";
      const int sb = r.cluster_meta[static_cast<size_t>(c * 4 + 1)], sc = r.cluster_meta[static_cast<size_t>(c * 4 + 2)];
      for(int i = sb; i < sb + sc; i++)
        o << "          { " << sphere_decl(c, i, "s") << "const float q2 = " << litf(sphere_r(i)) << " + margin; alive &= ~movemask(bsq(B, sx, sy, sz) <= q2 * q2); }\n";
      o << "          if(dead(alive)) return 0;\n        } }\n";
    }
    o << "      }\n    }\n  }\n";
  }
  if(cstage2) { o << "  return alive;\n}\n"; return o.str(); }
  // 球障害物
  o << "  for(int b = 0; b < nballs; b++) {\n    const float* L = balls + 8 * b;\n    const uint32_t allow = bits(L[4]);\n";
  for(int c = 0; c < r.num_clusters; c++) {
    const int grpc = r.cluster_meta[static_cast<size_t>(c * 4 + 3)];
    o << "    if(!((allow >> " << grpc << ") & 1u)) {\n";
    const int sb = r.cluster_meta[static_cast<size_t>(c * 4 + 1)], sc = r.cluster_meta[static_cast<size_t>(c * 4 + 2)];
    for(int i = sb; i < sb + sc; i++)
      o << "      { " << sphere_decl(c, i, "s") << "const v8f dx = sx - L[0], dy = sy - L[1], dz = sz - L[2]; const float q2 = " << litf(sphere_r(i)) << " + L[3] + margin; alive &= ~movemask(dx * dx + dy * dy + dz * dz <= q2 * q2); }\n";
    o << "      if(dead(alive)) return 0;\n    }\n";
  }
  o << "  }\n";
  // 自己衝突: クラスタペアは静的
  for(size_t k = 0; k + 1 < r.cluster_pairs.size(); k += 2) {
    const int ca = r.cluster_pairs[k], cb = r.cluster_pairs[k + 1];
    const float rc = r.cluster_bs[static_cast<size_t>(ca * 4 + 3)] + r.cluster_bs[static_cast<size_t>(cb * 4 + 3)];
    o << "  { const v8f ex = Cx" << ca << " - Cx" << cb << ", ey = Cy" << ca << " - Cy" << cb << ", ez = Cz" << ca << " - Cz" << cb << "; const float q = " << litf(rc) << " + 2.0f * margin;\n";
    o << "    if(movemask(ex * ex + ey * ey + ez * ez <= q * q) & alive) {\n";
    const int sba = r.cluster_meta[static_cast<size_t>(ca * 4 + 1)], sca = r.cluster_meta[static_cast<size_t>(ca * 4 + 2)];
    const int sbb = r.cluster_meta[static_cast<size_t>(cb * 4 + 1)], scb = r.cluster_meta[static_cast<size_t>(cb * 4 + 2)];
    for(int i = sba; i < sba + sca; i++)
      for(int jj = sbb; jj < sbb + scb; jj++)
        o << "      { const v8f wx = SX[" << i << "] - SX[" << jj << "], wy = SY[" << i << "] - SY[" << jj << "], wz = SZ[" << i << "] - SZ[" << jj << "]; const float z2 = " << litf(sphere_r(i) + sphere_r(jj))
          << " + 2.0f * margin; alive &= ~movemask(wx * wx + wy * wy + wz * wz <= z2 * z2); }\n";
    o << "      if(dead(alive)) return 0;\n    } }\n";
  }
  o << "  return alive;\n}\n";
  return o.str();
}

CpuMask8Fn load_cpu_kernel(const GpuRobotData& robot, size_t dof) {
  if(std::getenv("VKMOTION_NO_CODEGEN")) return nullptr;
  if(!file_exists(VKMOTION_CXX)) return nullptr;
  const std::string gen = generate_robot_cpp(robot, dof);
  uint64_t h            = fnv1a(gen);
  h                     = fnv1a(read_text(std::string(VKMOTION_SOURCE_DIR) + "/vkmotion/simd8.hpp"), h);
  h                     = fnv1a(VKMOTION_JIT_FLAGS, h);
  char hex[32];
  std::snprintf(hex, sizeof(hex), "%016llx", static_cast<unsigned long long>(h));
  const std::string dir = cache_dir();
  if(!mkdir_p(dir)) return nullptr;
  const std::string so = dir + "/cpu_kernel." + hex + ".so";
  if(!file_exists(so)) {
    const std::string src = dir + "/cpu_kernel." + hex + ".cpp";
    {
      std::ofstream f(src);
      f << gen;
      if(!f) return nullptr;
    }
    const std::string tmp = so + ".tmp" + std::to_string(::getpid());
    const std::string cmd = std::string("\"") + VKMOTION_CXX + "\" " + VKMOTION_JIT_FLAGS + " -I\"" + VKMOTION_SOURCE_DIR + "\" \"" + src + "\" -o \"" + tmp + "\" 2>&1";
    FILE* p               = ::popen(cmd.c_str(), "r");
    if(!p) return nullptr;
    std::string log;
    char buf[512];
    while(std::fgets(buf, sizeof(buf), p)) log += buf;
    const int rc = ::pclose(p);
    if(rc != 0 || !file_exists(tmp)) {
      if(std::getenv("VKM_DEBUG")) std::fprintf(stderr, "cpu kernel compile failed (%d): %s\n", rc, log.c_str());
      ::unlink(tmp.c_str());
      return nullptr;
    }
    if(std::rename(tmp.c_str(), so.c_str()) != 0) {
      ::unlink(tmp.c_str());
      if(!file_exists(so)) return nullptr;
    }
  }
  void* handle = ::dlopen(so.c_str(), RTLD_NOW | RTLD_LOCAL);  // プロセス終了までアンロードしない
  if(!handle) return nullptr;
  return reinterpret_cast<CpuMask8Fn>(::dlsym(handle, "vkm_mask8"));
}

}  // namespace vkmotion
