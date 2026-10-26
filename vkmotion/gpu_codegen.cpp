#include "gpu_codegen.hpp"

#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

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
  o << "  for(uint b = 0; b < pc.num_boxes; b++) {\n";
  o << "    const uint bo = b * 6;\n    const uint allow = floatBitsToUint(s_boxes[bo].w);\n    const vec3 center = s_boxes[bo].xyz;\n    const vec3 aabb_half = s_boxes[bo + 5].xyz;\n    uint cand = 0u;\n";
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
    o << "  if(pc.num_balls != 0u) {\n";
    for(int c = 0; c < r.num_clusters; c++) {
      const int sb = r.cluster_meta[static_cast<size_t>(c * 4 + 1)], sc = r.cluster_meta[static_cast<size_t>(c * 4 + 2)];
      for(int i = sb; i < sb + sc; i++) o << "    sph[" << i << "] = vec4(" << sphere_pos(c, i) << ", " << lit(sphere_r(i)) << ");\n";
    }
    o << "    if(balls_hit()) return false;\n  }\n";
  }
  // 自己衝突: リンクペアは静的
  for(int g = 0; g < r.num_groups; g++) {
    const int mask = r.pairs[static_cast<size_t>(g)];
    for(int h = g + 1; h < r.num_groups; h++) {
      if(!((mask >> h) & 1)) continue;
      const float rr = r.group_bs[static_cast<size_t>(g * 4 + 3)] + r.group_bs[static_cast<size_t>(h * 4 + 3)];
      o << "  { vec3 d = G" << g << " - G" << h << "; float rr = " << lit(rr) << " + 2.0 * MARGIN;\n    if(dot(d, d) <= rr * rr) {\n";
      const int fa = r.group_meta[static_cast<size_t>(g * 4)], na = r.group_meta[static_cast<size_t>(g * 4 + 1)];
      const int fb = r.group_meta[static_cast<size_t>(h * 4)], nb = r.group_meta[static_cast<size_t>(h * 4 + 1)];
      for(int ca = fa; ca < fa + na; ca++)
        for(int cb = fb; cb < fb + nb; cb++) {
          const float rc = r.cluster_bs[static_cast<size_t>(ca * 4 + 3)] + r.cluster_bs[static_cast<size_t>(cb * 4 + 3)];
          o << "      { vec3 e = C" << ca << " - C" << cb << "; float q = " << lit(rc) << " + 2.0 * MARGIN; if(dot(e, e) <= q * q) {\n";
          const int sba = r.cluster_meta[static_cast<size_t>(ca * 4 + 1)], sca = r.cluster_meta[static_cast<size_t>(ca * 4 + 2)];
          const int sbb = r.cluster_meta[static_cast<size_t>(cb * 4 + 1)], scb = r.cluster_meta[static_cast<size_t>(cb * 4 + 2)];
          for(int i = sba; i < sba + sca; i++) {
            o << "        { vec3 a = " << sphere_pos(ca, i) << ";\n";
            for(int jj = sbb; jj < sbb + scb; jj++)
              o << "          { vec3 b = " << sphere_pos(cb, jj) << "; vec3 w = a - b; float z = " << lit(sphere_r(i) + sphere_r(jj)) << " + 2.0 * MARGIN; if(dot(w, w) <= z * z) return false; }\n";
            o << "        }\n";
          }
          o << "      } }\n";
        }
      o << "    } }\n";
    }
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

}  // namespace vkmotion
