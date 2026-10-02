#include "gjk.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace vkmotion {

namespace {

using Vec3 = cutil::Vec3f;

Vec3 closest_on_segment(const Vec3& a, const Vec3& b) {
  Vec3 ab    = b - a;
  float denom = static_cast<float>(ab.dot(ab));
  if(denom < 1e-12f) return a;
  float t = static_cast<float>((-a).dot(ab)) / denom;
  t       = std::clamp(t, 0.0f, 1.0f);
  return a + ab * t;
}

// Closest point on triangle abc to the origin (Ericson, Real-Time Collision Detection 5.1.5).
Vec3 closest_on_triangle(const Vec3& a, const Vec3& b, const Vec3& c) {
  Vec3 ab = b - a, ac = c - a, ap = Vec3(0, 0, 0) - a;
  float d1 = static_cast<float>(ab.dot(ap)), d2 = static_cast<float>(ac.dot(ap));
  if(d1 <= 0.0f && d2 <= 0.0f) return a;

  Vec3 bp  = Vec3(0, 0, 0) - b;
  float d3 = static_cast<float>(ab.dot(bp)), d4 = static_cast<float>(ac.dot(bp));
  if(d3 >= 0.0f && d4 <= d3) return b;

  float vc = d1 * d4 - d3 * d2;
  if(vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) return a + ab * (d1 / (d1 - d3));

  Vec3 cp  = Vec3(0, 0, 0) - c;
  float d5 = static_cast<float>(ab.dot(cp)), d6 = static_cast<float>(ac.dot(cp));
  if(d6 >= 0.0f && d5 <= d6) return c;

  float vb = d5 * d2 - d1 * d6;
  if(vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) return a + ac * (d2 / (d2 - d6));

  float va = d3 * d6 - d5 * d4;
  if(va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));

  float denom = 1.0f / (va + vb + vc);
  float v = vb * denom, w = vc * denom;
  return a + ab * v + ac * w;
}

bool same_side(const Vec3& v1, const Vec3& v2, const Vec3& v3, const Vec3& v4, const Vec3& p) {
  Vec3 normal     = (v2 - v1).cross(v3 - v1);
  float dist_v4   = static_cast<float>(normal.dot(v4 - v1));
  float dist_p    = static_cast<float>(normal.dot(p - v1));
  return (dist_v4 >= 0.0f) == (dist_p >= 0.0f);
}

bool origin_in_tetrahedron(const Vec3& a, const Vec3& b, const Vec3& c, const Vec3& d) {
  Vec3 o = Vec3(0, 0, 0);
  return same_side(a, b, c, d, o) && same_side(b, c, d, a, o) && same_side(c, d, a, b, o) && same_side(d, a, b, c, o);
}

struct Simplex {
  Vec3 pts[4];
  int n = 0;
};

// Closest point to origin on conv(simplex); if n==4 not enclosed, reduces simplex to the winning face's 3 points so the caller can append the next support point and stay at the GJK size-4 invariant.
Vec3 closest_and_reduce(Simplex& s, bool& inside) {
  inside = false;
  if(s.n == 1) return s.pts[0];
  if(s.n == 2) return closest_on_segment(s.pts[0], s.pts[1]);
  if(s.n == 3) return closest_on_triangle(s.pts[0], s.pts[1], s.pts[2]);

  if(origin_in_tetrahedron(s.pts[0], s.pts[1], s.pts[2], s.pts[3])) {
    inside = true;
    return Vec3(0, 0, 0);
  }
  static const int faces[4][3] = {{0, 1, 2}, {0, 1, 3}, {0, 2, 3}, {1, 2, 3}};
  Vec3 best;
  float best_d2 = std::numeric_limits<float>::infinity();
  int best_f    = 0;
  for(int f = 0; f < 4; f++) {
    Vec3 c  = closest_on_triangle(s.pts[faces[f][0]], s.pts[faces[f][1]], s.pts[faces[f][2]]);
    float d2 = static_cast<float>(c.dot(c));
    if(d2 < best_d2) {
      best_d2 = d2;
      best    = c;
      best_f  = f;
    }
  }
  Vec3 p0 = s.pts[faces[best_f][0]], p1 = s.pts[faces[best_f][1]], p2 = s.pts[faces[best_f][2]];
  s.pts[0] = p0;
  s.pts[1] = p1;
  s.pts[2] = p2;
  s.n       = 3;
  return best;
}

}  // namespace

float gjk_point_convex_distance(const cutil::Vec3f& point, const std::vector<cutil::Vec3f>& hull_vertices, bool& inside) {
  inside = false;
  if(hull_vertices.empty()) return std::numeric_limits<float>::infinity();

  // Inside fallback (no EPA): nearest-vertex distance, not a constant, so finite-difference gradients aren't exactly zero and repair can still escape.
  float nearest_vertex_dist = std::numeric_limits<float>::infinity();
  for(const auto& v : hull_vertices) nearest_vertex_dist = std::min(nearest_vertex_dist, static_cast<float>((v - point).length()));

  auto support = [&](const Vec3& dir) -> Vec3 {
    float best = -std::numeric_limits<float>::infinity();
    Vec3 best_q;
    for(const auto& v : hull_vertices) {
      Vec3 q  = v - point;
      float d = static_cast<float>(q.dot(dir));
      if(d > best) {
        best   = d;
        best_q = q;
      }
    }
    return best_q;
  };

  Simplex s;
  s.pts[0] = support(hull_vertices[0] - point);
  s.n       = 1;

  constexpr int kMaxIter = 32;
  for(int iter = 0; iter < kMaxIter; iter++) {
    Vec3 closest = closest_and_reduce(s, inside);
    if(inside) return nearest_vertex_dist;
    float closest_len = static_cast<float>(closest.length());
    if(closest_len < 1e-7f) {
      inside = true;
      return nearest_vertex_dist;
    }
    Vec3 dir     = closest * -1.0f;
    Vec3 new_pt  = support(dir);
    float new_v  = static_cast<float>(new_pt.dot(dir));
    float cur_v  = static_cast<float>(closest.dot(dir));
    if(new_v - cur_v < 1e-6f) return closest_len;
    s.pts[s.n] = new_pt;
    s.n++;
  }
  bool unused;
  return static_cast<float>(closest_and_reduce(s, unused).length());
}

}  // namespace vkmotion
