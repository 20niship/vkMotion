#pragma once

#include <cutil/vec.hpp>
#include <vector>

namespace vkmotion {

// Distance from point to conv(hull_vertices) surface when outside; when inside, sets inside=true and returns the (positive) nearest-vertex distance as a perturbation-sensitive depth proxy (no EPA, not an exact penetration depth).
float gjk_point_convex_distance(const cutil::Vec3f& point, const std::vector<cutil::Vec3f>& hull_vertices, bool& inside);

}  // namespace vkmotion
