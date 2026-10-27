#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <vector>

#include "fast_collision.hpp"

namespace vkmotion {

using ConfigPath = std::vector<std::vector<float>>;

struct FastRrtSettings {
  float range         = 1.0f;  // 1回のextendで伸ばす最大のL2距離[rad] (VAMPのpanda既定と同じ)
  int max_iterations  = 200000;
  double timeout_s    = 5.0;
  uint64_t seed       = 1;
  float margin        = 0.0f;  // 内部点・新規ノードに持たせる安全余裕[m](start/goalは対象外)
  bool adapt_margin   = true;  // 片方の木が伸びないとき、margin を 1/4 -> 0 と縮めて続ける(結果のmarginに実際に使った値が入る)
  float motion_step   = 0.0f;  // 辺の検査間隔[m](0: FastCollisionの既定)。探索中は粗くして、最終経路を細かく検証し直す使い方を想定
  std::function<bool()> should_stop;  // 64反復ごとに呼ばれ、trueなら打ち切る(GPU探索が先に解けたか、など)
  const std::atomic<bool>* cancel = nullptr;  // 他スレッドが先に解いたら打ち切る
};

struct FastRrtResult {
  bool success = false;
  ConfigPath path;
  int iterations = 0;
  float margin   = 0.0f;  // 経路が満たしている余裕(adapt_marginで縮めた場合は縮めた値)
};

// 双方向RRT-Connect(VAMPのRRTCと同じ構成)。goalsのどれかに到達すればよい(複数IK解をゴール木の根に入れる用途)。
// start/goalsは衝突なしであること。経路は[start, ..., goalのどれか]。
FastRrtResult fast_rrt_connect(const FastCollision& fc, const std::vector<float>& lower, const std::vector<float>& upper, const std::vector<float>& start,
                               const std::vector<std::vector<float>>& goals, const FastRrtSettings& settings = {});

// ランダムな2点を直線で結び直すshortcut。検査はmotion_free(二分法)なので1回あたり数µs〜数十µs。
void fast_shortcut(const FastCollision& fc, ConfigPath& path, int iterations, uint64_t seed, float margin = 0.0f, float step = 0.0f);

// 先頭から貪欲に「結べる一番遠い点」へ飛ばす(shortcutの前処理。決定的でwaypoint数が一気に減る)。
void fast_reduce(const FastCollision& fc, ConfigPath& path, float margin = 0.0f, float step = 0.0f);

}  // namespace vkmotion
