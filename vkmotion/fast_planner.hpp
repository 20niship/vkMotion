#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <thread>
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

// 常駐スレッドプール。run(fn) は fn(0..size()-1) を並列に実行して全部終わるまで待つ(fn(0)は呼び出しスレッド)。
// 起動はfutex待ち+短いスピンなので、毎回スレッドを作るより数十µs速い。
class RrtThreadPool {
 public:
  explicit RrtThreadPool(int participants);  // 呼び出しスレッドを含む総数(1なら追加スレッド無し)
  ~RrtThreadPool();
  RrtThreadPool(const RrtThreadPool&)            = delete;
  RrtThreadPool& operator=(const RrtThreadPool&) = delete;
  int size() const { return participants_; }
  void run(const std::function<void(int)>& fn);

 private:
  struct Impl;
  int participants_;
  Impl* impl_;
};

// 複数スレッドが「1組の木」を共有して同時に伸ばす双方向RRT-Connect(GPU探索と同じ考え方をCPUスレッドで)。
// 各スレッドが別のHalton列でサンプルし、追加したノードは公開された時点で他スレッドから見える(ラウンドを待たない)。
// 1スレッドの fast_rrt_connect の数倍のノードを同じ時間で作れるので、難しい問題の時間を縮める。結果の経路とmarginの意味は fast_rrt_connect と同じ。
// should_stop は呼び出しスレッド(fn(0))からだけ呼ばれる。
FastRrtResult fast_rrt_connect_parallel(const FastCollision& fc, const std::vector<float>& lower, const std::vector<float>& upper, const std::vector<float>& start,
                                        const std::vector<std::vector<float>>& goals, const FastRrtSettings& settings, RrtThreadPool& pool);

// ランダムな2点を直線で結び直すshortcut。検査はmotion_free(二分法)なので1回あたり数µs〜数十µs。
void fast_shortcut(const FastCollision& fc, ConfigPath& path, int iterations, uint64_t seed, float margin = 0.0f, float step = 0.0f, double time_budget_s = 0.0);

// 先頭から貪欲に「結べる一番遠い点」へ飛ばす(shortcutの前処理。決定的でwaypoint数が一気に減る)。
void fast_reduce(const FastCollision& fc, ConfigPath& path, float margin = 0.0f, float step = 0.0f);

}  // namespace vkmotion
