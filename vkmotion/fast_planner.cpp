#include "fast_planner.hpp"
#include "simd8.hpp"

#include <thread>
#include <atomic>
#include <algorithm>
#include <chrono>
#include <cmath>

namespace vkmotion {

namespace {

// xoshiro128+ (十分速く、スレッドごとに独立に持てる)
struct Rng {
  uint32_t s[4];
  explicit Rng(uint64_t seed) {
    uint64_t z = seed + 0x9E3779B97F4A7C15ULL;
    for(int i = 0; i < 4; i++) {
      z += 0x9E3779B97F4A7C15ULL;
      uint64_t x = z;
      x          = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
      x          = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
      s[i]       = static_cast<uint32_t>((x ^ (x >> 31)) >> 16) | 1u;
    }
  }
  static uint32_t rotl(uint32_t x, int k) { return (x << k) | (x >> (32 - k)); }
  uint32_t next() {
    const uint32_t result = s[0] + s[3];
    const uint32_t t      = s[1] << 9;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rotl(s[3], 11);
    return result;
  }
  float uniform() { return static_cast<float>(next() >> 8) * (1.0f / 16777216.0f); }
  int below(int n) { return static_cast<int>((static_cast<uint64_t>(next()) * static_cast<uint64_t>(n)) >> 32); }
};

// Halton列(低食い違い量列)にCranley-Pattersonのランダムシフトを掛けたサンプラ。一様乱数より空間を均等に覆うので、
// 狭い通路のある問題(cageなど)で探索が早く終わる(VAMPのRRTCも既定でHaltonを使う)。シードごとにシフトが変わるので並列起動しても重複しない。
class HaltonSampler {
 public:
  HaltonSampler(uint64_t seed, size_t dof) : dof_(dof), shift_(dof) {
    Rng r(seed ^ 0x5bd1e995);
    for(size_t j = 0; j < dof; j++) shift_[j] = r.uniform();
    index_ = r.next() % 4096;
  }
  void next(float* out) {
    ++index_;
    for(size_t j = 0; j < dof_; j++) {
      float v = radical_inverse(index_, kPrimes[j % 16]) + shift_[j];
      out[j]  = v - std::floor(v);
    }
  }

 private:
  static float radical_inverse(uint64_t i, uint32_t base) {
    double f = 1.0, r = 0.0;
    while(i > 0) {
      f /= base;
      r += f * static_cast<double>(i % base);
      i /= base;
    }
    return static_cast<float>(r);
  }
  static constexpr uint32_t kPrimes[16] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53};
  size_t dof_;
  std::vector<float> shift_;
  uint64_t index_;
};

// 浮動小数の木。ノードはdofを4の倍数へパディングして連続配置(at()用)。最近傍探索用に、8ノードずつ「次元ごと」に並べ直した鏡も持つ
// (8ノードの距離を1回のSIMD演算列で出す。ノード数が数千になる難しい問題では、線形探索が計画時間の3割を占めていた)。
struct Tree {
  size_t dof;
  size_t stride;
  std::vector<float> q;
  std::vector<int> parent;
  std::vector<v8f> blocks;  // blocks[(i/8)*dof + j][i%8] = ノードiのj次元目。端数レーンは大きな値で埋める

  explicit Tree(size_t dof_) : dof(dof_), stride((dof_ + 3) & ~size_t{3}) {}
  size_t size() const { return parent.size(); }
  const float* at(size_t i) const { return q.data() + i * stride; }
  int add(const float* v, int par) {
    const size_t i = parent.size();
    q.resize(q.size() + stride, 0.0f);
    std::copy(v, v + dof, q.end() - static_cast<std::ptrdiff_t>(stride));
    parent.push_back(par);
    if(i % 8 == 0) {
      const v8f far = v8f{} + 1e18f;
      for(size_t j = 0; j < dof; j++) blocks.push_back(far);
    }
    for(size_t j = 0; j < dof; j++) blocks[(i / 8) * dof + j][i % 8] = v[j];
    return static_cast<int>(i);
  }
  int nearest(const float* v, float* out_d2) const {
    const size_t n = size();
    v8f bd         = v8f{} + 1e30f;
    v8i bi         = v8i{};
    for(size_t b = 0; b * 8 < n; b++) {
      const v8f* blk = blocks.data() + b * dof;
      v8f d2         = v8f{};
      for(size_t j = 0; j < dof; j++) {
        const v8f d = blk[j] - v[j];
        d2 += d * d;
      }
      const v8i better = d2 < bd;
      bd               = better ? d2 : bd;
      bi               = better ? v8i{} + static_cast<int>(b) : bi;
    }
    float best = 1e30f;
    int idx    = 0;
    for(int lane = 0; lane < 8; lane++) {
      if(bd[lane] < best) {
        best = bd[lane];
        idx  = bi[lane] * 8 + lane;
      }
    }
    *out_d2 = best;
    return idx;
  }
};

// from から to へ range までだけ進めた点を out に書く。到達したら true。
bool steer(const float* from, const float* to, size_t dof, float range, float* out) {
  float d2 = 0.0f;
  for(size_t j = 0; j < dof; j++) {
    float d = to[j] - from[j];
    d2 += d * d;
  }
  float d = std::sqrt(d2);
  if(d <= range) {
    std::copy(to, to + dof, out);
    return true;
  }
  float s = range / d;
  for(size_t j = 0; j < dof; j++) out[j] = from[j] + (to[j] - from[j]) * s;
  return false;
}

std::vector<float> trace(const Tree& t, int idx) {
  std::vector<float> out;
  while(idx >= 0) {
    out.insert(out.end(), t.at(static_cast<size_t>(idx)), t.at(static_cast<size_t>(idx)) + t.dof);
    idx = t.parent[static_cast<size_t>(idx)];
  }
  return out;  // 葉->根の順にdofずつ連結
}

}  // namespace

FastRrtResult fast_rrt_connect(const FastCollision& fc, const std::vector<float>& lower, const std::vector<float>& upper, const std::vector<float>& start,
                               const std::vector<std::vector<float>>& goals, const FastRrtSettings& st) {
  FastRrtResult res;
  const size_t dof = fc.dof();
  auto t0 = std::chrono::steady_clock::now();

  // 直線で結べるなら木を作るまでもない
  for(const auto& g : goals) {
    if(fc.motion_free(start.data(), g.data(), st.margin, false, st.motion_step)) {
      res.success = true;
      res.path    = {start, g};
      return res;
    }
  }

  Tree ta(dof), tb(dof);  // ta: start側, tb: goal側
  ta.add(start.data(), -1);
  for(const auto& g : goals) tb.add(g.data(), -1);
  bool a_is_start = true;
  // 端点が障害物すれすれだと、端点の近傍が余裕(margin)に引っかかって片方の木が育たない(MotionBenchMakerのゴールの多くがそう)。
  // 木の伸びが止まっていたら余裕を 1/4 -> 0 と縮める(それまでに木に入れたノードは余裕が大きいときのものなので有効なまま)。
  float margin_now = st.margin;
  res.margin       = margin_now;

  HaltonSampler halton(st.seed, dof);
  std::vector<float> sample(dof), nq(dof), cur(dof);
  for(int iter = 0; iter < st.max_iterations; iter++) {
    res.iterations = iter;
    if((iter & 63) == 0) {
      if(st.cancel && st.cancel->load(std::memory_order_relaxed)) return res;
      if(st.should_stop && st.should_stop()) return res;
      if(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > st.timeout_s) return res;
    }
    if(margin_now > 0.0f && st.adapt_margin && iter >= 256 && (iter & 127) == 0 && std::min(ta.size(), tb.size()) < 3 + static_cast<size_t>(iter) / 128) {
      // 1/4 までは早めに、0(障害物に接する経路を許す)は長く止まったときだけ。0にすると別の検証器で見たときに食い込む経路が増える
      if(margin_now > 0.3f * st.margin) margin_now = 0.25f * st.margin;
      else if(iter >= 768) margin_now = 0.0f;
      res.margin = margin_now;
    }
    Tree& A = a_is_start ? ta : tb;
    Tree& B = a_is_start ? tb : ta;

    halton.next(sample.data());
    for(size_t j = 0; j < dof; j++) sample[j] = lower[j] + sample[j] * (upper[j] - lower[j]);
    float d2;
    int ni = A.nearest(sample.data(), &d2);
    steer(A.at(static_cast<size_t>(ni)), sample.data(), dof, st.range, nq.data());
    if(fc.motion_free(A.at(static_cast<size_t>(ni)), nq.data(), margin_now, true, st.motion_step)) {
      int a_new = A.add(nq.data(), ni);

      // Bをnqへ向けて伸ばせるだけ伸ばす
      int bi = B.nearest(nq.data(), &d2);
      int bn = bi;
      bool reached = false;
      for(int guard = 0; guard < 10000; guard++) {
        bool at_target = steer(B.at(static_cast<size_t>(bn)), nq.data(), dof, st.range, cur.data());
        if(!fc.motion_free(B.at(static_cast<size_t>(bn)), cur.data(), margin_now, true, st.motion_step)) break;
        bn = B.add(cur.data(), bn);
        if(at_target) {
          reached = true;
          break;
        }
      }
      if(reached) {
        std::vector<float> pa = trace(A, a_new);  // a_new -> A根
        std::vector<float> pb = trace(B, bn);     // bn -> B根
        // pa/pbはdofずつの連結(葉->根)。start側: 根->葉、goal側: 葉->根 になるよう並べる
        auto split = [&](const std::vector<float>& flat, bool reverse) {
          ConfigPath out;
          for(size_t k = 0; k + dof <= flat.size(); k += dof) out.emplace_back(flat.begin() + static_cast<std::ptrdiff_t>(k), flat.begin() + static_cast<std::ptrdiff_t>(k + dof));
          if(reverse) std::reverse(out.begin(), out.end());
          return out;
        };
        ConfigPath path;
        if(a_is_start) {
          path = split(pa, true);                    // start -> a_new
          ConfigPath tail = split(pb, false);        // bn -> goal
          path.insert(path.end(), tail.begin() + 1, tail.end());  // bnはa_newと同一点
        } else {
          path = split(pb, true);                    // start -> bn  (Bがstart側)
          ConfigPath tail = split(pa, false);        // a_new -> goal
          path.insert(path.end(), tail.begin() + 1, tail.end());
        }
        res.success = true;
        res.path    = std::move(path);
        return res;
      }
    }
    a_is_start = !a_is_start;
  }
  return res;
}

// ---- スレッドプール ----

struct RrtThreadPool::Impl {
  std::vector<std::thread> threads;
  std::atomic<uint32_t> generation{0};
  std::atomic<int> remaining{0};
  std::atomic<bool> stop{false};
  const std::function<void(int)>* fn = nullptr;
};

RrtThreadPool::RrtThreadPool(int participants) : participants_(std::max(1, participants)), impl_(new Impl()) {
  for(int t = 1; t < participants_; t++) {
    impl_->threads.emplace_back([this, t] {
      uint32_t seen = 0;
      while(true) {
        // 短くスピンして新しい仕事を拾い、無ければfutexで眠る
        for(int spin = 0; spin < 4000 && impl_->generation.load(std::memory_order_acquire) == seen && !impl_->stop.load(std::memory_order_relaxed); spin++) {
#if defined(__x86_64__)
          __builtin_ia32_pause();
#endif
        }
        impl_->generation.wait(seen, std::memory_order_acquire);
        if(impl_->stop.load(std::memory_order_acquire)) return;
        seen = impl_->generation.load(std::memory_order_acquire);
        (*impl_->fn)(t);
        impl_->remaining.fetch_sub(1, std::memory_order_acq_rel);
        impl_->remaining.notify_all();
      }
    });
  }
}

RrtThreadPool::~RrtThreadPool() {
  impl_->stop.store(true, std::memory_order_release);
  impl_->generation.fetch_add(1, std::memory_order_acq_rel);
  impl_->generation.notify_all();
  for(auto& th : impl_->threads) th.join();
  delete impl_;
}

void RrtThreadPool::run(const std::function<void(int)>& fn) {
  if(participants_ == 1) {
    fn(0);
    return;
  }
  impl_->fn = &fn;
  impl_->remaining.store(participants_ - 1, std::memory_order_release);
  impl_->generation.fetch_add(1, std::memory_order_acq_rel);
  impl_->generation.notify_all();
  fn(0);
  for(int r = impl_->remaining.load(std::memory_order_acquire); r != 0; r = impl_->remaining.load(std::memory_order_acquire)) {
    for(int spin = 0; spin < 2000 && impl_->remaining.load(std::memory_order_acquire) != 0; spin++) {
#if defined(__x86_64__)
      __builtin_ia32_pause();
#endif
    }
    r = impl_->remaining.load(std::memory_order_acquire);
    if(r != 0) impl_->remaining.wait(r, std::memory_order_acquire);
  }
}

namespace {

// 複数スレッドが同時にノードを足せる木。追加は「番号を予約 -> 書く -> 番号順に公開」で、公開済みの範囲[0, published)だけが他スレッドから読める。
struct SharedTree {
  size_t dof = 0, stride = 0;
  uint32_t cap = 0;
  std::vector<float> q;
  std::vector<int> parent;
  std::vector<v8f> blocks;
  std::atomic<uint32_t> reserved{0}, published{0};

  void reset(size_t dof_, uint32_t cap_) {
    dof    = dof_;
    stride = (dof_ + 3) & ~size_t{3};
    cap    = cap_;
    q.assign(static_cast<size_t>(cap) * stride, 0.0f);
    parent.assign(cap, -1);
    blocks.assign(static_cast<size_t>((cap + 7) / 8) * dof, v8f{} + 1e18f);
    reserved.store(0);
    published.store(0);
  }
  const float* at(uint32_t i) const { return q.data() + static_cast<size_t>(i) * stride; }
  // 追加できなければ -1(容量超過)
  int append(const float* v, int par) {
    const uint32_t i = reserved.fetch_add(1, std::memory_order_acq_rel);
    if(i >= cap) return -1;
    std::copy(v, v + dof, q.begin() + static_cast<std::ptrdiff_t>(i * stride));
    parent[i] = par;
    // ブロック内の別レーンを別スレッドが同時に書くので、ベクトル全体の読み書きになりうる v[lane]= ではなく、スカラーのstoreで書く
    float* bf = reinterpret_cast<float*>(blocks.data());
    for(size_t j = 0; j < dof; j++) bf[((i / 8) * dof + j) * 8 + (i % 8)] = v[j];
    while(published.load(std::memory_order_acquire) != i) {
#if defined(__x86_64__)
      __builtin_ia32_pause();
#endif
    }
    published.store(i + 1, std::memory_order_release);
    return static_cast<int>(i);
  }
  // 公開済みのn個の中の最近傍
  int nearest(const float* v, uint32_t n) const {
    v8f bd = v8f{} + 1e30f;
    v8i bi = v8i{};
    for(uint32_t b = 0; static_cast<uint32_t>(b) * 8 < n; b++) {
      const v8f* blk = blocks.data() + static_cast<size_t>(b) * dof;
      v8f d2         = v8f{};
      for(size_t j = 0; j < dof; j++) {
        const v8f d = blk[j] - v[j];
        d2 += d * d;
      }
      if((b + 1) * 8 > n) {  // 最後のブロックの未公開レーンは使わない
        v8i idx = {0, 1, 2, 3, 4, 5, 6, 7};
        idx += static_cast<int>(b * 8);
        d2 = (idx < static_cast<int>(n)) ? d2 : v8f{} + 1e30f;
      }
      const v8i better = d2 < bd;
      bd               = better ? d2 : bd;
      bi               = better ? v8i{} + static_cast<int>(b) : bi;
    }
    float best = 1e30f;
    int idx    = 0;
    for(int lane = 0; lane < 8; lane++) {
      if(bd[lane] < best) {
        best = bd[lane];
        idx  = bi[lane] * 8 + lane;
      }
    }
    return idx;
  }
};

}  // namespace

FastRrtResult fast_rrt_connect_parallel(const FastCollision& fc, const std::vector<float>& lower, const std::vector<float>& upper, const std::vector<float>& start,
                                        const std::vector<std::vector<float>>& goals, const FastRrtSettings& st, RrtThreadPool& pool) {
  FastRrtResult res;
  const size_t dof = fc.dof();
  const auto t0    = std::chrono::steady_clock::now();
  for(const auto& g : goals) {
    if(fc.motion_free(start.data(), g.data(), st.margin, false, st.motion_step)) {
      res.success = true;
      res.path    = {start, g};
      return res;
    }
  }
  constexpr uint32_t kCap = 8192;
  static thread_local SharedTree trees_storage[2];  // 呼び出しスレッドごとに使い回す(毎回の確保を避ける)
  SharedTree& ts = trees_storage[0];                // start側
  SharedTree& tg = trees_storage[1];                // goal側
  ts.reset(dof, kCap);
  tg.reset(dof, kCap);
  ts.append(start.data(), -1);
  for(const auto& g : goals) tg.append(g.data(), -1);

  std::atomic<bool> stop{false};
  std::atomic<int> solved{0};
  std::atomic<float> margin_now{st.margin};
  int win_a = -1, win_b = -1;  // 勝ったスレッドが書く: 木Aのノード、木Bのノード(同じ配置)
  bool win_a_is_start = true;
  std::atomic<int> thread0_iters{0};

  pool.run([&](int tid) {
    HaltonSampler halton(st.seed + static_cast<uint64_t>(tid) * 7919ull, dof);
    std::vector<float> sample(dof), nq(dof), cur(dof);
    for(int iter = 0; iter < st.max_iterations; iter++) {
      if((iter & 15) == 0) {
        if(stop.load(std::memory_order_relaxed)) return;
        if(tid == 0) {
          thread0_iters.store(iter, std::memory_order_relaxed);
          if(st.cancel && st.cancel->load(std::memory_order_relaxed)) { stop.store(true); return; }
          if(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > st.timeout_s) { stop.store(true); return; }
          if(st.should_stop && st.should_stop()) { stop.store(true); return; }
          // 端点が障害物すれすれで片方の木が育たないとき、余裕を縮める(fast_rrt_connectと同じ規則。反復数は全スレッド分に換算)
          const float m = margin_now.load(std::memory_order_relaxed);
          const size_t total_iters = static_cast<size_t>(iter) * static_cast<size_t>(pool.size());
          if(m > 0.0f && st.adapt_margin && total_iters >= 256 && (iter & 127) == 0 &&
             std::min(ts.published.load(), tg.published.load()) < 3 + total_iters / 128) {
            if(m > 0.3f * st.margin) margin_now.store(0.25f * st.margin);
            else if(total_iters >= 768) margin_now.store(0.0f);
          }
        }
      }
      const bool a_is_start = ((iter + tid) & 1) == 0;
      SharedTree& A         = a_is_start ? ts : tg;
      SharedTree& B         = a_is_start ? tg : ts;
      const float mgn       = margin_now.load(std::memory_order_relaxed);

      halton.next(sample.data());
      for(size_t j = 0; j < dof; j++) sample[j] = lower[j] + sample[j] * (upper[j] - lower[j]);
      const uint32_t na = A.published.load(std::memory_order_acquire);
      const int ni      = A.nearest(sample.data(), na);
      steer(A.at(static_cast<uint32_t>(ni)), sample.data(), dof, st.range, nq.data());
      if(!fc.motion_free(A.at(static_cast<uint32_t>(ni)), nq.data(), mgn, true, st.motion_step)) continue;
      const int a_new = A.append(nq.data(), ni);
      if(a_new < 0) {  // 容量超過 -> この試行は終わり(呼び出し側が別シードで再開する)
        stop.store(true);
        return;
      }
      const uint32_t nb = B.published.load(std::memory_order_acquire);
      int bn            = B.nearest(nq.data(), nb);
      bool reached      = false;
      for(int guard = 0; guard < 10000; guard++) {
        const bool at_target = steer(B.at(static_cast<uint32_t>(bn)), nq.data(), dof, st.range, cur.data());
        if(!fc.motion_free(B.at(static_cast<uint32_t>(bn)), cur.data(), mgn, true, st.motion_step)) break;
        const int nn = B.append(cur.data(), bn);
        if(nn < 0) {
          stop.store(true);
          return;
        }
        bn = nn;
        if(at_target) {
          reached = true;
          break;
        }
      }
      if(reached) {
        int expected = 0;
        if(solved.compare_exchange_strong(expected, 1)) {
          win_a          = a_new;
          win_b          = bn;
          win_a_is_start = a_is_start;
          stop.store(true);
        }
        return;
      }
    }
  });

  res.iterations = thread0_iters.load();
  res.margin     = margin_now.load();
  if(solved.load() == 0) return res;
  // 経路: start側の根 -> ... -> 接続点 -> ... -> goal側の根
  const SharedTree& S = ts;
  const SharedTree& G = tg;
  const int s_node    = win_a_is_start ? win_a : win_b;
  const int g_node    = win_a_is_start ? win_b : win_a;
  ConfigPath from_start, to_goal;
  for(int i = s_node; i >= 0; i = S.parent[static_cast<size_t>(i)]) from_start.emplace_back(S.at(static_cast<uint32_t>(i)), S.at(static_cast<uint32_t>(i)) + dof);
  std::reverse(from_start.begin(), from_start.end());
  for(int i = g_node; i >= 0; i = G.parent[static_cast<size_t>(i)]) to_goal.emplace_back(G.at(static_cast<uint32_t>(i)), G.at(static_cast<uint32_t>(i)) + dof);
  res.path = std::move(from_start);
  res.path.insert(res.path.end(), to_goal.begin() + 1, to_goal.end());  // 接続点は重複するので1つ飛ばす
  res.success = true;
  return res;
}

void fast_reduce(const FastCollision& fc, ConfigPath& path, float margin, float step) {
  if(path.size() < 3) return;
  ConfigPath out;
  out.push_back(path.front());
  size_t i = 0;
  while(i + 1 < path.size()) {
    size_t j = path.size() - 1;
    while(j > i + 1 && !fc.motion_free(path[i].data(), path[j].data(), margin, false, step)) j--;
    out.push_back(path[j]);
    i = j;
  }
  path = std::move(out);
}

void fast_shortcut(const FastCollision& fc, ConfigPath& path, int iterations, uint64_t seed, float margin, float step, double time_budget_s) {
  Rng rng(seed ^ 0xABCDEF);
  const auto t_start = std::chrono::steady_clock::now();
  const size_t dof = fc.dof();
  for(int it = 0; it < iterations && path.size() >= 3; it++) {
    if(time_budget_s > 0.0 && std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count() > time_budget_s) break;
    int n = static_cast<int>(path.size());
    // 区間の途中の点(辺上のランダム点)同士を結ぶ: 頂点同士だけより短くなりやすい
    int e1 = rng.below(n - 1), e2 = rng.below(n - 1);
    if(e1 == e2) continue;
    if(e1 > e2) std::swap(e1, e2);
    float t1 = rng.uniform(), t2 = rng.uniform();
    std::vector<float> a(dof), b(dof);
    for(size_t j = 0; j < dof; j++) {
      a[j] = path[static_cast<size_t>(e1)][j] + t1 * (path[static_cast<size_t>(e1) + 1][j] - path[static_cast<size_t>(e1)][j]);
      b[j] = path[static_cast<size_t>(e2)][j] + t2 * (path[static_cast<size_t>(e2) + 1][j] - path[static_cast<size_t>(e2)][j]);
    }
    if(!fc.motion_free(a.data(), b.data(), margin, true, step) || !fc.is_free_margin(a.data(), margin)) continue;
    // a, b の間の頂点 e1+1..e2 を削って a,b を挿入
    ConfigPath out(path.begin(), path.begin() + e1 + 1);
    out.push_back(a);
    out.push_back(b);
    out.insert(out.end(), path.begin() + e2 + 1, path.end());
    path = std::move(out);
  }
}

}  // namespace vkmotion
