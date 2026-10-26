#pragma once

#include <cstdint>
#ifdef __AVX__
#include <immintrin.h>
#endif

// 8コンフィグを同時に扱うSIMDの小物(GCC/Clang vector extension)。fast_kinematics / fast_collision と、
// ロボット専用に生成したカーネル(gpu_codegen.cpp の generate_robot_cpp)が共有する。
namespace vkmotion {

typedef float v8f __attribute__((vector_size(32), aligned(32)));
typedef int v8i __attribute__((vector_size(32), aligned(32)));
constexpr int kSimdWidth = 8;

// 8レーン同時のsin/cos (cephesのsinf/cosfと同じCody-Waite縮約+最小二乗多項式。|x|<8192で誤差~1e-7)
inline void sincos8(v8f x, v8f* sn, v8f* cs) {
  const v8i sign_mask = {static_cast<int>(0x80000000), static_cast<int>(0x80000000), static_cast<int>(0x80000000), static_cast<int>(0x80000000),
                         static_cast<int>(0x80000000), static_cast<int>(0x80000000), static_cast<int>(0x80000000), static_cast<int>(0x80000000)};
  const v8i abs_mask  = ~sign_mask;
  const v8i x_bits    = reinterpret_cast<v8i>(x);
  const v8i sin_sign  = x_bits & sign_mask;
  const v8f ax        = reinterpret_cast<v8f>(x_bits & abs_mask);
  v8i j               = __builtin_convertvector(ax * 1.27323954473516f, v8i);  // 4/pi
  j                   = (j + 1) & ~v8i{1, 1, 1, 1, 1, 1, 1, 1};
  const v8f y         = __builtin_convertvector(j, v8f);
  const v8f r         = ((ax - y * 0.78515625f) - y * 2.4187564849853515625e-4f) - y * 3.77489497744594108e-8f;
  const v8f z         = r * r;
  const v8f sp        = ((-1.9515295891E-4f * z + 8.3321608736E-3f) * z - 1.6666654611E-1f) * z * r + r;
  const v8f cp        = ((2.443315711809948E-005f * z - 1.388731625493765E-003f) * z + 4.166664568298827E-002f) * z * z - 0.5f * z + 1.0f;
  const v8i two       = {2, 2, 2, 2, 2, 2, 2, 2}, four = {4, 4, 4, 4, 4, 4, 4, 4};
  const v8i swap      = ((j & two) == v8i{}) ? v8i{} - 1 : v8i{};  // j&2 == 0 -> 全ビット1
  v8f s_val           = reinterpret_cast<v8f>((reinterpret_cast<v8i>(sp) & swap) | (reinterpret_cast<v8i>(cp) & ~swap));
  v8f c_val           = reinterpret_cast<v8f>((reinterpret_cast<v8i>(cp) & swap) | (reinterpret_cast<v8i>(sp) & ~swap));
  const v8i sin_flip  = ((j & four) << 29) ^ sin_sign;                      // j&4 -> 符号ビット
  const v8i cos_flip  = (((j & four) ^ ((j & two) << 1)) << 29);
  *sn                 = reinterpret_cast<v8f>(reinterpret_cast<v8i>(s_val) ^ sin_flip);
  *cs                 = reinterpret_cast<v8f>(reinterpret_cast<v8i>(c_val) ^ cos_flip);
}

inline v8f sqrt8(v8f x) {
#ifdef __AVX__
  return reinterpret_cast<v8f>(_mm256_sqrt_ps(reinterpret_cast<__m256>(x)));
#else
  v8f r;
  for(int i = 0; i < 8; i++) r[i] = __builtin_sqrtf(x[i]);
  return r;
#endif
}

inline uint32_t movemask(v8i m) {
#ifdef __AVX__
  return static_cast<uint32_t>(_mm256_movemask_ps(reinterpret_cast<__m256>(m)));
#else
  uint32_t r = 0;
  for(int i = 0; i < 8; i++) r |= (m[i] != 0 ? 1u : 0u) << i;
  return r;
#endif
}


}  // namespace vkmotion
