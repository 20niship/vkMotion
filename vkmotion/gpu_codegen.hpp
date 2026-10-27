#pragma once

#include <string>
#include <vector>

#include "fast_collision.hpp"

namespace vkmotion {

// ロボット専用のGLSL(FK+ブロードフェーズ)を生成する。ロボットの定数(関節軸・オフセット・球・自己衝突ペア)をリテラルに展開し、リンク姿勢を静的インデックスのレジスタに置く。
// 汎用版(shaders/include/robot_collision_common.glsl)は同じ情報をバッファから読んで動的ループで回すため、1回の判定が依存ロードの連鎖で百µs台になる。
// 出力は shaders/include/robot_collision_common.glsl の ROBOT_GENERATED 分岐から #include される "robot_generated.glsl" の中身。
std::string generate_robot_glsl(const GpuRobotData& robot);

// entry_shader(shaders/ 直下の .comp)を、生成したGLSLを使って -DROBOT_GENERATED でコンパイルし、SPIR-Vのパスを返す。
// 結果はディスクにキャッシュする(キー=ソース全体のハッシュ)。glslcが無い・コンパイルに失敗した場合は空文字(呼び出し側は汎用のSPIR-Vに戻る)。
// 環境変数 VKMOTION_NO_CODEGEN が設定されていれば常に空文字、VKMOTION_SHADER_CACHE でキャッシュ先を変更できる。
std::string compile_specialized_shader(const std::string& entry_shader, const GpuRobotData& robot, const std::vector<std::string>& extra_defines = {});

// CPU版(8レーンSIMD)のロボット専用カーネルをC++として生成する。GPU版と同じ考え方(FK・境界球・自己衝突ペアを全部リテラルに展開してレジスタで回す)。
// 汎用実装(fast_collision.cpp の mask8_impl)は同じ情報をvectorから読んでループで回すので、8コンフィグの判定が約2µsかかる。
// シグネチャ: uint32_t vkm_mask8(const float* q /*lane*dof+j*/, float margin, uint32_t required, const float* boxes, int nboxes, const float* balls, int nballs)
// boxes/ballsはpack_gpu_worldの出力(boxは24 float、ballは8 float)。
std::string generate_robot_cpp(const GpuRobotData& robot, size_t dof);

using CpuMask8Fn = uint32_t (*)(const float*, float, uint32_t, const float*, int, const float*, int);

// 生成したC++をシステムのコンパイラ(CMake時に見つけたもの)で共有ライブラリにして読み込む(結果はディスクにキャッシュ)。
// コンパイラが無い・失敗した場合はnullptr(呼び出し側は汎用実装に戻る)。環境変数 VKMOTION_NO_CODEGEN で無効化。
CpuMask8Fn load_cpu_kernel(const GpuRobotData& robot, size_t dof);

}  // namespace vkmotion
