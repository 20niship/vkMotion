# vkMotion - Vulkan Motion planning library

- cuMotion (CUDAを使ったMotion planning)のVulkan版。できるだけAPIの関数名などを互換性持たせること


## 使用内容

- CMake, Vulkan (validation layer + VMA使うこと)
- test: doctest https://github.com/doctest/doctestを使用すること。testをcmake optionでenableした時だけfetchContentで取得しctestでtestできるように
- matrixとかvec4とかの内容, smart pointerなどはhttps://github.com/20niship/cutilを使って欲しい
- CI CDを使ってCPUでのmotion planning / 衝突判定やハードコードしたGT結果と比較すること



## 注意

- test用にOpenArm, SO101, URシリーズのURDFを読み込み衝突判定をするようにしたい
- cuMotionは球体として衝突空間を作っているのでmeshなどは不要。STLや他meshファイルはcommitせず純粋な.urdfとして定義すること
- 障害物(CollisionObject)側もcuMotionの内容と語感ん対応すること(Box, Cylinder, Sphere, SDF? ここら辺はcuMotionの仕様を調べてそれに合わせる)


## READMEと使い方を日本語で書くこと
