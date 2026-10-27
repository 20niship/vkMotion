#!/usr/bin/env python3
"""results/*.json から RESULTS.md を生成する。FK・衝突・経路の正誤はVAMPを基準にする(3実装は同一の衝突球/障害物/ACMを使う)。"""
# results/baseline/ にvkMotionの最適化前の結果があれば改善前後の表も出す。
import argparse
import json
import subprocess
from datetime import date
from pathlib import Path

import numpy as np
import vamp
import vamp.transformations as T

from common import DATA, RESULTS, load_samples
from prepare import make_vamp_env, quat_to_wxyz

ORDER = ["vamp", "curobo", "vkmotion-vulkan", "vkmotion-cpu", "vkmotion-cpu-mt", "vkmotion-gpu", "vkmotion-gpu-batch"]
LABEL = {
    "vamp": "VAMP (CPU SIMD, 1 thread)",
    "curobo": "cuRobo v2 (GPU)",
    "vkmotion-vulkan": "vkMotion (Vulkan + CPU)",
    "vkmotion-cpu": "vkMotion (CPU, 1 thread)",
    "vkmotion-cpu-mt": "vkMotion (CPU, 8 threads)",
    "vkmotion-gpu": "vkMotion (GPU探索)",
    "vkmotion-gpu-batch": "vkMotion (GPUバッチ32, 1問あたり)",
}
PLAN_SKIP = {"vkmotion-vulkan"}  # 計画ループの衝突判定は小バッチ(CPU SIMD)なのでVulkan有効でも-cpuと同一経路
KERNEL_ONCE = {"vkmotion-cpu-mt", "vkmotion-gpu", "vkmotion-gpu-batch"}  # FK/衝突判定スループットは-cpuと同じ計測なので重複させない


def pct(x, p):
    return float(np.percentile(x, p)) if len(x) else float("nan")


def fmt(x, nd=2):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:,.{nd}f}"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def load_dir(d):
    res = {}
    for lib in ORDER:
        p = d / f"{lib}.json"
        if p.exists():
            res[lib] = json.loads(p.read_text())
    return res


def problem_index():
    d = json.loads((DATA / "problems.json").read_text())["problems"]
    return {(p["scene"], p["index"]): p for p in d}


def path_valid(path, env):
    """全waypoint間をVAMPの連続判定(補間してvalidate)で検証。"""
    if path is None or len(path) < 2:
        return False
    q = [list(map(float, x)) for x in path]
    if not vamp.panda.validate(q[0], env) or not vamp.panda.validate(q[-1], env):
        return False
    return all(vamp.panda.validate_motion(a, b, env) for a, b in zip(q[:-1], q[1:]))


def ee_err(q_final, goal_pose):
    m = np.asarray(vamp.panda.eefk(list(map(float, q_final))), dtype=float)
    p_err = np.linalg.norm(m[:3, 3] - np.array(goal_pose[:3]))
    qa = np.array(quat_to_wxyz(T.quaternion_from_matrix(m)))
    qb = np.array(goal_pose[3:])
    r_err = 2 * np.degrees(np.arccos(min(1.0, abs(float(qa @ qb)))))
    return p_err * 1e3, r_err


def dynamics(row):
    """経路+時刻から最大関節速度・最大関節加速度を有限差分で求める(時刻が無ければNone)。"""
    q = np.asarray(row["path"], dtype=float)
    if "times" in row:
        t = np.asarray(row["times"], dtype=float)
    elif "dt" in row:
        t = np.arange(len(q)) * row["dt"]
    else:
        return None
    if len(q) < 3:
        return None
    dt = np.diff(t)
    ok = dt > 1e-6
    v = np.diff(q, axis=0)[ok] / dt[ok, None]
    dtm = 0.5 * (dt[ok][1:] + dt[ok][:-1])
    a = np.diff(v, axis=0) / dtm[:, None]
    return float(np.abs(v).max()), float(np.abs(a).max()) if len(a) else 0.0


def section_fk(res, samples):
    ref = np.array(samples["fk"])[:, :3]
    rows = []
    for lib, d in res.items():
        if "fk" not in d or lib in KERNEL_ONCE:
            continue
        err = np.linalg.norm(np.array(d["fk"]) - ref, axis=1) * 1e3
        rows.append([LABEL[lib], fmt(d["fk_us_per_call"], 3), fmt(err.mean(), 4), fmt(err.max(), 4)])
    return table(["実装", "FK [µs/config]", "平均誤差 [mm]", "最大誤差 [mm]"], rows)


def section_collision(res, samples):
    rows = []
    for lib, d in res.items():
        if "collision" not in d or lib in KERNEL_ONCE:
            continue
        us, us1, tp, fp, fn, tn = [], [], 0, 0, 0, 0
        for scene, v in d["collision"].items():
            ref = np.array(samples["scenes"][scene]["valid"])
            got = np.array(v["valid"])
            tp += int(((got == 1) & (ref == 1)).sum())
            tn += int(((got == 0) & (ref == 0)).sum())
            fp += int(((got == 1) & (ref == 0)).sum())  # 衝突を見逃す(危険側)
            fn += int(((got == 0) & (ref == 1)).sum())  # 過剰検出
            us.append(v["us_per_check"])
            if "us_per_check_1thread" in v:
                us1.append(v["us_per_check_1thread"])
        n = tp + tn + fp + fn
        med = float(np.median(us))
        rows.append([LABEL[lib], fmt(med, 3), f"{1e6 / med:,.0f}", fmt(float(np.median(us1)), 3) if us1 else "-", f"{100 * (tp + tn) / n:.2f}%", fp, fn])
    return table(["実装", "バッチ µs/check (中央値)", "checks/s", "1スレッド逐次 µs/check", "VAMPとの一致率", "見逃し(衝突を無衝突と判定)", "過剰検出"], rows)


def plan_stats(rows, envs):
    rows = [r for r in rows if r.get("valid")]
    ok = [r for r in rows if r["success"]]
    t = [r["time_ms"] for r in ok]
    n_valid = 0
    for r in ok:
        env = make_vamp_env(vamp, envs[(r["scene"], r["index"])]["objects"])
        n_valid += int(path_valid(r["path"], env))
    dyn = [d for d in (dynamics(r) for r in ok) if d]
    mt = [r["motion_time_s"] for r in ok if r.get("motion_time_s")]
    return dict(n=len(rows), succ=len(ok) / max(1, len(rows)), t=t, len=[r["path_len"] for r in ok], valid=n_valid / len(ok) if ok else float("nan"),
                vmax=[d[0] for d in dyn], amax=[d[1] for d in dyn], mt=mt, ok_ids={(r["scene"], r["index"]): r for r in ok})


def section_plan(res, key, envs, with_pose_err=False):
    stats = {lib: plan_stats(d[key], envs) for lib, d in res.items() if key in d and lib not in PLAN_SKIP}
    common = None
    for s in stats.values():
        ids = set(s["ok_ids"])
        common = ids if common is None else common & ids
    speed, quality = [], []
    for lib, s in stats.items():
        ct = [r["time_ms"] for k, r in s["ok_ids"].items() if k in common]
        cl = [r["path_len"] for k, r in s["ok_ids"].items() if k in common]
        speed.append([LABEL[lib], s["n"], f"{100 * s['succ']:.1f}%", fmt(pct(s["t"], 50)), fmt(float(np.mean(s["t"])) if s["t"] else float("nan")), fmt(pct(s["t"], 95)),
                      fmt(pct(s["t"], 99)), fmt(float(np.median(ct))) if ct else "-"])
        row = [LABEL[lib], fmt(pct(s["len"], 50), 3), fmt(float(np.median(cl)), 3) if cl else "-", f"{100 * s['valid']:.1f}%",
               fmt(pct(s["mt"], 50), 2) if s["mt"] else "-", fmt(pct(s["amax"], 50), 1) if s["amax"] else "-"]
        if with_pose_err:
            errs = [ee_err(r["path"][-1], envs[k]["goal_pose"]) for k, r in s["ok_ids"].items()]
            row += [fmt(float(np.median([e[0] for e in errs])), 3) if errs else "-", fmt(float(np.median([e[1] for e in errs])), 3) if errs else "-"]
        quality.append(row)
    qh = ["実装", "経路長 median [rad]", "経路長(共通集合)", "VAMP検証で衝突なし", "動作時間 median [s]", "最大関節加速度 median [rad/s²]"]
    if with_pose_err:
        qh += ["末端位置誤差 median [mm]", "末端姿勢誤差 median [deg]"]
    return (table(["実装", "問題数", "成功率", "時間 median [ms]", "mean", "p95", "p99", "時間 median(共通集合) [ms]"], speed) + "\n\n" + table(qh, quality) +
            f"\n\n共通集合 = 全実装が成功した問題 {len(common)} 件。動作時間・加速度は時刻を持つ実装(vkMotion, cuRobo)のみ。")


def section_ik(res):
    rows = []
    for lib, d in res.items():
        if "ik" not in d or lib in KERNEL_ONCE or lib in PLAN_SKIP:
            continue
        r = [x for x in d["ik"] if x.get("valid")]
        ok = [x for x in r if x["success"]]
        rows.append([LABEL[lib], len(r), f"{100 * len(ok) / max(1, len(r)):.1f}%", fmt(pct([x["time_ms"] for x in ok], 50)), fmt(pct([x["time_ms"] for x in ok], 95)),
                     fmt(pct([x["pos_err_mm"] for x in ok if "pos_err_mm" in x], 50), 4), fmt(pct([x["rot_err_deg"] for x in ok if "rot_err_deg" in x], 50), 4)])
    return table(["実装", "問題数", "成功率", "時間 median [ms]", "p95", "位置誤差 median [mm]", "姿勢誤差 median [deg]"], rows)


def section_before_after(now, base):
    """同じ問題集合(baselineが解いた問題)での vkMotion 最適化前後。"""
    rows = []
    for lib in ("vkmotion-cpu", "vkmotion-vulkan"):
        if lib not in base or lib not in now:
            continue
        for key, name in (("plan_js", "関節ゴール計画"), ("plan_pose", "姿勢ゴール計画"), ("ik", "IK")):
            if key not in base[lib] or key not in now[lib]:
                continue
            ids = {(r["scene"], r["index"]) for r in base[lib][key]}
            b = [r for r in base[lib][key]]
            n = [r for r in now[lib][key] if (r["scene"], r["index"]) in ids]
            bt = [r["time_ms"] for r in b if r["success"]]
            nt = [r["time_ms"] for r in n if r["success"]]
            rows.append([LABEL[lib], name, len(b), f"{100 * len(bt) / len(b):.1f}%", fmt(pct(bt, 50)), f"{100 * len(nt) / len(n):.1f}%", fmt(pct(nt, 50)),
                         f"{pct(bt, 50) / pct(nt, 50):.0f}x" if nt and bt else "-"])
        if "collision" in base[lib] and "collision" in now[lib]:
            bu = float(np.median([v["us_per_check"] for v in base[lib]["collision"].values()]))
            nu = float(np.median([v["us_per_check"] for v in now[lib]["collision"].values()]))
            rows.append([LABEL[lib], "衝突判定(バッチ µs/check)", "-", "-", fmt(bu, 3), "-", fmt(nu, 3), f"{bu / nu:.0f}x"])
    if not rows:
        return ""
    return table(["実装", "項目", "問題数", "成功率(前)", "median(前) [ms]", "成功率(後)", "median(後) [ms]", "高速化"], rows)


NOTES = """
- **VAMPとの比較(正直なところ)**: 関節ゴールの中央値はvkMotionが約0.2ms、VAMPが約0.35msで、1.6倍ほど速いが「数倍」ではない。簡単な問題が多い`box`/`table_*`では約2倍速い。
  裾(p95/p99)は依然としてVAMPが良い(VAMP p95≈0.9ms/p99≈1.7ms。vkMotionはGPU探索を使うGPUが温まっているとp95≈2ms/p99≈3ms、GPUが冷えているとp95≈3〜4ms/p99≈6ms)。
  難しいシーン`cage`は、vkMotionの中央値が約0.9〜1.9ms、VAMPが約0.7ms。ただし全210問の最大は、以前の20ms近くから、GPUが温まっていれば約3〜4msまで縮んだ(VAMPの最大は約6ms)。
  VAMPの時間は`time_ms`=1回目の実行(plan+simplify)で、`time_ms_median3`は3回の中央値(以前はこちらを使っていたため裾が隠れていた)。
- **今回効いたこと**: (1) 経路探索の木のバッファがホスト(システム)メモリに置かれていてノードの読み書きがPCIe越しだった。デバイスメモリに置いたところ、GPU探索の難しい問題が数倍速くなった(これが最大の改善)。
  (2) 最近傍探索を8ノードずつのSIMDにした(CPU)、(3) ロボット専用のSIMD/GPUカーネルを実行時に生成(`gpu_codegen`、FK・境界球・自己衝突ペアをリテラルに展開)、
  (4) 最初のRRT試行の制限時間を3msから20msに、(5) 端点がすれすれで木が伸びないときだけmarginを縮める、(6) 経路短縮(shortcut)の反復を60回から30回に(経路長は約2%増)。
- **GPU探索**(`MotionGenConfig::gpu_search`、既定ON): 1本の木を多数のワープで共有して伸ばすVulkan compute版RRT-Connect。ラウンドをdispatchに分けて1回のsubmitに並べる(CUDA graphに相当)。
  submitの固定費(約0.4ms)があるので、先にCPUで`gpu_handoff_s`(0.3ms)だけ試し、解けなければGPUに渡す。つまり簡単な問題はCPU、難しい問題(上限を決める裾)はGPUが解く。
  **GPUはアイドルが続くとクロックを数百MHzまで落とし**、間隔を空けて呼ぶGPU探索が2倍近く遅くなる(この表のGPU探索の行がぶれる主因。別のGPUジョブが動いていると逆に温まって速くなる)。
  `sudo nvidia-smi -lgc <min>,<max>`でクロックを固定すると安定する。補助キューで空回しを流してクロックを維持する案も試したが、計画のdispatchを待たせて逆効果だったので入れていない。
- **GPUバッチ**(`MotionGen::plan_batch_js`): 問題ごとに別の世界を持つ複数問題を1回のGPU submit列で解く。32問ずつだと全問を解く。1問あたりの時間(バッチ時間/32)は中央値0.46msと単発より長いが、最悪値(p95/p99)は約1.4msと単発より短い(簡単な問題がバッチの完了を待つため中央値は伸び、難しい問題は並列に解くため裾が縮む)。
  GPU全体の衝突判定スループット(約85M判定/秒)は、CPUのコア約2〜3個分に相当するので、多数の問題をまとめても単発CPUより大きくは速くならない。
- **まだ差が残る理由**: VAMPの`cage`は約1000反復を0.6msで回す(1反復0.6µs)。vkMotionは1反復あたり約1.8µsで、8コンフィグの衝突判定は約0.5〜0.8µs(VAMPは約0.2µs)。
  残りの大半は箱×リンクグループのブロードフェーズ(1回の判定の約3割)とFK。VAMPのように障害物を距離順に並べて打ち切る、箱をworld軸に揃った場合は回転を省く、といった工夫では速くならなかった。
- **精度**: FK・衝突判定(2000サンプル)・姿勢ゴールの末端誤差は各実装で同等(サブマイクロメートル〜0.01mm台、衝突判定はVAMPと100%一致)。`VAMP検証で衝突なし`はVAMP自身・cuRoboも100%ではない。
  サンプリングで検査するプランナーは、障害物に接した経路を作ると別の検証器の検査点では微小に食い込むため。vkMotionは経路の内部点に`collision_margin`(5mm)の余裕を持たせてこれを抑えるが、
  端点が障害物すれすれで木が伸びないときだけ余裕を縮めるので、その問題では食い込む経路が出うる(98〜99.5%)。
- **衝突判定スループット**: 10万config一括ではVulkanバッチ・CPU SIMD(8スレッド)ともcuRoboを上回る。VAMPの値はpythonバインディングの呼び出しオーバーヘッドを含む。
- **cuRoboとの関係**: cuRoboは姿勢ゴールをGPU上の多シードIK + 多シード軌道最適化(CUDA graph)で解き、PRMグラフは再試行時のシード用。スループット重視で1回数百ms。
  vkMotionは幾何経路(RRT-Connect)なので3桁速いが、経路は折れ線でjerk制限を持たず、経路長はcuRoboより長い。
- **IK**: vkMotionは運動学的IKの解を衝突判定で棄却する方式(解析ヤコビアン+LM、多シード)。cuRoboは衝突コストを最適化に含める。
- **計測の注意**: 他プロセスの負荷は制御していない(冒頭の「計測開始時のマシン負荷」)。以前の版はollamaがCPUを約1000%使っていた状態の数値で、VAMPが約1.5msに見えていたが、
  負荷が無い状態では0.35msだった。GPUも他セッションと共有しており、GPU探索の数値は±数十%ぶれる。cuRoboはmax_attempts=10・graph seed 2回目以降で実行(既定は5)。
"""


def env_info():
    def sh(c):
        try:
            return subprocess.check_output(c, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return "?"
    return {"cpu": sh("grep -m1 'model name' /proc/cpuinfo | sed 's/.*: *//'"), "gpu": sh("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader"),
            "commit": sh("git rev-parse --short HEAD")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).parent / "RESULTS.md"))
    a = ap.parse_args()
    res = load_dir(RESULTS)
    base = load_dir(RESULTS / "baseline")
    samples = load_samples()
    envs = problem_index()
    info = env_info()
    n_problems = len({(r["scene"], r["index"]) for d in res.values() for r in d.get("plan_js", [])})
    env_note = (RESULTS / "env.txt").read_text().strip().replace("\n", " / ") if (RESULTS / "env.txt").exists() else "不明"
    md = ["# vkMotion ベンチマーク結果 (vs VAMP / cuRobo)\n",
          f"- 生成日: {date.today()} / vkMotion commit: `{info['commit']}`(作業ツリー)",
          f"- CPU: {info['cpu']} / GPU: {info['gpu']}",
          "- ロボット: Franka Panda(7DoF)。衝突球はVAMPの`panda_spherized.urdf`(59球)を3実装で共有、ACMもVAMPのSRDFと同一。",
          f"- 問題: MotionBenchMaker (VAMP同梱, `bookshelf_*`/`cage`/`box`/`table_*`) {n_problems}件。障害物は直方体(円柱は外接直方体に正規化)。",
          "- 時間はプランナー呼び出し1回のwall clock(IK・時間パラメタライズ込み、シーン更新は除く)。cuRoboはCUDA graph warmup済み。",
          f"- 計測開始時のマシン負荷: {env_note}(負荷が高いとCPU側の時間が悪化する。特にスレッド並列の行)",
          "- 再現手順: [bench/README.md](./README.md)\n"]
    ba = section_before_after(res, base)
    if ba:
        md += ["## 0. vkMotionの改善前後 (同一問題・同一マシン)\n", ba, ""]
    md += ["## 1. FK (精度・速度)\n", section_fk(res, samples), ""]
    md += ["## 2. 衝突判定 (精度・スループット)\n",
           "基準はVAMP。バッチ=10万configを1回で判定(GPU実装はdispatch1回、vkMotion(CPU)はSIMD+8スレッド)。"
           "VAMPはpythonバインディング経由の1config呼び出しなので呼び出しオーバーヘッド込み(ネイティブはこれより速い)。\n",
           section_collision(res, samples), ""]
    md += ["## 3. 関節空間ゴールの軌道生成 (start/goal=関節角)\n", section_plan(res, "plan_js", envs), ""]
    md += ["## 4. 姿勢ゴールの軌道生成 (IK+軌道生成)\n", "VAMPは関節空間プランナーのため対象外。\n", section_plan(res, "plan_pose", envs, with_pose_err=True), ""]
    md += ["## 5. IK単体 (衝突回避あり)\n", section_ik(res), ""]
    md += ["## 6. 読み方と、まだ差が残っている点\n", NOTES]
    Path(a.out).write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
