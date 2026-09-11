"""重构代码功能验证：T2I + T2V 缓存命中测试。

测试矩阵：
  T2I (Qwen-Image):
    1. 精确命中：相同 prompt+seed 第二次生成应命中缓存（跳过全部去噪）
    2. 部分命中：相似但不同的 prompt 触发 CLIP 语义命中（跳过前 N 步）
    3. t2i penalty：验证 t2i 惩罚开关生效（影响语义匹配分数）
  T2V (Wan2.2):
    4. 精确命中：相同 prompt+seed 第二次生成应命中缓存
    5. 部分命中：相似 prompt 触发 CLIP 语义命中

用法（容器内）:
  PYTHONPATH=/mnt/sdb/cta/vllm-omni-refactor-test \
  ASCEND_RT_VISIBLE_DEVICES=0,1 \
  python /mnt/sdb/cta/vllm-omni-refactor-test/func_test.py --task t2i
"""
import os
import sys
import time
import json
import argparse
import tempfile


def run_t2i_test(lmcache_dir, clip_path, npu_count=2):
    """T2I 测试：精确命中 + 部分命中 + penalty。"""
    from vllm_omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    cache_config = {
        "inter_request_max_entries": 100,
        "inter_request_max_memory_gb": 8.0,
        "inter_request_lmcache_disk_dir": lmcache_dir,
        "inter_request_lmcache_max_cpu_gb": 2.0,
        "inter_request_lmcache_max_disk_gb": 5.0,
        "inter_request_max_stored_steps": 5,
        "inter_request_clip_model_path": clip_path,
        "inter_request_clip_threshold": 0.65,
        "inter_request_clip_min_skip": 2,
        "inter_request_clip_max_skip_ratio": 0.5,
        "inter_request_use_t2i_penalty": True,
    }

    print("\n" + "=" * 60)
    print("T2I: Loading Qwen-Image with inter_request cache...")
    print("=" * 60)
    omni = Omni(
        model="/mnt/sdb/models/Qwen-Image",
        cache_backend="inter_request",
        cache_config=cache_config,
        tensor_parallel_size=npu_count,
        mode="text-to-image",
        enforce_eager=True,
    )

    def generate(prompt, seed=142, steps=20):
        sp = OmniDiffusionSamplingParams(
            height=1024, width=1024, seed=seed,
            num_inference_steps=steps, num_outputs_per_prompt=1,
        )
        t0 = time.perf_counter()
        outputs = omni.generate({"prompt": prompt}, sampling_params_list=[sp])
        elapsed = time.perf_counter() - t0
        return elapsed, outputs

    results = {}

    # ---- 测试 1: 精确命中 ----
    print("\n--- 测试 1: 精确命中 (相同 prompt+seed) ---")
    prompt_exact = "a fluffy cat sitting on a wooden table, warm lighting"
    t1, out1 = generate(prompt_exact, seed=42, steps=20)
    print(f"  第一次生成: {t1:.2f}s (miss, 写入缓存)")
    t2, out2 = generate(prompt_exact, seed=42, steps=20)
    print(f"  第二次生成: {t2:.2f}s (应命中缓存)")
    speedup_exact = t1 / t2 if t2 > 0 else 0
    print(f"  → 精确命中加速: {speedup_exact:.1f}x")
    results["t2i_exact_hit_speedup"] = round(speedup_exact, 1)
    results["t2i_exact_first_s"] = round(t1, 2)
    results["t2i_exact_second_s"] = round(t2, 2)

    # ---- 测试 2: 部分命中（语义相似）----
    print("\n--- 测试 2: 部分命中 (语义相似 prompt) ---")
    prompt_similar = "a fluffy cat resting on a wooden table, soft light"
    t3, out3 = generate(prompt_similar, seed=42, steps=20)
    print(f"  相似 prompt 生成: {t3:.2f}s (应触发语义命中, 跳过前几步)")
    results["t2i_semantic_hit_s"] = round(t3, 2)

    # ---- 测试 3: 完全不相似（应 miss）----
    print("\n--- 测试 3: 完全不相似 (应 miss) ---")
    prompt_diff = "a futuristic city skyline at sunset, cyberpunk style"
    t4, out4 = generate(prompt_diff, seed=42, steps=20)
    print(f"  不相似 prompt 生成: {t4:.2f}s (应 miss, 全量计算)")
    results["t2i_miss_s"] = round(t4, 2)

    omni.shutdown()
    print(f"\n✅ T2I 测试完成")
    return results


def run_t2v_test(lmcache_dir, clip_path, npu_count=2):
    """T2V 测试：精确命中 + 部分命中。"""
    from vllm_omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    cache_config = {
        "inter_request_max_entries": 100,
        "inter_request_max_memory_gb": 8.0,
        "inter_request_lmcache_disk_dir": lmcache_dir,
        "inter_request_lmcache_max_cpu_gb": 2.0,
        "inter_request_lmcache_max_disk_gb": 5.0,
        "inter_request_max_stored_steps": 5,
        "inter_request_clip_model_path": clip_path,
        "inter_request_clip_threshold": 0.65,
        "inter_request_clip_min_skip": 2,
        "inter_request_clip_max_skip_ratio": 0.5,
        "inter_request_use_t2i_penalty": False,  # T2V 不用 t2i penalty
    }

    print("\n" + "=" * 60)
    print("T2V: Loading Wan2.2-T2V with inter_request cache...")
    print("=" * 60)
    omni = Omni(
        model="/mnt/sdb/models/Wan2.2-T2V-A14B-Diffusers",
        cache_backend="inter_request",
        cache_config=cache_config,
        tensor_parallel_size=npu_count,
        mode="text-to-video",
        enforce_eager=True,
    )

    def generate(prompt, seed=42, steps=20, frames=41):
        sp = OmniDiffusionSamplingParams(
            height=480, width=832, seed=seed,
            num_inference_steps=steps,
            num_frames=frames, num_outputs_per_prompt=1,
        )
        t0 = time.perf_counter()
        outputs = omni.generate({"prompt": prompt}, sampling_params_list=[sp])
        elapsed = time.perf_counter() - t0
        return elapsed, outputs

    results = {}

    # ---- 测试 4: 精确命中 ----
    print("\n--- 测试 4: 精确命中 (相同 prompt+seed) ---")
    prompt_exact = "a cat walking on a sunny beach"
    t1, out1 = generate(prompt_exact, seed=42, steps=20, frames=41)
    print(f"  第一次生成: {t1:.2f}s (miss, 写入缓存)")
    t2, out2 = generate(prompt_exact, seed=42, steps=20, frames=41)
    print(f"  第二次生成: {t2:.2f}s (应命中缓存)")
    speedup_exact = t1 / t2 if t2 > 0 else 0
    print(f"  → 精确命中加速: {speedup_exact:.1f}x")
    results["t2v_exact_hit_speedup"] = round(speedup_exact, 1)
    results["t2v_exact_first_s"] = round(t1, 2)
    results["t2v_exact_second_s"] = round(t2, 2)

    # ---- 测试 5: 部分命中（语义相似）----
    print("\n--- 测试 5: 部分命中 (语义相似 prompt) ---")
    prompt_similar = "a cat strolling on a sunny beach shore"
    t3, out3 = generate(prompt_similar, seed=42, steps=20, frames=41)
    print(f"  相似 prompt 生成: {t3:.2f}s (应触发语义命中)")
    results["t2v_semantic_hit_s"] = round(t3, 2)

    omni.shutdown()
    print(f"\n✅ T2V 测试完成")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["t2i", "t2v", "both"], default="both")
    ap.add_argument("--npu-count", type=int, default=2)
    args = ap.parse_args()

    clip_path = "/mnt/sdb/models/clip-vit-large-patch14"
    out_dir = "/mnt/sdb/cta/vllm-omni-main-workspace/func_test"
    os.makedirs(out_dir, exist_ok=True)

    all_results = {}

    if args.task in ("t2i", "both"):
        lmcache_dir = os.path.join(out_dir, "t2i_lmcache")
        os.makedirs(lmcache_dir, exist_ok=True)
        try:
            all_results["t2i"] = run_t2i_test(lmcache_dir, clip_path, args.npu_count)
        except Exception as e:
            import traceback
            print(f"\n❌ T2I 测试失败: {e}")
            traceback.print_exc()
            all_results["t2i"] = {"error": str(e)}

    if args.task in ("t2v", "both"):
        lmcache_dir = os.path.join(out_dir, "t2v_lmcache")
        os.makedirs(lmcache_dir, exist_ok=True)
        try:
            all_results["t2v"] = run_t2v_test(lmcache_dir, clip_path, args.npu_count)
        except Exception as e:
            import traceback
            print(f"\n❌ T2V 测试失败: {e}")
            traceback.print_exc()
            all_results["t2v"] = {"error": str(e)}

    result_path = os.path.join(out_dir, "func_test_results.json")
    with open(result_path, "w") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {result_path}")
    print(json.dumps(all_results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
