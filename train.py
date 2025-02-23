import os
import subprocess

# 设定通用参数
model = "meta-llama/Meta-Llama-3-8B"
cache_dir = "/root/autodl-tmp/llm_weights/"
save_base_dir = "results/llama3_8b"
ww_metric = "alpha_peak"
ww_metric_cache = "./data/llama-3-8b"
epsilon = 0.3

# 需要测试的剪枝方法
prune_methods = ["wanda_ww", "magnitude_ww", "sparsegpt_ww"]
# 需要测试的 sparsity 比例
sparsity_ratios = [0.3, 0.4, 0.5,0.55, 0.6, 0.7]

# 需要运行的 Python 文件，main.py 和 start.py 需要不同的保存路径
scripts = {
    "main.py": "llama3_8b_alpha",
    "start.py": "llama3_8b_fusion"
}

# 创建保存目录
for script_name, save_sub_dir in scripts.items():
    save_dir = os.path.join(save_base_dir, save_sub_dir)
    os.makedirs(save_dir, exist_ok=True)

# 遍历所有剪枝方法和 sparsity 组合，执行实验
for script_name, save_sub_dir in scripts.items():
    for sparsity in sparsity_ratios:
        for prune_method in prune_methods:
            # 构建保存路径
            save_dir = os.path.join(save_base_dir, save_sub_dir)
            os.makedirs(save_dir, exist_ok=True)

            # 构建命令
            command = [
                "python", script_name,
                "--model", model,
                "--cache_dir", cache_dir,
                "--prune_method", prune_method,
                "--sparsity_ratio", str(sparsity),
                "--save", save_dir,
                "--ww_metric", ww_metric,
                "--ww_metric_cache", ww_metric_cache,
                "--epsilon", str(epsilon)
            ]

            # 打印当前实验配置
            print(f"\n🚀 Running {script_name} with {prune_method} at sparsity {sparsity}...")
            print(f"💾 Results will be saved to: {save_dir}")

            # 运行实验
            try:
                subprocess.run(command, check=True)
                print(f"✅ Completed: {script_name} - {prune_method} - sparsity {sparsity}")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed: {script_name} - {prune_method} - sparsity {sparsity}")
                print(f"Error: {e}")

print("\n🎉 All experiments finished!")
