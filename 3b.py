import numpy as np
from esd_utils import get_esd_metrics

if __name__ == "__main__":
    # 1) 指定模型名
    model_name = "pinkmanlove/llama-7b-hf"   # 你自己的模型在HF仓库或者本地路径
    cache_dir = "/root/autodl-tmp/llm_weights"

    # 2) 计算 alpha_peak
    metrics = get_esd_metrics(model_name, metric_name="alpha_peak", cache_dir=cache_dir)

    # 如果 metrics 已经是个 numpy 数组就不用再转换了；否则可以:
    metrics_array = np.array(metrics)

    # 3) 将结果保存为 npy 文件
    save_path = "../data/llama-13b-hf/alpha_peak.npy"
    np.save(save_path, metrics_array)

    print(f"Saved alpha_peak metrics (shape={metrics_array.shape}) to {save_path}")
