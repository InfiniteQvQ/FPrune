import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, LlamaTokenizer
from typing import Dict, List
import argparse
from sklearn.cluster import KMeans

# --------------------------------------------------------------
# 1) 基于真实数据的输入函数
# --------------------------------------------------------------
def get_real_input(device, tokenizer, num_samples=8, seq_len=128):
    """
    用真实文本生成 input_ids，避免随机输入导致梯度分布不真实。
    你可以替换 sample_texts 的内容或来源。
    """
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial Intelligence is transforming the world.",
        "In a distant future, humans and AI coexist in harmony.",
        "OpenAI's ChatGPT demonstrates impressive reasoning capabilities.",
        "Large Language Models have revolutionized natural language processing."
    ]
    # 若文本不够，则重复填充
    if len(sample_texts) < num_samples:
        repeats = (num_samples // len(sample_texts)) + 1
        sample_texts = (sample_texts * repeats)[:num_samples]

    inputs = tokenizer(
        sample_texts, 
        return_tensors="pt", 
        padding="max_length", 
        truncation=True, 
        max_length=seq_len
    )
    input_ids = inputs["input_ids"].to(device)
    return input_ids


# --------------------------------------------------------------
# 2) 自动选择 k
# --------------------------------------------------------------
def auto_select_k(cka_mat: np.ndarray, max_k: int = 15):
    """
    自动计算最优分割数 k, 采用 KMeans 进行聚类。
    根据 CKA 矩阵，先对每层相似度求和，再做聚类。
    """
    num_layers = cka_mat.shape[0]
    A = np.sum(cka_mat, axis=1).reshape(-1, 1)  # [num_layers, 1]
    best_k = 1
    best_score = float('-inf')
    
    for k in range(2, min(max_k, num_layers)):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(A)
        intra_class_variance = kmeans.inertia_
        inter_class_variance = np.var([np.mean(A[labels == c]) for c in range(k)])
        score = inter_class_variance / (intra_class_variance + 1e-8)
        if score > best_score:
            best_k = k
            best_score = score

    return best_k


# --------------------------------------------------------------
# 3) CKA 计算
# --------------------------------------------------------------
def centered_cka(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    计算 Centered Kernel Alignment (CKA)
    X, Y: [N, d]
    """
    K = X @ X.T
    L = Y @ Y.T

    K_mean_row = K.mean(dim=0, keepdim=True)
    K_mean_col = K.mean(dim=1, keepdim=True)
    K_mean_all = K.mean()
    Kc = K - K_mean_row - K_mean_col + K_mean_all

    L_mean_row = L.mean(dim=0, keepdim=True)
    L_mean_col = L.mean(dim=1, keepdim=True)
    L_mean_all = L.mean()
    Lc = L - L_mean_row - L_mean_col + L_mean_all

    numerator = torch.trace(Kc @ Lc)
    denom = torch.sqrt(torch.trace(Kc @ Kc) * torch.trace(Lc @ Lc) + 1e-8)
    return numerator / denom


# --------------------------------------------------------------
# 4) Fisher 信息（可选）归一化函数
# --------------------------------------------------------------
def normalize_fisher_scores(fisher_scores: dict):
    """
    如果你在某处使用 Fisher 信息，可在此对其做 min-max 归一化 (可选)。
    这里只是一个示例函数，当前未被调用。
    """
    if not fisher_scores:
        return {}
    min_val = min(fisher_scores.values())
    max_val = max(fisher_scores.values())
    eps = 1e-8
    if abs(max_val - min_val) < eps:
        return {k: 1.0 for k in fisher_scores}
    normed = {}
    for k, v in fisher_scores.items():
        normed[k] = (v - min_val) / (max_val - min_val + eps)
    return normed


# --------------------------------------------------------------
# 5) SGLPPruner
# --------------------------------------------------------------
class SGLPPruner:
    def __init__(self, model_name: str, cache_dir: str="/root/autodl-tmp/llm_weights"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print("[Init] Loading model:", model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            cache_dir=cache_dir,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
        
        print("[Init] Loading tokenizer:", model_name)
        # 你可以根据实际情况换成适合自己模型的 tokenizer
        tokenizer_name = "meta-llama/Meta-llama-2-7b-hf"
        self.tokenizer = LlamaTokenizer.from_pretrained(tokenizer_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.config = self.model.config

        self.activations = {}
        self.gradients = {}

    def get_layers(self):
        """
        根据不同模型结构，定位到 model.layers (或 model.decoder.layers)
        """
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        elif (hasattr(self.model, "model") 
              and hasattr(self.model.model, "decoder") 
              and hasattr(self.model.model.decoder, "layers")):
            return self.model.model.decoder.layers
        else:
            raise RuntimeError("无法定位到模型 decoder layers, 请根据实际结构修改 get_layers()")

    # ----------------- Forward Hook 用于获取激活 ----------------- #
    def _hook_activations(self, module, inp, out, layer_idx: int):
        act = out[0].detach().cpu()
        self.activations[layer_idx] = act

    def compute_cka_matrix(self, num_samples=8, seq_len=512, sample_size=128):
        """
        获取中间激活后，计算各层之间的 CKA 相似度矩阵
        """
        self.activations.clear()
        layers = self.get_layers()
        num_layers = len(layers)

        # forward hooks
        hooks = []
        for idx, layer in enumerate(layers):
            h = layer.register_forward_hook(
                lambda m, i, o, idx=idx: self._hook_activations(m, i, o, idx)
            )
            hooks.append(h)
        
        self.model.eval()

        # 使用真实文本 或 随机输入
        if self.tokenizer is not None:
            input_ids = get_real_input(self.device, self.tokenizer, num_samples=num_samples, seq_len=seq_len)
        else:
            print("[Warning] No tokenizer available, fallback to random input.")
            input_ids = torch.randint(
                0, self.config.vocab_size,
                (num_samples, seq_len),
                device=self.device
            )

        with torch.no_grad():
            _ = self.model(input_ids)

        for h in hooks:
            h.remove()
        torch.cuda.empty_cache()

        cka_mat = np.zeros((num_layers, num_layers), dtype=np.float32)

        for i in range(num_layers):
            Xi = self.activations[i].float()
            Xi = Xi.reshape(-1, Xi.size(-1))
            if Xi.size(0) > sample_size:
                Xi = Xi[:sample_size, :]
            Xi_mean = Xi.mean(dim=0, keepdim=True)
            Xi = Xi - Xi_mean
            Xi = Xi.to(self.device)

            for j in range(i, num_layers):
                Xj = self.activations[j].float().reshape(-1, Xi.size(-1))
                if Xj.size(0) > sample_size:
                    Xj = Xj[:sample_size, :]
                Xj_mean = Xj.mean(dim=0, keepdim=True)
                Xj = Xj - Xj_mean
                Xj = Xj.to(self.device)

                val = centered_cka(Xi, Xj)
                item = val.item()
                cka_mat[i, j] = item
                cka_mat[j, i] = item

                del Xj
            del Xi
        
        return cka_mat

    # ----------------- Fisher Partition ----------------- #
    def fisher_segmentation(self, similarity_matrix: np.ndarray, k: int) -> Dict[int, List[int]]:
        """
        基于类似 "Fisher’s Optimal Partition" 的动态规划做分段
        输入：相似度矩阵 similarity_matrix (L x L)，以及段数 k
        输出：segments (seg_id -> [layer_idx, ...])
        """
        L = similarity_matrix.shape[0]
        A = np.sum(similarity_matrix, axis=1)

        dp = np.full((L+1, k+1), float('inf'), dtype=np.float64)
        pos = np.zeros((L+1, k+1), dtype=int)

        dp[0, 0] = 0.0
        prefix_sum = np.zeros(L+1)
        prefix_sqsum = np.zeros(L+1)
        for i in range(1, L+1):
            prefix_sum[i] = prefix_sum[i-1] + A[i-1]
            prefix_sqsum[i] = prefix_sqsum[i-1] + A[i-1]*A[i-1]

        def seg_intra(r, s):
            length = s - r
            seg_sum = prefix_sum[s] - prefix_sum[r]
            seg_sqsum = prefix_sqsum[s] - prefix_sqsum[r]
            mean_ = seg_sum / length
            return seg_sqsum - (seg_sum*seg_sum)/length

        for c in range(1, k+1):
            for i in range(c, L+1):
                best_val = float('inf')
                best_j = -1
                for j in range(c-1, i):
                    cost = dp[j, c-1] + seg_intra(j, i)
                    if cost < best_val:
                        best_val = cost
                        best_j = j
                dp[i, c] = best_val
                pos[i, c] = best_j

        boundaries = []
        cur = L
        c = k
        while c > 0:
            j = pos[cur, c]
            boundaries.append((j, cur))
            cur = j
            c -= 1
        boundaries.reverse()

        segments = {}
        seg_id_of_layer = np.zeros(L, dtype=int)
        for seg_i, (st, ed) in enumerate(boundaries):
            for layer_idx in range(st, ed):
                seg_id_of_layer[layer_idx] = seg_i

        for idx, seg_id in enumerate(seg_id_of_layer):
            segments.setdefault(seg_id, []).append(idx)

        return segments

    # ----------------- Backward Hook 获取梯度 ----------------- #
    def _hook_gradients(self, module, grad_in, grad_out, layer_idx: int):
        """
        这个函数在 backward 时被调用，grad_out[0] 就是对该层输出的梯度 (batch_size, seq_len, hidden_dim).
        """
        g = grad_out[0].detach().cpu()
        self.gradients[layer_idx] = g

    # ----------------- 计算每个 segment 的 GradNorm ----------------- #
    def compute_segment_scores2(self, segments: Dict[int, List[int]], 
                                num_samples: int = 8, seq_len: int = 128):
        """
        计算 GradNorm 作为每个 segment 的重要性 (去掉 GradNorm-Hill, 只返回原始的平均梯度范数).
        """
        self.gradients.clear()
        layers = self.get_layers()

        # **使用 register_full_backward_hook 代替 register_backward_hook (需要 PyTorch≥1.10)**
        hooks = []
        for idx, layer in enumerate(layers):
            h = layer.register_full_backward_hook(
                lambda m, gi, go, idx=idx: self._hook_gradients(m, gi, go, idx)
            )
            hooks.append(h)

        self.model.train()

        # 如果有 tokenizer，就用真实文本；否则用随机输入
        if self.tokenizer is not None:
            input_ids = get_real_input(self.device, self.tokenizer, num_samples=num_samples, seq_len=seq_len)
        else:
            print("[Warning] No tokenizer available, fallback to random input.")
            input_ids = torch.randint(
                0, self.config.vocab_size,
                (num_samples, seq_len),
                device=self.device
            )

        labels = input_ids.clone()
        outputs = self.model(input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()

        # 移除所有钩子，清理缓存
        for h in hooks:
            h.remove()
        torch.cuda.empty_cache()

        # 计算每个 segment 的平均梯度范数
        segment_gradnorms = {}
        for seg_id, layer_idxs in segments.items():
            g_norm_sum = 0.0
            valid_count = 0
            for lid in layer_idxs:
                if lid in self.gradients:
                    gn = torch.norm(self.gradients[lid]).item()
                    g_norm_sum += gn
                    valid_count += 1
            if valid_count > 0:
                segment_gradnorms[seg_id] = g_norm_sum / valid_count
            else:
                segment_gradnorms[seg_id] = 0.0

        return segment_gradnorms

    # ----------------- 按照分段分数进行剪枝 ----------------- #
    def prune_segments(self, segments: Dict[int, List[int]],
                       seg_scores: Dict[int, float],
                       prune_ratio: float=0.3):
        """
        根据 seg_scores (GradNorm 越小越不重要)，剪掉前 prune_ratio 比例的 segment。
        """
        sorted_segs = sorted(seg_scores.items(), key=lambda x: x[1])
        num_segs = len(sorted_segs)
        num_to_prune = int(num_segs * prune_ratio)
        prune_ids = [item[0] for item in sorted_segs[:num_to_prune]]

        kept_layers = []
        for sid, layer_idxs in segments.items():
            if sid not in prune_ids:
                kept_layers.extend(layer_idxs)
        kept_layers = sorted(kept_layers)

        original_layers = self.get_layers()
        new_modulelist = []
        for idx in kept_layers:
            new_modulelist.append(original_layers[idx])

        self.model.model.layers = torch.nn.ModuleList(new_modulelist)

        print(f"[Prune] Segment-based prune done: pruned {num_to_prune} / {num_segs} segments")
        print(f"[Prune] Original layers = {len(original_layers)}, remain = {len(new_modulelist)}")

    # ----------------- 可视化 ----------------- #
    def visualize(self, cka_mat: np.ndarray, segments: Dict[int, List[int]]):
        plt.figure(figsize=(12,6))

        plt.subplot(1,2,1)
        sns.heatmap(cka_mat, cmap='coolwarm', square=True)
        plt.title("Layer-wise Similarity (CKA)")

        plt.subplot(1,2,2)
        L = cka_mat.shape[0]
        seg_map = np.zeros(L, dtype=int)
        for sid, llist in segments.items():
            for lidx in llist:
                seg_map[lidx] = sid
        plt.scatter(range(L), seg_map, c=seg_map, cmap='tab10', s=50)
        plt.title("Fisher Segmentation Result")
        plt.xlabel("Layer Index")
        plt.ylabel("Segment ID")

        plt.tight_layout()
        plt.show()


# --------------------------------------------------------------
# 6) main
# --------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SGLP: Segment-based Layer Pruning via CKA & Fisher Partition")
    parser.add_argument("--model_name", type=str, required=True,
                        help="HuggingFace model name (e.g., decapoda-research/llama-7b-hf)")
    parser.add_argument("--prune_ratio", type=float, default=0.3,
                        help="How many segments to prune (fraction)")
    parser.add_argument("--no_visual", action="store_true",
                        help="whether to skip visualization")
    args = parser.parse_args()

    pruner = SGLPPruner(args.model_name)
    
    print("[Main] Computing CKA with Centered Kernel Alignment ...")
    cka_mat = pruner.compute_cka_matrix(num_samples=8, seq_len=512, sample_size=128)
    print(f"[Main] cka_mat shape={cka_mat.shape}")

    print(cka_mat)
    print("[Main] Auto-selecting num_segments with KMeans ...")
    num_segments = auto_select_k(cka_mat)
    print(f"[Main] Auto-selected num_segments = {num_segments}")

    print("[Main] Doing Fisher Optimal Segmentation ...")
    segments = pruner.fisher_segmentation(cka_mat, num_segments)
    print("Segments (seg_id -> layers):")
    for sid, ls in segments.items():
        print(f"  seg_id={sid}, layers={ls}")

    if not args.no_visual:
        pruner.visualize(cka_mat, segments)

    print("[Main] Compute GradNorm for each segment (using real text input if possible) ...")
    seg_scores = pruner.compute_segment_scores2(segments, num_samples=8, seq_len=128)

    print("[Segment Importance Scores]")
    for sid, score in seg_scores.items():
        print(f"Segment {sid}: importance={score:.4f}")

    print(f"[Main] Pruning ratio = {args.prune_ratio}")
    # 如果要执行真实剪枝，取消下行注释
    # pruner.prune_segments(segments, seg_scores, args.prune_ratio)

    # 可选: 保存剪枝后的模型
    # pruner.model.save_pretrained("pruned_model")
