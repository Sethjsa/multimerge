import json
import os
import sys
import numpy as np
from scipy import linalg
from safetensors import safe_open
from pathlib import Path

def load_weights_safetensors(model_path):
    """Load weights from safetensors format (most efficient)"""
    weights = {}
    safetensors_files = list(Path(model_path).glob("*.safetensors"))
    
    if not safetensors_files:
        return None
    
    for sf_path in safetensors_files:
        with safe_open(sf_path, framework="numpy") as f:
            for key in f.keys():
                weights[key] = f.get_tensor(key)
    
    return weights

def load_weights_pytorch(model_path):
    """Fallback: load PyTorch weights and convert to numpy"""
    import torch
    
    # Try to find .bin or .pth files
    pt_files = list(Path(model_path).glob("*.bin")) + list(Path(model_path).glob("*.pth"))
    
    if not pt_files:
        # Load using transformers as last resort
        from transformers import AutoModel, AutoConfig
        config = AutoConfig.from_pretrained(model_path)
        model = AutoModel.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True
        )
        state_dict = model.state_dict()
        return {k: v.cpu().numpy() for k, v in state_dict.items()}
    
    # Load from checkpoint files directly
    weights = {}
    for pt_file in pt_files:
        state_dict = torch.load(pt_file, map_location='cpu', weights_only=True)
        for k, v in state_dict.items():
            weights[k] = v.cpu().numpy() if hasattr(v, 'numpy') else v
    
    return weights

def get_state_dict(model_path):
    """Load model weights efficiently, preferring safetensors"""
    # Try safetensors first (fastest, no torch dependency)
    weights = load_weights_safetensors(model_path)
    
    if weights is None:
        # Fallback to PyTorch loading
        weights = load_weights_pytorch(model_path)
    
    return weights

def extract_relevant_keys(sd):
    """Keep keys for attention and FFNN layers"""
    keywords = ['attn', 'attention', 'q_proj', 'k_proj', 'v_proj', 'out_proj', 
                'qkv', 'wq', 'wk', 'wv', 'wo', 'mlp', 'ffn', 'fc', 'dense']
    filtered = {}
    
    for k, v in sd.items():
        lkey = k.lower()
        # Only compare weight matrices (ignore bias or norm params)
        if any(word in lkey for word in keywords) and v.ndim >= 2:
            filtered[k] = v
    
    return filtered

def frobenius_norm_numpy(arr):
    """Compute Frobenius norm using NumPy (more efficient than scipy for this)"""
    # Frobenius norm is just the L2 norm of the flattened array
    return np.linalg.norm(arr.astype(np.float32).ravel())

def compute_rank(arr):
    """Compute the rank of a matrix using NumPy"""
    # Convert to float32 for consistency
    arr_float = arr.astype(np.float32)
    # Use numpy's matrix_rank which uses SVD
    return np.linalg.matrix_rank(arr_float)

def main():
    if len(sys.argv) != 3:
        print("Usage: python magnitude.py <model_path_1> <model_path_2>")
        sys.exit(1)
    
    m1, m2 = sys.argv[1:3]
    print(f"Model 1: {m1}")
    print(f"Model 2: {m2}")
    
    name1 = os.path.basename(os.path.normpath(m1))
    name2 = os.path.basename(os.path.normpath(m2))
    
    out_dir = "02-analysis"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{name1}-{name2}.json")
    
    print(f"\nLoading model 1...")
    sd1 = get_state_dict(m1)
    params1 = extract_relevant_keys(sd1)
    print(f"Found {len(params1)} relevant parameters in model 1")
    
    print(f"\nLoading model 2...")
    sd2 = get_state_dict(m2)
    params2 = extract_relevant_keys(sd2)
    print(f"Found {len(params2)} relevant parameters in model 2")
    
    # Find common keys
    common_keys = sorted(set(params1.keys()) & set(params2.keys()))
    print(f"\nComparing {len(common_keys)} common parameters...\n")
    
    layer_diffs = {}
    abs_diffs = []
    rank_diffs = []
    
    for i, k in enumerate(common_keys, 1):
        print(f"[{i}/{len(common_keys)}] {k}")
        
        v1 = params1[k]
        v2 = params2[k]
        
        # Compute L2 Frobenius norms using NumPy
        n1 = frobenius_norm_numpy(v1)
        n2 = frobenius_norm_numpy(v2)
        diff = abs(n1 - n2)
        
        # Compute matrix ranks
        # r1 = compute_rank(v1)
        # r2 = compute_rank(v2)
        r1, r2 = 0, 0
        rank_diff = abs(r1 - r2)

        layer_diffs[k] = {
            "model1": float(n1), 
            "model2": float(n2), 
            "abs_diff": float(diff),
            "rank_model1": int(r1),
            "rank_model2": int(r2),
            "rank_abs_diff": int(rank_diff)
        }
        abs_diffs.append(diff)
        rank_diffs.append(rank_diff)
    
    mean_diff = float(np.mean(abs_diffs)) if abs_diffs else 0.0
    mean_rank_diff = float(np.mean(rank_diffs)) if rank_diffs else 0.0
    
    results = {
        "model1": name1,
        "model2": name2,
        "mean_abs_diff_L2_Frobenius_norm": mean_diff,
        "mean_abs_diff_rank": mean_rank_diff,
        "num_compared_layers": len(common_keys),
        "per_layer_diffs": layer_diffs
    }
    
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Saved result to {out_path}")
    print(f"Mean abs difference in L2 Frobenius norms: {mean_diff:.6f}")
    print(f"Mean abs difference in rank: {mean_rank_diff:.6f}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()