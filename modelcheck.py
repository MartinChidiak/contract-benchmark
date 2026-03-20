# check_vram.py — corré esto dentro del contenedor antes de bajar el modelo
import subprocess
import json

def get_free_vram_gb():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True, text=True
    )
    free_mb, total_mb = map(int, result.stdout.strip().split(", "))
    return free_mb / 1024, total_mb / 1024

free, total = get_free_vram_gb()
print(f"VRAM total:     {total:.1f} GB")
print(f"VRAM libre:     {free:.1f} GB")
print()

models = {
    "Llama 3.1 8B  (bfloat16)":  16.0,
    "Mistral Nemo 12B (bfloat16)": 24.0,
    "Qwen2.5 14B AWQ (INT4)":     9.0,
    "Qwen2.5 32B AWQ (INT4)":     18.0,
}

KV_CACHE_MIN_GB = 4.0  # mínimo razonable para contratos largos

print(f"{'Modelo':<35} {'Modelo':>8} {'+ KV min':>10} {'Entra?':>8}")
print("-" * 65)
for name, size in models.items():
    total_needed = size + KV_CACHE_MIN_GB
    fits = "✅ SÍ" if total_needed <= free else "❌ NO"
    print(f"{name:<35} {size:>6.1f}GB {total_needed:>8.1f}GB {fits:>8}")
