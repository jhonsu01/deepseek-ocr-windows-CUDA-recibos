import os
import torch
import time
from transformers import AutoModel, AutoTokenizer

# ======================================================
# CONFIGURACIÓN DE DISPOSITIVO
# ======================================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Usando {device.upper()} device")

# ======================================================
# RUTAS LOCALES (usa \\ en Windows) Editala
# ======================================================
model_path = 'D:\\IA\\DeepSeek-OCR'
image_file = 'D:\\IA\\DeepSeek-OCR\\descargas\\img20251031_23101209.jpg'
output_path = 'output_dir'

# ======================================================
# CARGA DE MODELO Y TOKENIZER
# ======================================================
print("Cargando tokenizer y modelo...")
load_start = time.time()

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
    local_files_only=True
)

model = AutoModel.from_pretrained(
    model_path,
    trust_remote_code=True,
    local_files_only=True,
    use_safetensors=True
).to(device).eval()

load_end = time.time()
print(f"✅ Modelo cargado en {load_end - load_start:.2f} segundos")

# ======================================================
# CONFIGURACIÓN DE OCR
# ======================================================
prompt = "<image>\nFree OCR."

print(f"\n{'='*50}")
print(f"Iniciando OCR...")
print(f"{'='*50}")
print(f"Imagen: {image_file}")
print(f"Device: {device}")
print("-" * 50)

# ======================================================
# INFERENCIA
# ======================================================
inference_start = time.time()

with torch.no_grad():
    res = model.infer(
        tokenizer,
        prompt=prompt,
        image_file=image_file,
        output_path=output_path,
        base_size=1024,
        image_size=640,
        crop_mode=True,
        save_results=True,
        test_compress=False
    )

inference_end = time.time()
total_time = (load_end - load_start) + (inference_end - inference_start)

# ======================================================
# RESULTADOS
# ======================================================
print("\n" + "="*50)
print("✅ OCR COMPLETADO!")
print("="*50)
print(f"⏱️  Tiempo carga: {load_end - load_start:.2f} seg")
print(f"⚡ Tiempo inferencia: {inference_end - inference_start:.2f} seg")
print(f"📊 Total: {total_time:.2f} seg")
print(f"\n📁 Output en: {output_path}\\")
print(f"   - result.mmd (Markdown)")
print(f"   - result_with_boxes.jpg (Imagen anotada)")
print("="*50)
print(res)
