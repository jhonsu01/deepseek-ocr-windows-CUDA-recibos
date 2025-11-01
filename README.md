
# DeepSeek-OCR en Windows con CUDA: OCR para Recibos

Script simple ocr.py para leer recibos con DeepSeek-OCR local en windows Usando CUDA. Basado en tutorial YouTube de Tech With Mary.

## Instalación Rápida
1. Instala Miniconda Windows : https://docs.conda.io/en/latest/miniconda.html (elige PowerShell).
2. Clona DeepSeek-OCR: `git clone https://huggingface.co/deepseek-ai/DeepSeek-OCR`
3. En la carpeta DeepSeek-OCR: `python -m venv .venv`
   `.venv\Scripts\activate`
5. Instala deps: `pip install torch==2.6.0 transformers==4.46.3 tokenizers==0.20.3 einops addict easydict`
   `pip install torchvision Pillow`
6. Instala CUDA: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124`
7. Verifica que funcione: `python -c "import torch; print(torch.cuda.is_available())"`
   Si devuelve True, podrás correr el modelo con GPU sin cambios.
8. Copia ocr.py a la carpeta, cambia image_file a tu recibo, y corre `python ocr.py`.

## Uso
- Prompt: "<image>\nFree OCR." para simple.
- image_size=640 para detalle en números.
- test_compress=False para precisión.
- Output en output_dir/result.mmd (texto extraído).

Ver video para fixes MPS y tests con arrugada. ¡Prueba y comenta tu total!

Licencia: MIT – usa libre.

