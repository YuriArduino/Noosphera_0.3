#!/usr/bin/env python3
"""
Download automático dos modelos Tesseract:
- tessdata (padrão)
- tessdata_fast (rápido)
- tessdata_best (preciso)
"""

import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).parent

models = {
    "tessdata": {
        "url": "https://github.com/tesseract-ocr/tessdata/raw/main/",
        "langs": ["por.traineddata", "eng.traineddata"],
    },
    "tessdata_fast": {
        "url": "https://github.com/tesseract-ocr/tessdata_fast/raw/main/",
        "langs": ["por.traineddata", "eng.traineddata"],
    },
    "tessdata_best": {
        "url": "https://github.com/tesseract-ocr/tessdata_best/raw/main/",
        "langs": ["por.traineddata", "eng.traineddata"],
    },
}

print("⬇️  Baixando modelos Tesseract...\n")

for model_name, config in models.items():
    model_dir = BASE_DIR / model_name
    model_dir.mkdir(exist_ok=True)
    print(f"📦 {model_name}/")

    for lang_file in config["langs"]:
        dest = model_dir / lang_file
        if dest.exists():
            print(
                f"   ✅ {lang_file} já existe ({dest.stat().st_size / 1024 / 1024:.1f} MB)"
            )
        else:
            url = config["url"] + lang_file
            print(f"   ⬇️  Baixando {lang_file}...", end=" ")
            try:
                subprocess.run(
                    ["wget", "-q", "-O", str(dest), url],
                    check=True,
                    capture_output=True,
                )
                size_mb = dest.stat().st_size / 1024 / 1024
                print(f"✅ ({size_mb:.1f} MB)")
            except Exception as e:
                print(f"❌ Erro: {e}")

print("\n✅ Todos os modelos baixados!")
print("\nTamanhos esperados:")
print("  • tessdata_fast/por.traineddata: ~2.5 MB")
print("  • tessdata/por.traineddata: ~5.0 MB")
print("  • tessdata_best/por.traineddata: ~25 MB")
