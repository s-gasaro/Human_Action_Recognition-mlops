"""Convert models/har_model.h5 to models/har_model.tflite:  python scripts/convert_tflite.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tflite_convert import convert

out = convert(ROOT / "models" / "har_model.h5", ROOT / "models" / "har_model.tflite")
print(f"Saved {out} ({out.stat().st_size / 1e6:.1f} MB)")
