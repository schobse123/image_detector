import argparse
import os
from pathlib import Path
import tempfile

import tensorflow as tf


def _set_if_present(obj, attr: str, value) -> bool:
  if hasattr(obj, attr):
    setattr(obj, attr, value)
    return True
  return False


def _load_model_any(model_path: Path):
  # Prefer standalone Keras (v3) to load `.keras` models.
  # Fallback to tf.keras for older/legacy models.
  load_errors: list[str] = []

  try:
    import keras  # type: ignore

    model = keras.saving.load_model(str(model_path), compile=False)
    return model, "keras"
  except Exception as e:  # noqa: BLE001
    load_errors.append(f"keras.saving.load_model failed: {e}")

  try:
    model = tf.keras.models.load_model(str(model_path), compile=False)
    return model, "tf.keras"
  except Exception as e:  # noqa: BLE001
    load_errors.append(f"tf.keras.models.load_model failed: {e}")

  msg = "\n".join(load_errors)
  raise RuntimeError(f"Failed to load model at {model_path}.\n{msg}")


def convert(model_path: Path, out_path: Path, *, allow_select_tf_ops: bool, force_legacy_converter: bool) -> None:
  print(f"TensorFlow: {tf.__version__}")
  print(f"Loading model: {model_path}")

  model, loaded_via = _load_model_any(model_path)
  print(f"Loaded via: {loaded_via}")

  # Export to SavedModel first. This path works for both Keras v3 models and tf.keras models,
  # and typically gives the TFLite converter a cleaner graph.
  with tempfile.TemporaryDirectory(prefix="tflite_export_") as tmpdir:
    saved_model_dir = Path(tmpdir) / "saved_model"
    print("Exporting to SavedModel...")
    if hasattr(model, "export"):
      # Keras v3
      model.export(str(saved_model_dir))
    else:
      # tf.keras
      tf.saved_model.save(model, str(saved_model_dir))

    print("Creating TFLite converter (SavedModel path)...")
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))

    # Prefer pure TFLite builtins; enable Select TF Ops only if required.
    supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    if allow_select_tf_ops:
      supported_ops.append(tf.lite.OpsSet.SELECT_TF_OPS)
    converter.target_spec.supported_ops = supported_ops

    # Workarounds for TensorFlow builds that crash inside the MLIR converter.
    if force_legacy_converter:
      changed = False
      changed |= _set_if_present(converter, "experimental_new_converter", False)
      changed |= _set_if_present(converter, "_experimental_disable_converter_mlir", True)
      if changed:
        print("Enabled legacy / non-MLIR converter flags.")

    print("Converting...")
    tflite_model = converter.convert()

  out_path.parent.mkdir(parents=True, exist_ok=True)
  out_path.write_bytes(tflite_model)
  print(f"\n✅ Success! Wrote: {out_path}")


def main() -> None:
  parser = argparse.ArgumentParser(description="Convert a Keras model to TFLite.")
  parser.add_argument("--in", dest="in_path", default="trained_cnn_model.keras", help="Input .keras/.h5 model path")
  parser.add_argument("--out", dest="out_path", default="model.tflite", help="Output .tflite path")
  parser.add_argument(
    "--select-tf-ops",
    action="store_true",
    help="Allow Select TF Ops (larger binary; requires tflite-runtime with Flex delegate)",
  )
  parser.add_argument(
    "--legacy-converter",
    action="store_true",
    help="Try legacy/non-MLIR converter flags (often avoids MLIR aborts)",
  )
  args = parser.parse_args()

  convert(
    Path(args.in_path),
    Path(args.out_path),
    allow_select_tf_ops=bool(args.select_tf_ops),
    force_legacy_converter=bool(args.legacy_converter),
  )


if __name__ == "__main__":
  # Reduce TF log noise (optional)
  os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
  main()