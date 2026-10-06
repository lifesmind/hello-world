"""Load a pinned frozen encoder in Colab; synthetic smoke test only.

This does not train a classifier, score research images, or verify old results.
Authentication is read from Colab Secrets and is never logged or saved.
"""
from pathlib import Path
from datetime import datetime, timezone
from importlib.metadata import version
import gc
import hashlib
import json
import shutil
import sys

REPO_ID = "google/medsiglip-448"
REVISION = "9cea28a1a1195f665105faa6e8544c112fd960a4"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model():
    import torch
    from PIL import Image
    from google.colab import drive, userdata
    from huggingface_hub import snapshot_download
    from transformers import AutoImageProcessor, AutoModel

    if not torch.cuda.is_available():
        raise RuntimeError("ไม่พบ GPU: เลือก T4 แล้วเชื่อมต่อ Colab")
    drive.mount("/content/drive", force_remount=False)
    root = Path("/content/drive/MyDrive/MAKE_Reanalysis")
    model_dir = root / "models" / ("medsiglip-448_" + REVISION)
    print("1/4 กำลังดาวน์โหลดไฟล์โมเดลลง Drive", flush=True)
    token = userdata.get("HF_TOKEN").strip()
    try:
        snapshot_download(
            repo_id=REPO_ID, revision=REVISION, token=token,
            local_dir=str(model_dir), max_workers=2,
            allow_patterns=["config.json", "preprocessor_config.json",
                            "model*.safetensors", "model*.safetensors.index.json"]
        )
    finally:
        del token

    print("2/4 กำลังโหลดโมเดลบน GPU", flush=True)
    processor = AutoImageProcessor.from_pretrained(
        str(model_dir), local_files_only=True, backend="pil"
    )
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = AutoModel.from_pretrained(
        str(model_dir), local_files_only=True, use_safetensors=True,
        dtype=torch.float32, attn_implementation="sdpa"
    ).to("cuda").eval()
    model.requires_grad_(False)

    print("3/4 กำลังทดสอบด้วยภาพสีเทาที่สร้างขึ้น", flush=True)
    image = Image.new("RGB", (448, 448), (128, 128, 128))
    inputs = processor(images=image, return_tensors="pt").to("cuda")
    assert tuple(inputs["pixel_values"].shape) == (1, 3, 448, 448)
    with torch.inference_mode():
        output = model.get_image_features(**inputs, return_dict=True)
        pooled = output.pooler_output
        assert tuple(pooled.shape) == (1, 1152), "ขนาดเวกเตอร์ไม่ตรง"
        assert torch.isfinite(pooled).all().item(), "พบค่าที่ไม่เป็นจำนวนจำกัด"
        assert (pooled.norm(dim=1) > 0).all().item(), "พบเวกเตอร์ศูนย์"
        normalized = torch.nn.functional.normalize(pooled.float(), p=2, dim=1)
        norm = normalized.norm(dim=1).item()
        assert abs(norm - 1.0) < 1e-5, "ตรวจ L2 normalization ไม่ผ่าน"

    print("4/4 กำลังบันทึกหลักฐานเวอร์ชันและ SHA-256", flush=True)
    source_files = sorted(p for p in model_dir.iterdir()
                          if p.is_file() and p.suffix in (".json", ".safetensors"))
    assert any(p.suffix == ".safetensors" for p in source_files)
    source_script = Path(__file__).resolve()
    code_dir = root / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    saved_code = code_dir / source_script.name
    if saved_code.exists() and sha256_file(saved_code) != sha256_file(source_script):
        raise RuntimeError("ไฟล์โค้ดชื่อเดิมมีเนื้อหาต่างกัน: หยุดเพื่อรักษาหลักฐาน")
    if not saved_code.exists():
        shutil.copyfile(source_script, saved_code)
    audit = {
        "stage": "pinned_encoder_load_and_synthetic_smoke_test",
        "research_images_processed": 0, "historical_results_verified": False,
        "repo_id": REPO_ID, "revision": REVISION, "model_directory": str(model_dir),
        "model_class": type(model).__name__, "weights_frozen": True,
        "dtype": "float32", "attention_implementation": "sdpa",
        "image_processor_class": type(processor).__name__,
        "image_processor_backend": getattr(processor, "backend", None),
        "image_processor_config": json.loads(processor.to_json_string()),
        "feature_method": "get_image_features.pooler_output_then_L2_normalize",
        "feature_shape": [1, 1152], "normalized_vector_norm": norm,
        "synthetic_image": {"mode": "RGB", "size": [448, 448], "color": [128, 128, 128]},
        "environment": {
            "python": sys.version, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "packages": {name: version(name) for name in
                         ("torch", "transformers", "huggingface_hub", "Pillow", "safetensors")}
        },
        "source_files": {p.name: {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
                         for p in source_files},
        "code_file": str(saved_code), "code_sha256": sha256_file(saved_code),
        "created_utc": datetime.now(timezone.utc).isoformat()
    }
    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    audit_path = log_dir / ("medsiglip_load_audit_" + stamp + ".json")
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("MODEL LOAD: PASS")
    print("Feature shape: (1, 1152)")
    print("L2 norm:", round(norm, 6))
    print("Model revision:", REVISION)
    print("Saved model, code and audit to Drive")
    return model, processor, audit


if __name__ == "__main__":
    MEDSIGLIP_MODEL, MEDSIGLIP_PROCESSOR, MEDSIGLIP_INFO = load_model()
