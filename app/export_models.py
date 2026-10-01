from __future__ import annotations

import json
import sys
from pathlib import Path

from app.ocr import DET_MODEL, MODEL_DIR, REC_MODEL


def model_config(cfg: dict) -> dict:
    post = cfg["PostProcess"]
    ops = {k: v for op in cfg["PreProcess"]["transform_ops"] for k, v in op.items()}
    if post["name"] == "DBPostProcess":
        assert not post.get("use_dilation") and post.get("score_mode", "fast") == "fast" \
            and post.get("box_type", "quad") == "quad", post
        norm = ops["NormalizeImage"]
        return {"type": "det", "unclip_ratio": post["unclip_ratio"], "max_candidates": post.get("max_candidates", 1000),
                "mean": norm["mean"], "std": norm["std"], "scale": eval(str(norm["scale"]))}
    if post["name"] == "CTCLabelDecode":
        return {"type": "rec", "image_shape": ops["RecResizeImg"]["image_shape"], "characters": post["character_dict"]}
    raise ValueError(f"desteklenmeyen model: {post['name']}")


def export(name: str, out: Path) -> Path:
    import paddle2onnx
    import yaml
    from paddlex.inference.utils.official_models import official_models

    src = Path(official_models[name])
    dst = out / name
    dst.mkdir(parents=True, exist_ok=True)
    cfg = model_config(yaml.safe_load((src / "inference.yml").read_text(encoding="utf-8")))
    tmp = dst / "inference.onnx.tmp"
    paddle2onnx.export(str(src / "inference.json"), str(src / "inference.pdiparams"), str(tmp), opset_version=14)
    tmp.replace(dst / "inference.onnx")
    (dst / "config.json").write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
    return dst


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else MODEL_DIR
    for m in sys.argv[2:] or [DET_MODEL, REC_MODEL]:
        print("hazır:", export(m, out))
