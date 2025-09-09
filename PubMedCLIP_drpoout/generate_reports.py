
import json
import os
import glob
import torch
import _init_paths
from config import cfg, update_config
import clip
from PIL import Image


def load_or_default_config(module_root: str):
    cfg_path = os.path.join(module_root, "configs", "medclip_roco_all.yaml")
    if os.path.exists(cfg_path):
        class _A: pass
        a = _A(); a.cfg = cfg_path
        update_config(cfg, a)
    else:
        cfg.defrost()
        cfg.NAME = "MEDCLIP.ROCO.50epoch"
        cfg.OUTPUT_DIR = os.path.join(module_root, "output", "medclip", "ROCO")
        cfg.DATASET.DATA_DIR = os.path.join(module_root, "data")
        cfg.TRAIN.VISION_ENCODER = "ViT-B/32"
        cfg.TRAIN.MAX_SEQ_LENGTH = 77
        cfg.freeze()


def find_checkpoint(module_root: str) -> str:
    ckpt = os.path.join(cfg.OUTPUT_DIR, cfg.NAME, "models", "best_model.pth")
    if os.path.exists(ckpt):
        return ckpt
    candidates = glob.glob(os.path.join(module_root, "output", "**", "models", "best_model.pth"), recursive=True)
    return candidates[-1] if candidates else ckpt


def read_test_records(module_root: str):
    test_json = os.path.join(module_root, "data", "test_dataset.json")
    if not os.path.exists(test_json):
        raise SystemExit(f"Missing {test_json}. Generate it with main/create_jsons.py.")
    records = []
    with open(test_json, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line)
            if "caption" in obj: records.append(obj)
    if not records:
        raise SystemExit("No records in test_dataset.json")
    return records, test_json


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    module_root = os.path.normpath(os.path.join(here, ".."))

    load_or_default_config(module_root)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt_path = find_checkpoint(module_root)
    if not os.path.exists(ckpt_path):
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")
    print(f"Checkpoint: {ckpt_path}")
    ckpt_obj = torch.load(ckpt_path, map_location=str(device))
    # Extract possible state dict
    state = None
    if isinstance(ckpt_obj, dict):
        for k in ("state_dict", "model", "model_state_dict"):
            if k in ckpt_obj and isinstance(ckpt_obj[k], dict):
                state = ckpt_obj[k]
                break
        if state is None:
            # might already be a plain state dict
            state = ckpt_obj
    else:
        state = ckpt_obj

    if not isinstance(state, dict):
        raise SystemExit("Checkpoint does not contain a valid state_dict")

    # Normalize common wrappers/prefixes
    def _norm_keys(sd: dict):
        out = {}
        for k, v in sd.items():
            nk = k
            if nk.startswith("module."):
                nk = nk[len("module."):]
            if nk.startswith("clip."):
                nk = nk[len("clip."):]
            if nk.startswith("clip_model."):
                nk = nk[len("clip_model."):]
            if nk.startswith("model."):
                nk = nk[len("model."):]
            out[nk] = v
        return out

    state = _norm_keys(state)

    # Auto-detect encoder family from keys and try candidates
    keys = state.keys()
    has_transformer = any("visual.transformer" in k for k in keys)
    cfg_enc = getattr(cfg.TRAIN, "VISION_ENCODER", "ViT-B/32")
    candidates = []
    if has_transformer:
        # ViT family
        if "ViT" in cfg_enc:
            candidates.append(cfg_enc)
        candidates += ["ViT-B/32", "ViT-B/16", "ViT-L/14"]
    else:
        # ResNet family
        if cfg_enc.startswith("RN"):
            candidates.append(cfg_enc)
        candidates += ["RN50", "RN50x4", "RN101"]
    # de-duplicate
    seen = set(); cand_list = []
    for c in candidates:
        if c not in seen:
            seen.add(c); cand_list.append(c)

    model = None; preprocess = None; loaded = False; last_err = None
    for enc in cand_list:
        try:
            print(f"Trying encoder: {enc}")
            model, preprocess = clip.load(enc, device=device, jit=False)
            model.eval()
            model.load_state_dict(state, strict=True)
            print(f"Loaded checkpoint with encoder: {enc}")
            loaded = True
            break
        except Exception as e:
            last_err = e
            continue
    if not loaded:
        print("Failed to load checkpoint with tried encoders:", cand_list)
        print(f"state_dict keys: {len(state.keys())}; sample: {list(state.keys())[:5]}")
        raise last_err if last_err else SystemExit("Could not load checkpoint")

    records, test_json = read_test_records(module_root)
    captions = [r["caption"].strip() for r in records if r.get("caption", "").strip()]

    # Build list of resolvable images (as-is or relative to module root)
    resolved = []
    skipped = 0
    for r in records:
        ip = (r.get("image_path") or "").strip()
        if not ip:
            skipped += 1
            continue
        if os.path.exists(ip):
            resolved.append((ip, r, "as-is"))
            continue
        cand1 = os.path.normpath(os.path.join(module_root, ip))
        if os.path.exists(cand1):
            resolved.append((cand1, r, "resolved relative to module root"))
        else:
            skipped += 1

    if not resolved:
        raise SystemExit("No valid image paths could be resolved from test_dataset.json")

    print(f"CAPTIONS: {len(captions)} from {test_json}")
    print(f"IMAGES: {len(resolved)} resolved, {skipped} skipped")

    @torch.no_grad()
    def enc_img(p):
        img = Image.open(p).convert("RGB")
        x = preprocess(img).unsqueeze(0).to(device)
        f = model.encode_image(x)
        return f / f.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def enc_txt(txts, bs=256, ctx_len=int(getattr(cfg.TRAIN, "MAX_SEQ_LENGTH", 77))):
        outs = []
        for i in range(0, len(txts), bs):
            toks = clip.tokenize(txts[i:i+bs], context_length=ctx_len, truncate=True).to(device)
            f = model.encode_text(toks)
            outs.append(f / f.norm(dim=-1, keepdim=True))
        return torch.cat(outs, 0)

    # Encode all captions once
    ft = enc_txt(captions)

    # Iterate over all resolved test images
    ok = 0
    for i, (img_path, rec, how) in enumerate(resolved, 1):
        try:
            fi = enc_img(img_path)
            sims = (fi @ ft.t()).squeeze(0).float()
            idx = int(torch.argmax(sims).item())
            pred = captions[idx]
            gt = rec.get("caption") or ""
            match = int(pred.strip() == gt.strip())
            ok += match
            print(f"[{i}/{len(resolved)}] IMAGE: {img_path} [{how}] | PRED: {pred[:120]} | GT: {gt[:120]} | match={bool(match)}")
        except Exception as e:
            print(f"[{i}/{len(resolved)}] IMAGE: {img_path} [{how}] | ERROR: {e}")

    print(f"\nSummary: processed={len(resolved)}, skipped={skipped}, top1_matches={ok}, accuracy={(ok/len(resolved))*100:.2f}%")


if __name__ == "__main__":
    main()

