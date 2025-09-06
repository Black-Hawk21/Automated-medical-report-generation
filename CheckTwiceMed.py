import os
import math
import json
import time
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
)

# open_clip is the reference loader for SigLIP/BiomedCLIP here
from open_clip import create_model_and_transforms


# -----------------------------
# Utilities
# -----------------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    #os.environ["PYTHONHASHSEED"] = str(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_image(path: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img


def batched(iterable, n=8):
    """Yield successive n-sized chunks from iterable."""
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]


# -----------------------------
# Model wrappers
# -----------------------------
@dataclass
class VisionBackbone:
    name: str
    model: nn.Module
    preprocess: object
    embed_dim: int


class Projector(nn.Module):
    """Simple projector from vision embedding -> LLM hidden dim.
    Replace with a 2-3 layer MLP if you want more capacity."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_dim] -> [B, out_dim]
        return self.net(x)


class CrossCheckMed:
    """
    Two encoders (SigLIP, BiomedCLIP) feed a shared LLM (Meditron) via
    independent projectors. Provides:
      - generate_report(image, chain='siglip'|'biomed')
      - crosscheck(image, max_iter=5)
    """

    def __init__(
        self,
        siglip_ckpt: str = "hf-hub:timm/ViT-B-16-SigLIP",
        biomed_ckpt: str = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        llm_ckpt: str = "microsoft/BioGPT",  # Use BioGPT for medical text generation
        visual_tokens: int = 32,
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
    ):
        self.device = device or get_device()
        self.dtype = dtype
        self.visual_tokens = visual_tokens

        # ---- Load vision encoders
        self.siglip = self._load_openclip_model(siglip_ckpt, "siglip")
        self.biomed = self._load_openclip_model(biomed_ckpt, "biomed")

        # Freeze vision encoders initially
        for p in self.siglip.model.parameters():
            p.requires_grad = False
        for p in self.biomed.model.parameters():
            p.requires_grad = False

        # ---- Load LLM (text-only model with custom vision encoders)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_ckpt, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_ckpt,
            torch_dtype=self.dtype,
            device_map="auto" if self.device.type == "cuda" else None,
        ).to(self.device)
        
        print(f"[Info] Loaded {llm_ckpt} as text-only model with custom vision encoders")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Get hidden size from LLM config (different attribute names for different models)
        if hasattr(self.llm.config, 'hidden_size'):
            self.hidden_size = self.llm.config.hidden_size
        elif hasattr(self.llm.config, 'text_config') and hasattr(self.llm.config.text_config, 'hidden_size'):
            self.hidden_size = self.llm.config.text_config.hidden_size
        elif hasattr(self.llm.config, 'd_model'):
            self.hidden_size = self.llm.config.d_model
        else:
            # Fallback - try to get from the language model component
            try:
                if hasattr(self.llm, 'language_model'):
                    self.hidden_size = self.llm.language_model.config.hidden_size
                else:
                    self.hidden_size = 4096  # Default fallback
                    print(f"[Warning] Could not determine hidden_size, using default: {self.hidden_size}")
            except:
                self.hidden_size = 4096  # Default fallback
                print(f"[Warning] Could not determine hidden_size, using default: {self.hidden_size}")

        # ---- Projectors
        self.proj_siglip = Projector(self.siglip.embed_dim, self.hidden_size).to(self.device, dtype=self.dtype)
        self.proj_biomed = Projector(self.biomed.embed_dim, self.hidden_size).to(self.device, dtype=self.dtype)

        # Put models in eval mode by default (switch to train() if you fine-tune)
        self.llm.eval()
        self.siglip.model.eval()
        self.biomed.model.eval()
        self.proj_siglip.eval()
        self.proj_biomed.eval()

    def _load_openclip_model(self, ckpt: str, name: str) -> VisionBackbone:
        model, _, preprocess = create_model_and_transforms(ckpt, pretrained=True, device=self.device)
        # Try to infer embedding dimension for encode_image
        # First try to get a dummy image and see what dimension we get
        try:
            with torch.no_grad():
                dummy_img = torch.randn(1, 3, 224, 224).to(self.device)
                dummy_emb = model.encode_image(dummy_img)
                if dummy_emb.ndim == 3:
                    embed_dim = dummy_emb.shape[-1]
                else:
                    embed_dim = dummy_emb.shape[-1]
        except Exception:
            # Fallback: try different attributes
            try:
                embed_dim = model.visual.output_dim
            except Exception:
                embed_dim = getattr(model, "embed_dim", 768)  # Default to 768 for SigLIP
        return VisionBackbone(name=name, model=model, preprocess=preprocess, embed_dim=embed_dim)

    # -----------------------------
    # Encoding & token packing
    # -----------------------------
    @torch.no_grad()
    def _encode_image(self, img: Image.Image, chain: str) -> torch.Tensor:
        """
        Returns a [1, embed_dim] image embedding from the chosen chain.
        """
        if chain == "siglip":
            proc = self.siglip.preprocess(img).unsqueeze(0).to(self.device)
            emb = self.siglip.model.encode_image(proc)
            if emb.ndim == 3:
                emb = emb.mean(dim=1)
            return emb.to(self.dtype)
        elif chain == "biomed":
            proc = self.biomed.preprocess(img).unsqueeze(0).to(self.device)
            emb = self.biomed.model.encode_image(proc)
            if emb.ndim == 3:
                emb = emb.mean(dim=1)
            return emb.to(self.dtype)
        else:
            raise ValueError("chain must be 'siglip' or 'biomed'")

    def _project_to_llm(self, emb: torch.Tensor, chain: str) -> torch.Tensor:
        """
        Maps [B, embed_dim] -> [B, hidden_size] using the appropriate projector.
        """
        if chain == "siglip":
            return self.proj_siglip(emb)
        return self.proj_biomed(emb)

    def _expand_visual_tokens(self, z: torch.Tensor, n_tokens: int) -> torch.Tensor:
        """
        Turns [B, hidden] into [B, n_tokens, hidden] by repetition.
        Replace with a learned mapping (e.g., MLP->(N x hidden)) if desired.
        """
        b, h = z.shape
        return z.unsqueeze(1).repeat(1, n_tokens, 1)

    # -----------------------------
    # Generation
    # -----------------------------
    @torch.no_grad()
    def generate_report(
        self,
        img: Image.Image,
        chain: str = "siglip",
        system_prompt: str = (
            "Generate a chest X-ray radiology report with clinical findings.\n"
            "FINDINGS: "
        ),
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> str:
        """
        Generates text using custom visual embeddings from SigLIP/BiomedCLIP.
        """
        # 1) Encode image to get visual features
        emb = self._encode_image(img, chain=chain)       # [1, D]
        h = self._project_to_llm(emb, chain=chain)       # [1, hidden]
        vtoks = self._expand_visual_tokens(h, self.visual_tokens)  # [1, N, hidden]
        
        # 2) Create a visual context description based on the chain used
        if chain == "siglip":
            visual_context = "Based on SigLIP vision analysis: "
        else:
            visual_context = "Based on BiomedCLIP medical analysis: "
        
        # 3) Enhanced prompt with visual context
        enhanced_prompt = visual_context + system_prompt
        
        # 4) Tokenize and get text embeddings
        inputs = self.tokenizer(enhanced_prompt, return_tensors="pt").to(self.device)
        text_emb = self.llm.get_input_embeddings()(inputs["input_ids"])  # [1, T, hidden]
        
        # 5) Concat visual + text embeddings for multimodal understanding
        full_emb = torch.cat([vtoks, text_emb], dim=1)  # [1, N+T, hidden]
        
        out = self.llm.generate(
            inputs_embeds=full_emb,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0.0,
            temperature=temperature if temperature > 0.0 else None,
            top_p=top_p if temperature > 0.0 else None,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            repetition_penalty=1.2,
        )
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)

        # Strip the original prompt if the model echoes it
        if text.startswith(enhanced_prompt):
            text = text[len(enhanced_prompt):].lstrip()
        return text

    # -----------------------------
    # Claim extraction & verification (placeholders you can improve)
    # -----------------------------
    @staticmethod
    def _extract_impression(text: str) -> str:
        """Very rough splitter; replace with a structured parser or RadGraph."""
        key = "IMPRESSION"
        idx = text.upper().find(key)
        if idx == -1:
            return text.strip()
        return text[idx + len(key):].strip(": \n\t")

    @staticmethod
    def _simple_jaccard(a: str, b: str) -> float:
        """Token-level Jaccard on impressions as a crude convergence proxy."""
        A = set(a.lower().split())
        B = set(b.lower().split())
        if not A and not B:
            return 1.0
        return len(A & B) / max(1, len(A | B))

    def _verify_claims_placeholder(
        self,
        img: Image.Image,
        peer_report: str,
        verifier_chain: str = "biomed",
    ) -> Dict:
        """
        Placeholder verification that you should replace with:
          - claim extraction (RadGraph or regex)
          - similarity via CLIP text/image scoring
          - Grad-CAM/attention rollout heatmaps
          - LLM entailment (Agree/Disagree/Uncertain)
        For now it returns a dummy dict with the peer impression and a fake score.
        """
        imp = self._extract_impression(peer_report)
        # crude "support" score: higher if impression is short & concrete (lol)
        score = 1.0 / (1.0 + math.log2(2 + len(imp.split())))
        return {
            "verifier": verifier_chain,
            "peer_impression": imp,
            "support_score": float(score),
            "evidence": None,  # place for heatmap paths / bbox JSON
        }

    # -----------------------------
    # Cross-swap loop
    # -----------------------------
    @torch.no_grad()
    def crosscheck(
        self,
        img: Image.Image,
        max_iter: int = 5,
        jaccard_thresh: float = 0.85,
    ) -> Dict:
        """
        Run the iterative cross-verification between the two chains.
        Returns either a converged report or both reports with transparent evidence.
        """
        history = []

        # Initial drafts
        rep_a = self.generate_report(img, chain="siglip")
        rep_b = self.generate_report(img, chain="biomed")

        for it in range(max_iter):
            ver_b_on_a = self._verify_claims_placeholder(img, rep_a, verifier_chain="biomed")
            ver_a_on_b = self._verify_claims_placeholder(img, rep_b, verifier_chain="siglip")

            imp_a = self._extract_impression(rep_a)
            imp_b = self._extract_impression(rep_b)
            jac = self._simple_jaccard(imp_a, imp_b)

            history.append({
                "iter": it,
                "impression_a": imp_a,
                "impression_b": imp_b,
                "jaccard": jac,
                "verify_b_on_a": ver_b_on_a,
                "verify_a_on_b": ver_a_on_b,
            })

            if jac >= jaccard_thresh:
                return {
                    "status": "converged",
                    "iterations": it + 1,
                    "agreed_report": rep_b,  # or merge(rep_a, rep_b)
                    "history": history,
                }

            # Simple refinement heuristic:
            # prepend feedback summary to each chain's next prompt (you can do something smarter)
            fb_a = f"Consider this peer analysis: '{imp_b}'. Now generate an improved chest X-ray radiology report.\nFINDINGS: "
            fb_b = f"Consider this peer analysis: '{imp_a}'. Now generate an improved chest X-ray radiology report.\nFINDINGS: "

            rep_a = self.generate_report(
                img,
                chain="siglip",
                system_prompt=fb_b,
            )
            rep_b = self.generate_report(
                img,
                chain="biomed",
                system_prompt=fb_a,
            )

        # Not converged within max_iter
        return {
            "status": "disagreed",
            "iterations": max_iter,
            "report_siglip": rep_a,
            "report_biomed": rep_b,
            "history": history,
        }


# -----------------------------
# Example CLI usage
# -----------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="CrossCheckMed demo")
    parser.add_argument("--image", required=True, help="Path to input medical image (e.g., chest x-ray)")
    parser.add_argument("--iters", type=int, default=3, help="Max cross-check iterations")
    parser.add_argument("--vtoks", type=int, default=32, help="Number of visual tokens injected")
    parser.add_argument("--temp", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--jaccard", type=float, default=0.85, help="Jaccard threshold for convergence")
    args = parser.parse_args()

    seed_everything(123)
    device = get_device()
    print(f"[Info] Using device: {device}")

    img = load_image(args.image)

    engine = CrossCheckMed(
        visual_tokens=args.vtoks,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        device=device,
    )

    # One-shot generations
    print("\n=== SigLIP chain draft ===")
    rep_a = engine.generate_report(img, chain="siglip", temperature=args.temp)
    print(rep_a)

    print("\n=== BiomedCLIP chain draft ===")
    rep_b = engine.generate_report(img, chain="biomed", temperature=args.temp)
    print(rep_b)

    # Cross-check
    print("\n=== Cross-check ===")
    result = engine.crosscheck(img, max_iter=args.iters, jaccard_thresh=args.jaccard)
    print(json.dumps(result if result["status"] != "converged" else {
        "status": result["status"],
        "iterations": result["iterations"],
        "agreed_report": result["agreed_report"],
    }, indent=2))


if __name__ == "__main__":
    main()