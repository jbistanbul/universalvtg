#!/usr/bin/env python
"""
Embed all sentences from ego4d_egovlp.json with text features and save as <sentence_id>.npy

Supports multiple text encoders:
  - evaclip: BAAI/EVA-CLIP-8B-448 (token-level features)
  - clip: openai/clip-vit-large-patch14-336 (pooled or token-level)
  - pe: PE-Core-L14-336 (PerceptionEncoder, pooled or token-level)

What it saves:
  - token-level mode: [T_valid, H] per sentence (no padding)
  - pooled mode: [1, H] per sentence (single vector with leading dim for consistency)
  - poolandtoken mode: [1 + T_valid, H] per sentence (pooled vector prepended to token features)

Defaults:
  json_path     = ./ego4d_egovlp.json
  output_dir    = ./text_feats (will be suffixed with model type)
  model_type    = clip
  batch_size    = 256
  save_dtype    = float16
  skip_existing = True
  splits        = (all top-level keys)
  feature_type  = pooled (for clip/pe), token (for evaclip), poolandtoken

Example usage:
  # Extract CLIP text features (pooled)
  python extract_text_features.py --model_type clip --output_dir ./model_features/clip-vit-l14-336/text_pooled
  python extract_text_features.py --model_type clip --feature_type token --output_dir ./model_features/clip-vit-l14-336/text_token

  # Extract PerceptionEncoder text features (pooled)
  python extract_text_features.py --model_type pe --output_dir ./pe-core-l14-336/text_pooled
  python extract_text_features.py --model_type pe --feature_type token --output_dir ./pe-core-l14-336/text_token
  
  python extract_text_features.py --json_path vtg_annotations/charades_sta_processed_v2.json --model_type pe --feature_type token --output_dir ./datasets/charades-sta/text_token
  python extract_text_features.py --json_path vtg_annotations/tacos_processed_v2.json --model_type pe --feature_type token --output_dir ./datasets/TACoS/pe-core-l14-336/text_token
  python extract_text_features.py --json_path vtg_annotations/coin_vtg_format_available.json --model_type pe --feature_type token --output_dir ./datasets/COIN/pe-core-l14-336/text_token
  
  # poolandtoken
  python feature_extraction/extract_text_features.py --json_path data/datasets/Ego4D-GoalStep/pe-core-l14-336/goalstep_converted_vtg_converted.json --model_type pe --feature_type poolandtoken --output_dir data/datasets/Ego4D-GoalStep/pe-core-l14-336/text_poolandtoken_converted
  python feature_extraction/extract_text_features.py --json_path data/datasets/ego4d_nlq_v2/pe-core-l14-336/converted/nlq_v2_vtg_converted_final.json --model_type pe --feature_type poolandtoken --output_dir data/datasets/ego4d_nlq_v2/pe-core-l14-336/converted/text_poolandtoken
  # Extract EVA-CLIP text features (token-level)
  python extract_text_features.py --model_type evaclip --output_dir ./evaclip_text_feats
"""

import os
import json
import argparse
from typing import Dict, List, Optional, Tuple

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PERCEPTION_MODELS_DIR = REPO_ROOT / "perception_models"
if PERCEPTION_MODELS_DIR.exists():
    sys.path.insert(0, str(PERCEPTION_MODELS_DIR))


import numpy as np
import torch
from tqdm import tqdm

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json_path", type=str, default="data/datasets/Ego4D-GoalStep/pe-core-l14-336/goalstep_converted_vtg_converted.json",
                    help="Path to annotation json file")
    ap.add_argument("--output_dir", type=str, default=None,
                    help="Directory to save <sentence_id>.npy files. Defaults to ./text_features/<model_type>/<feature_type> if omitted.")
    ap.add_argument("--model_type", type=str, choices=["evaclip", "clip", "pe"], default="pe",
                    help="Which text encoder to use: evaclip, clip, or pe")
    ap.add_argument("--model_name", type=str, default=None,
                    help="Override model name/path (uses defaults per model_type if not set)")
    ap.add_argument("--feature_type", type=str, choices=["pooled", "token", "poolandtoken"], default="poolandtoken",
                    help="Feature type: 'pooled' for single vector, 'token' for per-token, "
                         "'poolandtoken' for concatenated [pooled, token]. "
                         "Defaults to 'token' for evaclip, 'pooled' for clip/pe")
    ap.add_argument("--batch_size", type=int, default=256,
                    help="Text batch size")
    ap.add_argument("--save_dtype", type=str, choices=["float16", "float32"], default="float16",
                    help="Numpy dtype used when saving features")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                    help="Device for inference")
    ap.add_argument("--skip_existing", action="store_true", default=True,
                    help="Skip sentences whose .npy already exists")
    ap.add_argument("--splits", type=str, nargs="*", default=None,
                    help="Which splits to process. If omitted, process all top-level keys.")
    ap.add_argument("--drop_special_tokens", action="store_true", default=False,
                    help="If set, drop BOS/EOS tokens before saving (only for token-level features)")
    return ap.parse_args()


def load_json(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def collect_all_sentences(js: Dict, splits: Optional[List[str]]) -> List[Tuple[str, str]]:
    """
    Returns a flat list of (sentence_id, sentence) across requested splits (or all splits).
    Deduplicates by sentence_id, keeping the first occurrence.
    """
    split_keys = splits if splits else list(js.keys())
    seen = {}
    for split in split_keys:
        if split not in js:
            print(f"[WARN] Split '{split}' not found. Skipping.")
            continue
        for _vid, meta in js[split].items():
            for ann in meta.get("annotations", []):
                sid = str(ann["sentence_id"])
                sent = (ann.get("sentence") or "").strip()
                if not sent:
                    continue
                if sid not in seen:
                    seen[sid] = sent
                else:
                    if seen[sid] != sent:
                        print(f"[WARN] Duplicate sentence_id '{sid}' with differing text encountered; keeping first.")
    return [(sid, sent) for sid, sent in seen.items()]


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def np_dtype_from_str(s: str):
    return np.float16 if s == "float16" else np.float32


# ============================================================================
# Model-specific loaders and extractors
# ============================================================================

def load_evaclip_model(model_name: str, device: str):
    """Load EVA-CLIP model and tokenizer."""
    from transformers import AutoModel, CLIPTokenizer
    
    print(f"[INFO] Loading EVA-CLIP model '{model_name}' ...")
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True
    ).to(device).eval()
    
    return model, tokenizer


def load_clip_model(model_name: str, device: str):
    """Load OpenAI CLIP model and tokenizer."""
    from transformers import CLIPModel, CLIPTokenizer
    
    print(f"[INFO] Loading CLIP model '{model_name}' ...")
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    try:
        model = CLIPModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            use_safetensors=True,
        )
    except TypeError:
        # Newer transformers versions use 'dtype' instead of 'torch_dtype'
        model = CLIPModel.from_pretrained(
            model_name,
            dtype=torch.float16,
            use_safetensors=True,
        )
    model = model.to(device).eval()
    
    return model, tokenizer


def load_pe_model(model_name: str, device: str):
    """Load PerceptionEncoder model and tokenizer."""
    import core.vision_encoder.pe as pe
    import core.vision_encoder.transforms as transforms
    
    print(f"[INFO] Loading PerceptionEncoder model '{model_name}' ...")
    model = pe.CLIP.from_config(model_name, pretrained=True)
    model = model.to(device).eval()
    
    # Get tokenizer with model's context length
    tokenizer = transforms.get_text_tokenizer(model.context_length)
    
    return model, tokenizer


@torch.no_grad()
def extract_evaclip_features(model, tokenizer, sentences: List[str], device: str, 
                              feature_type: str, drop_special_tokens: bool):
    """Extract features using EVA-CLIP text encoder."""
    toks = tokenizer(
        sentences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77
    )
    input_ids = toks.input_ids.to(device)
    attention_mask = toks.attention_mask.to(device) if "attention_mask" in toks else None
    
    if attention_mask is None:
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        attention_mask = (input_ids != pad_id).long()
    
    # Get token-level hidden states
    if hasattr(model, "text_model"):
        out = model.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
        )
        hidden_states = out.last_hidden_state  # [B, T, H]
    elif hasattr(model, "text_encoder"):
        out = model.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
        )
        if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
            hidden_states = out.last_hidden_state
        elif hasattr(out, "hidden_states") and out.hidden_states is not None:
            hidden_states = out.hidden_states[-1]
        else:
            raise AttributeError("Could not get hidden states from text_encoder")
    else:
        raise AttributeError("Could not locate text transformer on the model.")
    
    hidden_states = hidden_states.cpu()
    input_ids_cpu = input_ids.cpu()
    attention_mask_cpu = attention_mask.cpu()
    
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    
    results = []
    for b in range(len(sentences)):
        mask = attention_mask_cpu[b].bool()
        
        if feature_type == "pooled":
            # Use EOS token position for pooling (CLIP-style)
            if eos_id is not None:
                eos_positions = (input_ids_cpu[b] == eos_id).nonzero(as_tuple=True)[0]
                if len(eos_positions) > 0:
                    pooled = hidden_states[b, eos_positions[0]]
                else:
                    # Fallback: use last valid token
                    valid_len = mask.sum().item()
                    pooled = hidden_states[b, valid_len - 1]
            else:
                valid_len = mask.sum().item()
                pooled = hidden_states[b, valid_len - 1]
            results.append(pooled)
        else:  # token-level
            if drop_special_tokens:
                ids_b = input_ids_cpu[b]
                if bos_id is not None:
                    mask = mask & (ids_b != bos_id)
                if eos_id is not None:
                    mask = mask & (ids_b != eos_id)
            trimmed = hidden_states[b][mask]
            results.append(trimmed)
    
    return results


@torch.no_grad()
def extract_clip_features(model, tokenizer, sentences: List[str], device: str,
                          feature_type: str, drop_special_tokens: bool):
    """Extract features using OpenAI CLIP text encoder."""
    toks = tokenizer(
        sentences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77
    )
    input_ids = toks.input_ids.to(device)
    attention_mask = toks.attention_mask.to(device) if "attention_mask" in toks else None
    
    if attention_mask is None:
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        attention_mask = (input_ids != pad_id).long()
    
    with torch.amp.autocast('cuda', enabled=device.startswith("cuda")):
        if feature_type == "pooled":
            # Use get_text_features for pooled output (already projected)
            pooled_features = model.get_text_features(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            pooled_features = pooled_features.cpu()
            return [pooled_features[b] for b in range(len(sentences))]
        else:
            # Get token-level features via text_model
            out = model.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
            )
            # Output can be tuple or dict depending on transformers version
            if hasattr(out, "last_hidden_state"):
                hidden_states = out.last_hidden_state.cpu()  # [B, T, H]
            else:
                # Tuple output: (last_hidden_state, pooler_output)
                hidden_states = out[0].cpu()  # [B, T, H]
            input_ids_cpu = input_ids.cpu()
            attention_mask_cpu = attention_mask.cpu()
            
            bos_id = tokenizer.bos_token_id
            eos_id = tokenizer.eos_token_id
            
            results = []
            for b in range(len(sentences)):
                mask = attention_mask_cpu[b].bool()
                if drop_special_tokens:
                    ids_b = input_ids_cpu[b]
                    if bos_id is not None:
                        mask = mask & (ids_b != bos_id)
                    if eos_id is not None:
                        mask = mask & (ids_b != eos_id)
                trimmed = hidden_states[b][mask]
                results.append(trimmed)
            return results


@torch.no_grad()
def extract_pe_features(model, tokenizer, sentences: List[str], device: str,
                        feature_type: str, drop_special_tokens: bool):
    """Extract features using PerceptionEncoder text encoder.
    
    Supports both pooled and token-level features via the TextTransformer's
    built-in output_tokens flag.
    """
    # Tokenize using PE's tokenizer
    text_tokens = tokenizer(sentences).to(device)
    
    # Enable token output if needed (TextTransformer returns (pooled, tokens) when True)
    original_output_tokens = model.output_tokens
    need_tokens = feature_type in ("token", "poolandtoken")
    if need_tokens:
        model.output_tokens = True
    
    with torch.amp.autocast('cuda', enabled=device.startswith("cuda")):
        if hasattr(model, 'encode_text'):
            result = model.encode_text(text_tokens)
        else:
            dummy_image = torch.zeros(1, 3, model.image_size, model.image_size, 
                                      device=device, dtype=torch.float16)
            _, result, _ = model(dummy_image.expand(len(sentences), -1, -1, -1), text_tokens)
    
    # Restore original setting
    model.output_tokens = original_output_tokens
    
    if feature_type == "pooled":
        text_features = result.cpu() if not isinstance(result, tuple) else result[0].cpu()
        return [text_features[b] for b in range(len(sentences))]
    else:
        # Token-level (or poolandtoken): result is (pooled, tokens) where tokens is [B, T, H]
        if isinstance(result, tuple):
            pooled_out, tokens = result
        else:
            # Fallback: if output_tokens didn't work, result is just pooled
            print("[WARN] Could not get token-level features; falling back to pooled.")
            text_features = result.cpu()
            return [text_features[b] for b in range(len(sentences))]
        
        pooled_out = pooled_out.cpu()
        tokens = tokens.cpu()
        text_tokens_cpu = text_tokens.cpu()
        
        # PE tokenizer uses pad_id=0; build attention mask from that
        pad_id = getattr(model, 'pad_id', 0)
        attention_mask = (text_tokens_cpu != pad_id)
        
        # Identify BOS/EOS for optional stripping
        # In PE's CLIP tokenizer: BOS = 49406, EOS = 49407 (standard CLIP vocab)
        bos_id = 49406
        eos_id = 49407
        
        results = []
        for b in range(len(sentences)):
            mask = attention_mask[b].bool()
            if drop_special_tokens:
                ids_b = text_tokens_cpu[b]
                mask = mask & (ids_b != bos_id) & (ids_b != eos_id)
            trimmed = tokens[b][mask]  # [T_valid, H]
            
            if feature_type == "poolandtoken":
                pooled = pooled_out[b]
                if pooled.dim() == 1:
                    pooled = pooled.unsqueeze(0)  # [1, H]
                trimmed = torch.cat([pooled, trimmed], dim=0)  # [1+T_valid, H]
            
            results.append(trimmed)
        return results


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()
    
    # Set default model names based on model_type
    default_models = {
        "evaclip": "BAAI/EVA-CLIP-8B-448",
        "clip": "openai/clip-vit-large-patch14-336",
        "pe": "PE-Core-L14-336",
    }
    
    if args.model_name is None:
        args.model_name = default_models[args.model_type]
    
    # Set default feature_type based on model_type
    if args.feature_type is None:
        args.feature_type = "token" if args.model_type == "evaclip" else "pooled"
    
    # Set default output_dir based on model_type
    if args.output_dir is None:
        args.output_dir = os.path.join("text_features", args.model_type, args.feature_type)
    
    ensure_dir(args.output_dir)
    
    print(f"[INFO] Model type: {args.model_type}")
    print(f"[INFO] Model name: {args.model_name}")
    print(f"[INFO] Feature type: {args.feature_type}")
    print(f"[INFO] Output dir: {args.output_dir}")

    print(f"[INFO] Loading JSON from {args.json_path}")
    js = load_json(args.json_path)

    pairs = collect_all_sentences(js, args.splits)
    print(f"[INFO] Collected {len(pairs)} unique sentences across splits {args.splits if args.splits else '(all)'}.")

    if args.skip_existing:
        before = len(pairs)
        pairs = [(sid, s) for sid, s in pairs if not os.path.exists(os.path.join(args.output_dir, f"{sid}.npy"))]
        skipped = before - len(pairs)
        if skipped:
            print(f"[INFO] Skipping {skipped} already-saved embeddings.")

    if not pairs:
        print("[INFO] Nothing to encode. Exiting.")
        return

    # Load model based on model_type
    if args.model_type == "evaclip":
        model, tokenizer = load_evaclip_model(args.model_name, args.device)
        extract_fn = extract_evaclip_features
    elif args.model_type == "clip":
        model, tokenizer = load_clip_model(args.model_name, args.device)
        extract_fn = extract_clip_features
    elif args.model_type == "pe":
        model, tokenizer = load_pe_model(args.model_name, args.device)
        extract_fn = extract_pe_features
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    torch.set_grad_enabled(False)
    np_dtype = np_dtype_from_str(args.save_dtype)
    bsz = args.batch_size

    for i in tqdm(range(0, len(pairs), bsz), ncols=100, desc=f"Extracting {args.model_type} text features"):
        chunk = pairs[i:i+bsz]
        sids, sents = zip(*chunk)

        with torch.cuda.amp.autocast(enabled=args.device.startswith("cuda")):
            if args.feature_type == "poolandtoken" and args.model_type != "pe":
                # For CLIP/EVA-CLIP: two separate calls to get pooled and token features
                pooled_features = extract_fn(
                    model, tokenizer, list(sents), args.device,
                    "pooled", args.drop_special_tokens
                )
                token_features = extract_fn(
                    model, tokenizer, list(sents), args.device,
                    "token", args.drop_special_tokens
                )
                features = []
                for b in range(len(sents)):
                    pooled = pooled_features[b]
                    if pooled.dim() == 1:
                        pooled = pooled.unsqueeze(0)  # [1, H]
                    features.append(torch.cat([pooled, token_features[b]], dim=0))  # [1+T, H]
            else:
                # PE handles poolandtoken natively in a single forward pass
                features = extract_fn(
                    model, tokenizer, list(sents), args.device,
                    args.feature_type, args.drop_special_tokens
                )

        for b, sid in enumerate(sids):
            feat = features[b]
            
            # For pooled features, add a leading dimension: (Dim,) -> (1, Dim)
            # This makes it consistent with token-level features which are (T, Dim)
            if args.feature_type == "pooled" and feat.dim() == 1:
                feat = feat.unsqueeze(0)
            
            # Cast to save dtype
            if args.save_dtype == "float32":
                feat = feat.float()
            arr = feat.numpy().astype(np_dtype)

            out_path = os.path.join(args.output_dir, f"{sid}.npy")
            if args.skip_existing and os.path.exists(out_path):
                continue
            np.save(out_path, arr)

    print(f"[DONE] Saved {len(pairs)} text embeddings to {args.output_dir}")


if __name__ == "__main__":
    main()

