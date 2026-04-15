"""
UniversalVTG Off-the-Shelf Inference Tool

This module provides a clean, easy-to-use interface for UniversalVTG model inference.

Features are expected at 2 fps with clip_size=15, clip_stride=15, downsample_rate=1.
Text and video encoding via PerceptionEncoder (PE-Core-L14-336) is built in.

Important:
    - UniversalVTG defaults to the release feature contract:
      ``feature_fps=2.0`` plus the feature length to produce timestamps
      in seconds.
    - If you know the original/source video FPS, you can still pass it
      via ``fps`` for a more explicit source-time conversion path.

Usage (raw features):
    model = UniversalVTG()
    results = model.predict(
        vid_features,
        text_features,
        feature_fps=2.0,
    )

Usage (raw text string — auto-encodes with PE):
    model = UniversalVTG()
    results = model.predict(
        vid_features,
        "a person cooking on the stove",
        feature_fps=2.0,
    )

Usage (end-to-end from video file):
    model = UniversalVTG()
    results = model.predict_video("video.mp4", "a person cooking on the stove")
"""

import copy
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from libs.core import load_opt
from libs.modeling.model import make_models_net, PtGenerator
from libs.nms import batched_nms

# ── Fixed data constants for UniversalVTG inference ─────────────────────
# These are invariant across all datasets the model was trained on.
CLIP_SIZE = 15
CLIP_STRIDE = 15
DOWNSAMPLE_RATE = 1
FEATURES_FPS = 2.0           # feature extraction rate
MAX_VID_LEN = 35584           # generous upper bound; model handles variable lengths
MAX_TEXT_LEN = 72             # PE-Core text encoder context length
PE_MODEL_NAME = 'PE-Core-L14-336'
EXTRACTION_BATCH_SIZE = 64


class UniversalVTG:
    """Off-the-shelf UniversalVTG model for video moment retrieval.

    Features are expected at 2 fps with clip_size=15, clip_stride=15,
    and downsample_rate=1. Text and video encoding via PerceptionEncoder
    is built in — you can pass raw strings or video paths directly.

    By default, timestamp conversion follows the release feature contract
    (`feature_fps=2.0` plus feature length). If the original/source video
    FPS is known, you can pass it explicitly via ``fps``.
    """

    def __init__(self, experiment_name='experiments/universalvtg',
                 checkpoint_name='best', device='cuda',
                 enable_query_unifier: bool = False,
                 query_unifier_model_name: str = 'Qwen/Qwen3-4B-Instruct-2507',
                 query_unifier_prompt_path: Union[str, Path] = 'unifier.txt',
                 query_unifier_device: Optional[str] = None,
                 query_unifier_max_new_tokens: int = 128,
                 query_unifier_temperature: float = 0.0):
        """
        Initialize the UniversalVTG model.

        Args:
            experiment_name: Path to the experiment folder containing
                             ``opt.yaml`` and ``models/<checkpoint>.pth``.
                             Defaults to ``'experiments/universalvtg'``.
            checkpoint_name: Checkpoint file stem (default: ``'best'``).
            device: Torch device string (default: ``'cuda'``).
            enable_query_unifier: Whether to opt into query unification by
                                  default for raw-string inference calls.
            query_unifier_model_name: Hugging Face model ID or local path for
                                      the Query Unifier LLM.
            query_unifier_prompt_path: Prompt file path. Relative paths are
                                       resolved from the repo/module root.
            query_unifier_device: Device for the Query Unifier. Defaults to the
                                  main inference device when unspecified.
            query_unifier_max_new_tokens: Generation cap for converted queries.
            query_unifier_temperature: Sampling temperature for the unifier.
        """
        self.device = device
        self.experiment_path = str(Path(experiment_name).resolve())
        self.enable_query_unifier = enable_query_unifier
        self.query_unifier_model_name = query_unifier_model_name
        self.query_unifier_prompt_path = query_unifier_prompt_path
        self.query_unifier_device = query_unifier_device or device
        self.query_unifier_max_new_tokens = query_unifier_max_new_tokens
        self.query_unifier_temperature = query_unifier_temperature

        # ── Load configuration from opt.yaml ──────────────────────────
        opt_path = os.path.join(self.experiment_path, 'opt.yaml')
        self.opt = self._load_opt(opt_path)

        # ── Create & load model ───────────────────────────────────────
        self.model = make_models_net(self.opt).to(device)

        checkpoint_path = os.path.join(
            self.experiment_path, 'models', f'{checkpoint_name}.pth',
        )
        self._load_checkpoint(checkpoint_path)

        self.model.eval()
        self.model.requires_grad_(False)

        # ── Point generator ───────────────────────────────────────────
        self.pt_gen = PtGenerator(**self.opt['pt_gen']).to(device)

        # ── Read all remaining hyperparameters from opt ───────────────
        nms_cfg = self.opt['eval']['nms']
        self.nms_iou_thresh = nms_cfg['iou_thresh']
        self.nms_min_score = nms_cfg['min_score']
        self.nms_max_num_segs = nms_cfg['max_num_segs']
        self.nms_mode = nms_cfg['mode']
        self.nms_sigma = nms_cfg['sigma']
        self.nms_voting_thresh = nms_cfg['voting_thresh']

        eval_cfg = self.opt['eval']
        self.pre_nms_topk = eval_cfg['pre_nms_topk']
        self.pre_nms_thresh = eval_cfg['pre_nms_thresh']
        self.seg_len_thresh = eval_cfg['seg_len_thresh']

        model_cfg = self.opt['model']
        self.early_fusion = model_cfg.get('early_fusion', True)
        self.use_mst = model_cfg.get('use_mst', False)
        self.vid_stride = model_cfg['vid_stride']

        # ── FPN padding stride ────────────────────────────────────────
        num_fpn_levels = model_cfg['num_fpn_levels']
        mha_win_size = model_cfg['vid_net'].get('mha_win_size', 0)

        min_chunk_size = 1
        for i in range(num_fpn_levels):
            stride = 2 ** i
            if mha_win_size > 0:
                stride *= (mha_win_size // 2) * 2
            min_chunk_size = max(min_chunk_size, stride)
        self.min_chunk_size = min_chunk_size
        self.padding_stride = self.min_chunk_size * self.vid_stride

        # ── Fixed data constants ──────────────────────────────────────
        self.clip_size = CLIP_SIZE
        self.clip_stride = CLIP_STRIDE
        self.downsample_rate = DOWNSAMPLE_RATE

        # ── Lazy-loaded encoders (initialized on first use) ───────────
        self._text_model = None
        self._text_tokenizer = None
        self._video_extractor = None
        self._video_preprocess = None
        self._query_unifier_model = None
        self._query_unifier_tokenizer = None
        self._query_unifier_prompt_text = None

        # ── Summary ──────────────────────────────────────────────────
        print(f"✓ UniversalVTG model loaded successfully!")
        print(f"  - Experiment: {self.experiment_path}")
        print(f"  - Checkpoint: {checkpoint_name}")
        print(f"  - Device: {device}")
        print(f"  - Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  - pt_gen.max_seq_len: {self.opt['pt_gen']['max_seq_len']}")
        print(f"  - Clip stride: {self.clip_stride}, Clip size: {self.clip_size}")
        print(f"  - Min chunk size: {self.min_chunk_size}, Vid stride: {self.vid_stride}")
        if self.enable_query_unifier:
            print(
                f"  - Query Unifier: enabled by default "
                f"({self.query_unifier_model_name} on {self.query_unifier_device})"
            )

    # ------------------------------------------------------------------
    # Config loading
    # ------------------------------------------------------------------
    @staticmethod
    def _load_opt(opt_path):
        """Load experiment opt.yaml with proper multi-dataset handling.

        The experiment config uses ``train.datasets`` / ``eval.datasets``
        (multi-dataset format).  ``load_opt`` expects flat ``train.data`` /
        ``eval.data``, so we extract the first dataset's data block into
        those slots and set ``max_vid_len`` / ``max_text_len`` to the
        generous inference-time values defined at module level.
        """
        with open(opt_path, 'r') as f:
            raw = yaml.load(f, Loader=yaml.FullLoader)

        # Flatten multi-dataset config for load_opt
        is_multi = (
            'datasets' in raw.get('eval', {})
            or 'datasets' in raw.get('train', {})
        )
        if is_multi:
            processed = copy.deepcopy(raw)
            if 'datasets' in processed.get('train', {}):
                processed['train']['data'] = copy.deepcopy(
                    processed['train']['datasets'][0]['data'],
                )
            if 'datasets' in processed.get('eval', {}):
                processed['eval']['data'] = copy.deepcopy(
                    processed['eval']['datasets'][0]['data'],
                )
            # Override with inference-time constants
            processed['eval']['data']['max_vid_len'] = MAX_VID_LEN
            processed['eval']['data']['max_text_len'] = MAX_TEXT_LEN
            processed['eval']['data']['clip_size'] = CLIP_SIZE
            processed['eval']['data']['clip_stride'] = CLIP_STRIDE
            processed['eval']['data']['downsample_rate'] = DOWNSAMPLE_RATE
            processed['eval']['data']['to_fixed_len'] = False
            processed['eval']['max_vid_len'] = MAX_VID_LEN

            with tempfile.NamedTemporaryFile(
                mode='w', suffix='.yaml', delete=False,
            ) as tmp:
                yaml.dump(processed, tmp)
                tmp_path = Path(tmp.name)

            try:
                opt = load_opt(str(tmp_path), is_training=False)
            finally:
                tmp_path.unlink(missing_ok=True)
        else:
            opt = load_opt(opt_path, is_training=False)

        return opt

    # ------------------------------------------------------------------
    # Checkpoint loading
    # ------------------------------------------------------------------
    def _load_checkpoint(self, checkpoint_path):
        """Load model weights from checkpoint."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        if 'model_ema' in ckpt:
            self.model.load_state_dict(ckpt['model_ema'])
            print("  → Loaded EMA weights")
        elif 'model' in ckpt:
            self.model.load_state_dict(ckpt['model'])
            print("  → Loaded model weights")
        else:
            raise ValueError("Invalid checkpoint format")

    # ==================================================================
    # Built-in text encoder (lazy-loaded)
    # ==================================================================
    def _ensure_text_encoder(self):
        """Lazy-load the PE-Core text encoder on first use."""
        if self._text_model is not None:
            return

        from feature_extraction.extract_text_features import (
            load_pe_model,
        )

        print(f"  [encoder] Loading PE text encoder ({PE_MODEL_NAME}) …")
        self._text_model, self._text_tokenizer = load_pe_model(
            PE_MODEL_NAME, self.device,
        )

    @torch.no_grad()
    def encode_text(self, query: Union[str, List[str]]) -> torch.Tensor:
        """Encode a raw text query into feature tensor(s).

        Args:
            query: A single string or list of strings.

        Returns:
            Tensor of shape ``(feature_dim, T)`` for a single query,
            or ``(B, feature_dim, T)`` for a list.
        """
        self._ensure_text_encoder()
        from feature_extraction.extract_text_features import (
            extract_pe_features,
        )

        single = isinstance(query, str)
        sentences = [query] if single else list(query)

        features_list = extract_pe_features(
            self._text_model,
            self._text_tokenizer,
            sentences,
            self.device,
            'poolandtoken',   # match training format
            False,            # keep special tokens
        )

        # Each element is (T_valid, D) — transpose to (D, T)
        tensors = []
        for feat in features_list:
            arr = feat.detach().cpu().numpy().astype(np.float32)
            tensors.append(
                torch.from_numpy(np.ascontiguousarray(arr.T)),
            )

        if single:
            return tensors[0]
        return torch.stack(tensors, dim=0)

    # ==================================================================
    # Optional Query Unifier (lazy-loaded)
    # ==================================================================
    def _resolve_query_unifier_prompt_path(self) -> Path:
        """Resolve the Query Unifier prompt path."""
        prompt_path = Path(self.query_unifier_prompt_path)
        if prompt_path.is_absolute():
            return prompt_path
        return Path(__file__).resolve().parent / prompt_path

    def _load_query_unifier_prompt(self) -> str:
        """Load and cache the Query Unifier system prompt."""
        if self._query_unifier_prompt_text is not None:
            return self._query_unifier_prompt_text

        prompt_path = self._resolve_query_unifier_prompt_path()
        if not prompt_path.exists():
            raise FileNotFoundError(
                f"Query Unifier prompt not found: {prompt_path}",
            )
        self._query_unifier_prompt_text = prompt_path.read_text().strip()
        return self._query_unifier_prompt_text

    def _query_unifier_enabled(self, use_unifier: Optional[bool]) -> bool:
        """Resolve per-call Query Unifier enablement."""
        if use_unifier is None:
            return bool(getattr(self, 'enable_query_unifier', False))
        return bool(use_unifier)

    def _query_unifier_dtype(self) -> Optional[torch.dtype]:
        """Choose a stable default dtype for the Query Unifier."""
        device = str(getattr(self, 'query_unifier_device', self.device)).lower()
        if device.startswith('cuda'):
            return torch.float16
        return None

    def _ensure_query_unifier(self) -> None:
        """Lazy-load the Query Unifier model/tokenizer on first use."""
        if self._query_unifier_model is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        prompt_path = self._resolve_query_unifier_prompt_path()
        device = getattr(self, 'query_unifier_device', self.device)
        dtype = self._query_unifier_dtype()

        print(
            f"  [unifier] Loading Query Unifier "
            f"({self.query_unifier_model_name}) on {device} …",
        )
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.query_unifier_model_name,
                trust_remote_code=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.query_unifier_model_name,
                torch_dtype=dtype,
                trust_remote_code=True,
            )
            model = model.to(device)
            model.eval()
        except Exception as exc:
            raise RuntimeError(
                "Failed to load the Query Unifier model "
                f"'{self.query_unifier_model_name}' on device "
                f"'{device}'. Update query_unifier_model_name / "
                f"query_unifier_device or ensure weights are available. "
                f"Prompt path: {prompt_path}"
            ) from exc

        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token

        self._query_unifier_tokenizer = tokenizer
        self._query_unifier_model = model
        self._load_query_unifier_prompt()

    def unload_query_unifier(self) -> None:
        """Release Query Unifier model/tokenizer references."""
        self._query_unifier_model = None
        self._query_unifier_tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _normalize_unifier_output(self, text: str) -> str:
        """Normalize Query Unifier output to a single statement."""
        stripped = text.strip().strip('"').strip("'")
        lines = [line.strip() for line in stripped.splitlines() if line.strip()]
        if not lines:
            raise RuntimeError("Query Unifier returned an empty response.")
        return lines[0]

    @torch.no_grad()
    def unify_query(self, query: str) -> str:
        """Convert a raw query into the unified canonical format."""
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("Query must not be empty.")

        self._ensure_query_unifier()
        tokenizer = self._query_unifier_tokenizer
        model = self._query_unifier_model
        prompt_text = self._load_query_unifier_prompt()

        messages = [
            {'role': 'system', 'content': prompt_text},
            {'role': 'user', 'content': normalized_query},
        ]
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        batch = tokenizer(rendered, return_tensors='pt')
        batch = {
            key: value.to(getattr(self, 'query_unifier_device', self.device))
            for key, value in batch.items()
        }

        generate_kwargs = {
            'max_new_tokens': getattr(self, 'query_unifier_max_new_tokens', 128),
            'pad_token_id': tokenizer.pad_token_id,
            'eos_token_id': tokenizer.eos_token_id,
        }
        temperature = float(getattr(self, 'query_unifier_temperature', 0.0))
        if temperature > 0:
            generate_kwargs['do_sample'] = True
            generate_kwargs['temperature'] = temperature
        else:
            generate_kwargs['do_sample'] = False

        output_ids = model.generate(**batch, **generate_kwargs)
        prompt_length = batch['input_ids'].shape[-1]
        generated_ids = output_ids[0, prompt_length:]
        decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)
        return self._normalize_unifier_output(decoded)

    def _prepare_query_text(
        self,
        query: Union[str, List[str]],
        use_unifier: Optional[bool] = None,
    ) -> tuple[Union[str, List[str]], Optional[Dict[str, Any]]]:
        """Prepare raw text queries and optional query_info metadata."""
        resolved_use_unifier = self._query_unifier_enabled(use_unifier)

        if isinstance(query, list):
            if resolved_use_unifier:
                raise ValueError(
                    "Query Unifier v1 only supports a single raw string query. "
                    "Disable use_unifier for batched/list inputs.",
                )
            return query, None

        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("Query must not be empty.")

        converted_query = None
        model_query = normalized_query
        if resolved_use_unifier:
            converted_query = self.unify_query(normalized_query)
            model_query = converted_query

        query_info = {
            'original_query': normalized_query,
            'model_query': model_query,
            'converted_query': converted_query,
            'used_unifier': resolved_use_unifier,
            'query_unifier_model': (
                getattr(self, 'query_unifier_model_name', None)
                if resolved_use_unifier else None
            ),
            'query_unifier_device': (
                getattr(self, 'query_unifier_device', self.device)
                if resolved_use_unifier else None
            ),
        }
        return model_query, query_info

    # ==================================================================
    # Built-in video encoder (lazy-loaded)
    # ==================================================================
    def _ensure_video_encoder(self):
        """Lazy-load the PE-Core video feature extractor on first use."""
        if self._video_extractor is not None:
            return

        from feature_extraction.extract_visual_features import (
            PEFeatureExtractor,
        )
        import core.vision_encoder.transforms as pe_transforms

        print(f"  [encoder] Loading PE video encoder ({PE_MODEL_NAME}) …")
        self._video_extractor = (
            PEFeatureExtractor(PE_MODEL_NAME).to(self.device).eval()
        )
        self._video_preprocess = pe_transforms.get_image_transform(
            self._video_extractor.image_size,
        )

    @torch.no_grad()
    def encode_video(self, video_path: str) -> dict:
        """Extract video features from a video file at 2 fps.

        Args:
            video_path: Path to a video file (mp4, avi, etc.).

        Returns:
            dict with keys:
                - ``'features'``: Tensor of shape ``(feature_dim, T)``
                - ``'fps'``: Original video FPS (float)
                - ``'duration'``: Video duration in seconds (float)
                - ``'num_frames'``: Total number of frames (int)
        """
        self._ensure_video_encoder()

        import decord
        from PIL import Image
        from feature_extraction.extract_visual_features import (
            sample_frames_clips,
        )

        reader = decord.VideoReader(str(video_path), num_threads=1)
        total_frames = len(reader)
        video_fps = float(reader.get_avg_fps()) or 30.0
        duration = total_frames / video_fps
        num_samples = max(1, int(duration * FEATURES_FPS))
        frame_idxs = sample_frames_clips(
            0, max(total_frames - 1, 0), total_frames, num_samples,
        )

        batches = []
        for start in range(0, len(frame_idxs), EXTRACTION_BATCH_SIZE):
            batch_indices = frame_idxs[start:start + EXTRACTION_BATCH_SIZE]
            batch_frames = reader.get_batch(batch_indices)
            if batch_frames.device.type != 'cpu':
                batch_frames = batch_frames.cpu()
            processed = []
            for frame in batch_frames:
                processed.append(
                    self._video_preprocess(Image.fromarray(frame.numpy())),
                )
            pixel_values = torch.stack(processed, dim=0).to(
                self.device, dtype=torch.float16,
            )
            with torch.inference_mode(), torch.amp.autocast(
                'cuda', enabled=self.device.startswith('cuda'),
            ):
                batch_feats = self._video_extractor(pixel_values).detach().cpu()
            batches.append(batch_feats)

        features = torch.cat(batches, dim=0)  # (T, D)
        del reader

        # Transpose to (D, T) — the format expected by predict()
        features_dt = features.T.contiguous().float()

        print(f"  [encoder] Video: {Path(video_path).name}, "
              f"{features.shape[0]} frames @ {FEATURES_FPS} fps, "
              f"duration={duration:.1f}s")

        return {
            'features': features_dt,
            'fps': video_fps,
            'duration': duration,
            'num_frames': total_frames,
        }

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, vid_features, text_features, vid_mask=None, text_mask=None,
                top_k=None, score_thresh=None, return_raw=False,
                fps=None, duration=None, feature_fps=FEATURES_FPS,
                use_unifier: Optional[bool] = None,
                return_query_info: bool = True):
        """
        Predict temporal segments given video and text features.

        Args:
            vid_features: (feature_dim, T) or (B, feature_dim, T) tensor.
            text_features: (feature_dim, L) or (B, feature_dim, L) tensor,
                           **or a raw text string** (auto-encoded via PE).
            vid_mask: Optional bool mask for video.
            text_mask: Optional bool mask for text.
            top_k: Number of top predictions (default from opt).
            score_thresh: Min score threshold (default from opt).
            return_raw: Return raw predictions without NMS.
            fps: Original/source video FPS used for timestamp conversion.
                 This is optional when using the release feature contract
                 (features extracted at ``feature_fps`` with the fixed
                 clip geometry). If ``None``, timestamps are derived from
                 ``feature_fps`` instead.
            duration: Video duration in seconds for clamping. If ``None``,
                      automatically computed as ``T / feature_fps`` from the
                      video feature length.
            feature_fps: Feature extraction FPS for ``vid_features``.
                         Defaults to the release extraction rate (2.0).
            use_unifier: Whether to canonicalize raw string queries before
                         PE encoding. Only supported for a single raw string.
            return_query_info: Whether to attach ``query_info`` metadata for
                               supported raw-string calls.

        Returns:
            dict with 'segments', 'scores', 'raw_segments', 'raw_scores'.
            Segments are in seconds.
        """
        resolved_use_unifier = self._query_unifier_enabled(use_unifier)
        query_info = None

        # ── Auto-encode text if a string is passed ────────────────────
        if isinstance(text_features, str):
            prepared_query, query_info = self._prepare_query_text(
                text_features,
                use_unifier=resolved_use_unifier,
            )
            text_features = self.encode_text(prepared_query)
        elif isinstance(text_features, list) and text_features and isinstance(text_features[0], str):
            prepared_queries, query_info = self._prepare_query_text(
                text_features,
                use_unifier=resolved_use_unifier,
            )
            text_features = self.encode_text(prepared_queries)
        elif resolved_use_unifier:
            raise ValueError(
                "Query Unifier requires a raw text string input. "
                "Disable use_unifier when passing precomputed text tensors.",
            )

        if vid_features.ndim == 2:
            vid_features = vid_features.unsqueeze(0)
        if text_features.ndim == 2:
            text_features = text_features.unsqueeze(0)

        vid_len = vid_features.size(-1)

        # Auto-compute duration from feature length if not provided
        if duration is None:
            duration = vid_len / feature_fps

        if feature_fps <= 0:
            raise ValueError("feature_fps must be positive")

        if vid_mask is None:
            vid_mask = torch.ones(
                vid_features.size(0), vid_len,
                dtype=torch.bool, device=vid_features.device,
            )
        if text_mask is None:
            text_mask = torch.ones(
                text_features.size(0), text_features.size(-1),
                dtype=torch.bool, device=text_features.device,
            )

        vid_features = vid_features.to(self.device)
        text_features = text_features.to(self.device)
        vid_mask = vid_mask.to(self.device)
        text_mask = text_mask.to(self.device)

        # Encode text
        if text_mask.ndim == 2:
            text_mask = text_mask.unsqueeze(1)
        text_encoded, text_mask_encoded = self.model.encode_text(
            text_features, text_mask,
        )

        # Pad video to FPN-aligned length
        input_vid_len = (
            (vid_len + self.padding_stride - 1)
            // self.padding_stride * self.padding_stride
        )
        if input_vid_len > vid_len:
            pad_len = input_vid_len - vid_len
            vid_features = F.pad(vid_features, (0, pad_len))
            vid_mask = F.pad(vid_mask, (0, pad_len), value=0)

        if vid_mask.ndim == 2:
            vid_mask = vid_mask.unsqueeze(1)

        # Forward pass
        if self.early_fusion:
            vid_proj, vid_mask_proj = self.model.vid_proj(vid_features, vid_mask)
            vid_fused, vid_mask_fused = self.model.fusion(
                vid_proj, vid_mask_proj,
                text_encoded, text_mask_encoded, text_size=None,
            )
            fpn, fpn_masks, _, _ = self.model.encode_video(
                vid_fused, vid_mask_fused,
            )
            if self.use_mst:
                fpn_logits, _, fpn_offsets, _ = self.model.fuse_and_predict_mst(
                    fpn, fpn_masks,
                    text_encoded, text_mask_encoded, text_size=None,
                )
            else:
                fpn_logits, _, fpn_offsets, _ = self.model.fuse_and_predict(
                    fpn, fpn_masks,
                    text_encoded, text_mask_encoded, text_size=None,
                )
        else:
            fpn, fpn_masks, _, _ = self.model.encode_video(
                vid_features, vid_mask,
            )
            fpn_logits, fpn_offsets, _ = self.model.fuse_and_predict(
                fpn, fpn_masks,
                text_encoded, text_mask_encoded, text_size=None,
            )

        # Decode
        fpn_n_points = [m.size(-1) for m in fpn_masks]
        fpn_points = self.pt_gen(fpn_n_points)
        raw_segments, raw_scores = self._decode_predictions(
            fpn_points, fpn_logits, fpn_offsets, fpn_masks,
        )

        is_batched = isinstance(raw_segments, list)

        if return_raw:
            return {
                'segments': raw_segments,
                'scores': raw_scores,
                'raw_segments': raw_segments,
                'raw_scores': raw_scores,
                'query_info': query_info if return_query_info else None,
            }

        if score_thresh is None:
            score_thresh = self.nms_min_score
        if top_k is None:
            top_k = self.nms_max_num_segs

        # NMS
        if is_batched:
            nms_segments_list, nms_scores_list = [], []
            for seg, score in zip(raw_segments, raw_scores):
                if len(seg) > 0:
                    nms_seg, nms_sc = batched_nms(
                        seg.cpu(), score.cpu(),
                        iou_thresh=self.nms_iou_thresh,
                        min_score=score_thresh,
                        max_num_segs=top_k,
                        mode=self.nms_mode,
                        sigma=self.nms_sigma,
                        voting_thresh=self.nms_voting_thresh,
                    )
                    nms_segments_list.append(nms_seg.to(self.device))
                    nms_scores_list.append(nms_sc.to(self.device))
                else:
                    nms_segments_list.append(torch.zeros(0, 2, device=self.device))
                    nms_scores_list.append(torch.zeros(0, device=self.device))

            for i in range(len(nms_segments_list)):
                if len(nms_segments_list[i]) > 0:
                    nms_segments_list[i] = self._convert_segments_to_seconds(
                        nms_segments_list[i],
                        fps=fps,
                        feature_fps=feature_fps,
                    )
                    if duration is not None:
                        nms_segments_list[i] = torch.clamp(
                            nms_segments_list[i], min=0, max=duration,
                        )

            return {
                'segments': nms_segments_list,
                'scores': nms_scores_list,
                'raw_segments': raw_segments,
                'raw_scores': raw_scores,
                'query_info': query_info if return_query_info else None,
            }
        else:
            if len(raw_segments) > 0:
                nms_segments, nms_scores = batched_nms(
                    raw_segments.cpu(), raw_scores.cpu(),
                    iou_thresh=self.nms_iou_thresh,
                    min_score=score_thresh,
                    max_num_segs=top_k,
                    mode=self.nms_mode,
                    sigma=self.nms_sigma,
                    voting_thresh=self.nms_voting_thresh,
                )
                nms_segments = nms_segments.to(self.device)
                nms_scores = nms_scores.to(self.device)
            else:
                nms_segments = torch.zeros(0, 2, device=self.device)
                nms_scores = torch.zeros(0, device=self.device)

            if len(nms_segments) > 0:
                nms_segments = self._convert_segments_to_seconds(
                    nms_segments,
                    fps=fps,
                    feature_fps=feature_fps,
                )
                if duration is not None:
                    nms_segments = torch.clamp(nms_segments, min=0, max=duration)

            return {
                'segments': [nms_segments],
                'scores': [nms_scores],
                'raw_segments': [raw_segments],
                'raw_scores': [raw_scores],
                'query_info': query_info if return_query_info else None,
            }

    # ------------------------------------------------------------------
    # End-to-end video prediction
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_video(self, video_path: str, query: Union[str, List[str]],
                      top_k: Optional[int] = None,
                      score_thresh: Optional[float] = None,
                      use_unifier: Optional[bool] = None) -> dict:
        """End-to-end prediction from a video file and raw text query.

        Extracts video features at 2 fps, encodes text via PE, runs the
        model, and returns timestamp predictions in seconds.

        Args:
            video_path: Path to a video file (mp4, avi, etc.).
            query: A text query string (or list of strings).
            top_k: Number of top predictions (default from opt).
            score_thresh: Min score threshold (default from opt).
            use_unifier: Whether to canonicalize a raw string query before PE
                         encoding.

        Returns:
            dict with 'segments', 'scores' (timestamps in seconds),
            plus 'fps', 'duration', 'query'.
        """
        # Encode video
        video_info = self.encode_video(video_path)
        vid_features = video_info['features']
        video_fps = video_info['fps']
        duration = video_info['duration']

        # Encode text (handled automatically by predict if string)
        results = self.predict(
            vid_features, query,
            top_k=top_k,
            score_thresh=score_thresh,
            fps=video_fps,
            feature_fps=FEATURES_FPS,
            duration=duration,
            use_unifier=use_unifier,
        )

        results['fps'] = video_fps
        results['duration'] = duration
        results['query'] = query
        return results

    def _convert_segments_to_seconds(self, segments: torch.Tensor, *, fps: float | None, feature_fps: float) -> torch.Tensor:
        """Convert decoded segment coordinates into seconds.

        If ``fps`` is provided, use the original/raw-video FPS conversion.
        Otherwise, fall back to the release feature-time contract:

            raw_fps = feature_fps * clip_stride

        which yields:

            seconds = (segment * vid_stride + 0.5 * clip_size / clip_stride) / feature_fps
        """
        segments = segments * self.vid_stride
        if fps is not None:
            return (segments * self.clip_stride + 0.5 * self.clip_size) / fps

        clip_center_offset = 0.5 * (self.clip_size / self.clip_stride)
        return (segments + clip_center_offset) / feature_fps

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _decode_predictions(self, fpn_points, fpn_logits, fpn_offsets, fpn_masks):
        """Decode raw predictions into temporal segments."""
        batch_size = fpn_logits[0].size(0)
        batch_segments_list, batch_scores_list = [], []

        for batch_idx in range(batch_size):
            segments_list, scores_list = [], []

            for points, logits, offsets, mask in zip(
                fpn_points, fpn_logits, fpn_offsets, fpn_masks,
            ):
                scores = logits[batch_idx].sigmoid()
                valid_mask = mask[batch_idx].squeeze(0).bool()
                batch_offsets = offsets[batch_idx]

                scores = scores[valid_mask]
                points_valid = points[valid_mask]
                offsets_valid = batch_offsets[valid_mask]

                keep = scores >= self.pre_nms_thresh
                if keep.sum() == 0:
                    continue

                scores = scores[keep]
                points_valid = points_valid[keep]
                offsets_valid = offsets_valid[keep]

                pt_ctr = points_valid[:, 0]
                segments = torch.stack([
                    pt_ctr - offsets_valid[:, 0] * points_valid[:, 3],
                    pt_ctr + offsets_valid[:, 1] * points_valid[:, 3],
                ], dim=1)

                seg_lengths = segments[:, 1] - segments[:, 0]
                keep_len = seg_lengths >= self.seg_len_thresh
                if keep_len.sum() == 0:
                    continue

                segments_list.append(segments[keep_len])
                scores_list.append(scores[keep_len])

            if not segments_list:
                batch_segments_list.append(
                    torch.zeros(0, 2, device=fpn_logits[0].device),
                )
                batch_scores_list.append(
                    torch.zeros(0, device=fpn_logits[0].device),
                )
            else:
                all_segments = torch.cat(segments_list, dim=0)
                all_scores = torch.cat(scores_list, dim=0)

                if len(all_scores) > self.pre_nms_topk:
                    topk_idx = torch.topk(all_scores, k=self.pre_nms_topk)[1]
                    all_segments = all_segments[topk_idx]
                    all_scores = all_scores[topk_idx]

                batch_segments_list.append(all_segments)
                batch_scores_list.append(all_scores)

        if batch_size == 1:
            return batch_segments_list[0], batch_scores_list[0]
        return batch_segments_list, batch_scores_list


# ── CLI entry point ──────────────────────────────────────────────────────
def main():
    """Example usage of UniversalVTG inference."""

    print("=" * 60)
    print("UniversalVTG Off-the-Shelf Inference")
    print("=" * 60)

    model = UniversalVTG(
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )

    # ------------------------------------------------------------------
    # Example 1: Predict with raw feature tensors
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Example 1: Predict with raw feature tensors")
    print("=" * 60)

    vid_features = torch.randn(1024, 481)
    text_features = torch.randn(1024, 16)

    print(f"\nInput shapes:")
    print(f"  Video features: {vid_features.shape}")
    print(f"  Text features:  {text_features.shape}")

    results = model.predict(vid_features, text_features)

    # predict() returns lists (one per batch item); unwrap the first
    segments = results['segments'][0]   # (N, 2)
    scores = results['scores'][0]       # (N,)

    print(f"\nPredictions (after NMS):")
    print(f"  Number of segments: {len(segments)}")

    if len(segments) > 0:
        print(f"\n  Top predictions:")
        for i in range(len(segments)):
            start, end = segments[i]
            score = scores[i]
            print(f"    [{i+1}] Segment: [{start:.2f}, {end:.2f}], Score: {score:.4f}")

    # ------------------------------------------------------------------
    # Example 2: Predict with a raw text string (auto-encoded)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Example 2: Predict with a raw text string")
    print("=" * 60)

    query = "a person cooking on the stove"
    print(f"\n  Query: \"{query}\"")

    results = model.predict(vid_features, query)
    segments = results['segments'][0]
    scores = results['scores'][0]

    print(f"\nPredictions (after NMS):")
    print(f"  Number of segments: {len(segments)}")
    if len(segments) > 0:
        print(f"\n  Top predictions:")
        for i in range(len(segments)):
            start, end = segments[i]
            score = scores[i]
            print(f"    [{i+1}] Segment: [{start:.2f}, {end:.2f}], Score: {score:.4f}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()
