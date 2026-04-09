#!/usr/bin/env python
"""
Extract frame-level visual features for UniversalVTG datasets.

This release ships the Perception Encoder backbone used in the paper via the
`perception_models` submodule. The script also supports an optional DINOv3
backbone through Hugging Face for comparison experiments.
"""

import os
os.environ["DECORD_LOG_LEVEL"] = "ERROR"
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PERCEPTION_MODELS_DIR = REPO_ROOT / "perception_models"
if PERCEPTION_MODELS_DIR.exists():
    sys.path.insert(0, str(PERCEPTION_MODELS_DIR))

import glob
import json
import time
import signal
import argparse
import tempfile
import shutil
from typing import List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
import gc
import multiprocessing as py_mp

import decord
decord.bridge.set_bridge("torch")

# PerceptionEncoder imports
import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as transforms

# DINOv3 / HuggingFace imports
from transformers import AutoModel
from torchvision.transforms import v2 as transforms_v2

import tqdm
from PIL import Image


# --- Helper for Atomic Writes ---
def atomic_save(obj, path):
    """
    Saves object to a temp file first, then renames to ensure atomicity.
    Prevents corrupt .pt files if the process is killed during write.
    """
    dir_name = os.path.dirname(path)
    # Create temp file in the same directory to ensure atomic move is possible (same filesystem)
    with tempfile.NamedTemporaryFile(delete=False, dir=dir_name, suffix='.tmp') as tmp:
        torch.save(obj, tmp)
        tmp_name = tmp.name
    
    # Atomic move
    shutil.move(tmp_name, path)


def _get_video_metadata_worker(video_path: str) -> tuple:
    """Worker function for parallel video metadata caching."""
    video_name = os.path.basename(video_path)
    try:
        vr = decord.VideoReader(video_path, num_threads=1)
        vlen = len(vr)
        fps = vr.get_avg_fps()
        del vr
        return video_name, {"frames": vlen, "fps": round(fps, 3)}
    except Exception:
        return video_name, None


def collate_keep_valid(batch):
    # Top-level collate to be picklable with spawn workers
    return [item for item in batch if item is not None]


class PEFeatureExtractor(nn.Module):
    """PerceptionEncoder feature extractor using PE-Core-L14-336."""
    
    def __init__(self, config_name: str = "PE-Core-L14-336"):
        super().__init__()
        # Load PerceptionEncoder model
        self.model = pe.CLIP.from_config(config_name, pretrained=True)
        self.image_size = self.model.image_size

    @torch.inference_mode()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract image features using PerceptionEncoder."""
        with torch.amp.autocast('cuda'):
            # Use encode_image for image-only features (standard CLIP API)
            image_features = self.model.encode_image(pixel_values)
            return image_features


def make_dinov3_transform(resize_size: int = 256):
    """Official DINOv3 image transform for LVD-1689M pretrained models.
    
    Reference: https://github.com/facebookresearch/dinov3#image-transforms
    """
    to_tensor = transforms_v2.ToImage()
    resize = transforms_v2.Resize((resize_size, resize_size), antialias=True)
    to_float = transforms_v2.ToDtype(torch.float32, scale=True)
    normalize = transforms_v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return transforms_v2.Compose([to_tensor, resize, to_float, normalize])


class DINOv3FeatureExtractor(nn.Module):
    """DINOv3 feature extractor using HuggingFace AutoModel.
    
    Default model: facebook/dinov3-vitl16-pretrain-lvd1689m
    """
    
    def __init__(self, model_name: str = "facebook/dinov3-vitl16-pretrain-lvd1689m"):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        # DINOv3 LVD models use 256x256 input by default
        self.image_size = 256
        self.model_name = model_name

    @torch.inference_mode()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract pooled image features using DINOv3."""
        with torch.amp.autocast('cuda'):
            outputs = self.model(pixel_values=pixel_values)
            # pooler_output: (batch, feature_dim) — CLS/pooled embedding per frame
            return outputs.pooler_output


class _DecordTimeoutError(Exception):
    """Raised when decord operations exceed the timeout."""
    pass


def _decord_timeout_handler(signum, frame):
    raise _DecordTimeoutError("Decord operation timed out")


class VideoStreamDataset(Dataset):
    """
    Optimized dataset for video feature extraction.
    
    Improvements:
    1. Preprocessing (PIL -> Tensor -> Norm) happens HERE (in worker processes).
    2. Minimizes Decord thread contention.
    """
    def __init__(self, base_dataset: "VideoDataset", video_indices: list, batch_size: int, 
                 image_size: int, decord_threads: int = 1, timeout: int = 120,
                 encoder: str = "pe"):
        self.base_dataset = base_dataset
        self.video_indices = video_indices
        self.batch_size = batch_size
        self.decord_threads = decord_threads
        self.image_size = image_size
        self.timeout = timeout  # seconds per video read operation
        self.encoder = encoder  # "pe" or "dinov3"
        
        # Build index: maps dataset index -> (video_idx, batch_order, frame_idxs)
        self.index_map = []
        cache_updated = False
        
        for video_idx in video_indices:
            try:
                # Use cached metadata
                metadata = self.base_dataset.get_video_metadata(video_idx)
                vlen = metadata.get("frames", 0)
                video_fps = metadata.get("fps", self.base_dataset.default_fps)
                
                if vlen == 0:
                    continue
                
                video_name = os.path.basename(self.base_dataset.video_paths[video_idx])
                if video_name not in self.base_dataset.video_metadata:
                    cache_updated = True
                
                start = 0
                end = vlen - 1
                # Use actual video FPS
                num_frames = int((vlen / video_fps) * self.base_dataset.fps_to_read)
                frame_idxs = sample_frames_clips(start, end, vlen, num_frames)
                
                if len(frame_idxs) == 0:
                    continue
                
                num_batches = (len(frame_idxs) + batch_size - 1) // batch_size
                for batch_order in range(num_batches):
                    batch_frame_idxs = frame_idxs[batch_order * batch_size : (batch_order + 1) * batch_size]
                    self.index_map.append({
                        "video_idx": video_idx,
                        "batch_order": batch_order,
                        "frame_idxs": batch_frame_idxs,
                    })
            except Exception:
                continue
        
        if cache_updated:
            self.base_dataset._save_metadata_cache()

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, index):
        task = self.index_map[index]
        video_idx = task["video_idx"]
        batch_order = task["batch_order"]
        frame_idxs = task["frame_idxs"]
        video_path = self.base_dataset.video_paths[video_idx]
        v_id = self.base_dataset.get_video_name(video_idx)
        
        video_reader = None
        use_timeout = self.timeout > 0
        try:
            # Set a timeout alarm to prevent decord from hanging on corrupted videos
            if use_timeout:
                old_handler = signal.signal(signal.SIGALRM, _decord_timeout_handler)
                signal.alarm(self.timeout)
            
            try:
                # OPTIMIZATION: Keep num_threads low (1) inside the worker to prevent 
                # thread explosion when using multiple DataLoader workers.
                video_reader = decord.VideoReader(video_path, num_threads=1)
                frames = video_reader.get_batch(frame_idxs)  # (T, H, W, C) torch uint8
            finally:
                # Cancel the alarm and restore the old handler
                if use_timeout:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
            
            # Move to CPU for preprocessing (avoid pinning GPU memory for uint8)
            if frames.device.type != "cpu":
                frames = frames.cpu()

            # --- PREPROCESSING START (Run in CPU Worker) ---
            if self.encoder == "dinov3":
                # DINOv3: use official torchvision v2 transforms
                # Convert to PIL first so v2.ToImage() correctly handles HWC -> CHW
                preprocess = make_dinov3_transform(self.image_size)
                processed_frames = []
                for i in range(frames.shape[0]):
                    pil_img = Image.fromarray(frames[i].numpy())
                    processed = preprocess(pil_img)
                    processed_frames.append(processed)
            else:
                # PE: use PerceptionEncoder transforms (requires PIL input)
                preprocess = transforms.get_image_transform(self.image_size)
                processed_frames = []
                for i in range(frames.shape[0]):
                    pil_img = Image.fromarray(frames[i].numpy())
                    processed = preprocess(pil_img)
                    processed_frames.append(processed)
            
            # Stack into (Batch, C, H, W)
            pixel_values = torch.stack(processed_frames, dim=0)
            # --- PREPROCESSING END ---
            
            # Return fully preprocessed tensors
            return video_idx, v_id, batch_order, pixel_values
            
        except _DecordTimeoutError:
            print(f"⚠️  TIMEOUT ({self.timeout}s): Skipping video {v_id} ({video_path}) batch {batch_order}")
            return None
        except Exception as e:
            print(f"Failed to process frames from {video_path} batch {batch_order}: {e}")
            return None
        finally:
            if video_reader is not None:
                del video_reader


def sample_frames_clips(start: int, end: int, vlen: int, acc_samples: int) -> List[int]:
    start = max(0, start)
    end = min(vlen, end)
    intervals = np.linspace(start=start, stop=end, num=int(acc_samples) + 1).astype(int)
    frame_idxs = [(intervals[i] + intervals[i + 1] - 1) // 2 for i in range(len(intervals) - 1)]
    return frame_idxs


class VideoDataset:
    # Common video extensions supported by decord
    COMMON_VIDEO_EXTENSIONS = ["mp4", "avi", "mkv", "mov", "webm", "flv", "wmv", "m4v", "mpeg", "mpg"]
    
    def __init__(self, root: str, video_fmt: Optional[str] = None, video_uids: Optional[List[str]] = None, default_fps: float = 30.0, read_fps: float = 1.0, cache_dir: Optional[str] = None, metadata_cache_file: Optional[str] = None, cache_filename: Optional[str] = None):
        self.root = root
        self.default_fps = default_fps  
        self.fps_to_read = read_fps
        self.video_fmt = video_fmt  # Can be None for auto-detection
        
        # Find video files
        if video_fmt:
            # Single format specified
            self.video_paths = sorted(glob.glob(os.path.join(root, f"*.{video_fmt}")))
        else:
            # Search for all common video formats
            self.video_paths = []
            for ext in self.COMMON_VIDEO_EXTENSIONS:
                self.video_paths.extend(glob.glob(os.path.join(root, f"*.{ext}")))
                self.video_paths.extend(glob.glob(os.path.join(root, f"*.{ext.upper()}")))
            self.video_paths = sorted(set(self.video_paths))

        if video_uids is not None:
            video_uids_set = set(video_uids)
            self.video_paths = [vp for vp in self.video_paths if os.path.splitext(os.path.basename(vp))[0] in video_uids_set]
        
        self.video_metadata = {}
        
        self.external_cache_file = metadata_cache_file
        if metadata_cache_file and os.path.exists(metadata_cache_file):
            self._load_metadata_cache(metadata_cache_file)
        
        if cache_filename:
            self.cache_file = cache_filename
        else:
            cache_location = cache_dir if cache_dir else root
            cache_suffix = video_fmt if video_fmt else "all"
            self.cache_file = os.path.join(cache_location, f".video_metadata_cache_{cache_suffix}.json")
        
        if os.path.exists(self.cache_file) and self.cache_file != metadata_cache_file:
            self._load_metadata_cache(self.cache_file, merge=True)

    def _load_metadata_cache(self, cache_path: str = None, merge: bool = False):
        path = cache_path or self.cache_file
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    loaded = json.load(f)
                for key, value in loaded.items():
                    if isinstance(value, int):
                        loaded[key] = {"frames": value, "fps": self.default_fps}
                if merge:
                    for key, value in loaded.items():
                        if key not in self.video_metadata:
                            self.video_metadata[key] = value
                else:
                    self.video_metadata = loaded
            except Exception as e:
                print(f"Warning: Failed to load cache from {path}: {e}")
    
    def _save_metadata_cache(self):
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.video_metadata, f, indent=2)
            print(f"Cache saved to: {self.cache_file}")
        except Exception as e:
            print(f"Warning: Failed to save cache to {self.cache_file}: {e}")
    
    def get_video_metadata(self, idx: int) -> dict:
        video_path = self.video_paths[idx]
        video_name = os.path.basename(video_path)
        
        if video_name in self.video_metadata:
            meta = self.video_metadata[video_name]
            if isinstance(meta, int):
                return {"frames": meta, "fps": self.default_fps}
            return meta
        
        try:
            vr = decord.VideoReader(video_path, num_threads=1)
            vlen = len(vr)
            fps = vr.get_avg_fps()
            del vr
            self.video_metadata[video_name] = {"frames": vlen, "fps": round(fps, 3)}
            return self.video_metadata[video_name]
        except:
            return {"frames": 0, "fps": self.default_fps}

    def __len__(self) -> int:
        return len(self.video_paths)

    def get_video_name(self, idx: int) -> str:
        base_name = os.path.basename(self.video_paths[idx])
        # Use splitext to handle any extension (more robust than relying on video_fmt)
        return os.path.splitext(base_name)[0]


def worker(gpu_id: int, args, dataset: VideoDataset, video_indices: List[int], processed_count: mp.Value, num_gpus: int, status_dict=None):
    import time as time_module
    
    device = torch.device(f"cuda:{gpu_id}")
    
    # Init Model based on encoder choice
    if args.encoder == "dinov3":
        net = DINOv3FeatureExtractor(model_name=args.dinov3_model)
    else:
        net = PEFeatureExtractor(config_name=args.model_config)
    net = net.to(device)
    net.eval()
    
    # Filter out already-processed videos (using atomic checks would represent them as .pt files)
    videos_to_process = []
    videos_already_done = 0
    for video_idx in video_indices:
        v_id = dataset.get_video_name(video_idx)
        save_path = f"{args.save_dir}/{v_id}.pt"
        if os.path.isfile(save_path):
            try:
                # Basic validity check
                torch.load(save_path, weights_only=True)
                videos_already_done += 1
                if status_dict is not None:
                    status_dict[v_id] = "already_done"
                continue
            except Exception:
                pass
        videos_to_process.append(video_idx)
    
    with processed_count.get_lock():
        processed_count.value += videos_already_done
    
    if len(videos_to_process) == 0:
        return
    
    # Process videos in chunks
    chunk_size = len(videos_to_process) if len(videos_to_process) <= 100 else 50
    for chunk_start in range(0, len(videos_to_process), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(videos_to_process))
        video_chunk = videos_to_process[chunk_start:chunk_end]
        
        # Log video names in this chunk
        chunk_video_names = [dataset.get_video_name(vi) for vi in video_chunk]
        print(f"GPU {gpu_id}: Processing chunk {chunk_start//chunk_size + 1}, "
              f"videos {chunk_start+1}-{chunk_end}/{len(videos_to_process)}: "
              f"{chunk_video_names[:3]}{'...' if len(chunk_video_names) > 3 else ''}")
        
        # Dataset
        stream_dataset = VideoStreamDataset(
            dataset, 
            video_chunk, 
            args.batch_size, 
            net.image_size,
            args.decord_threads,
            timeout=args.timeout,
            encoder=args.encoder
        )
        
        if len(stream_dataset) == 0:
            # All videos in this chunk had unreadable/empty frames
            if status_dict is not None:
                for v_idx in video_chunk:
                    v_id = dataset.get_video_name(v_idx)
                    status_dict[v_id] = "failed"
            continue
        
        # Build map of expected batches
        video_batch_counts = {}
        for item in stream_dataset.index_map:
            v_idx = item["video_idx"]
            video_batch_counts[v_idx] = video_batch_counts.get(v_idx, 0) + 1
        
        # Videos not in stream_dataset (vlen=0 or unreadable) → failed
        if status_dict is not None:
            for v_idx in video_chunk:
                if v_idx not in video_batch_counts:
                    v_id = dataset.get_video_name(v_idx)
                    status_dict[v_id] = "failed"
        
        # Optimal workers calculation
        cpu_count = py_mp.cpu_count() or 8
        workers_per_gpu = max(2, min(args.dataloader_num_workers, cpu_count // num_gpus))
        num_workers = workers_per_gpu
        
        # Loader
        loader = DataLoader(
            stream_dataset,
            batch_size=1, # Important: Batching is handled inside Dataset logic, effectively. 
                          # Each yield from Dataset is one batch of frames (e.g. 64 frames).
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_keep_valid,
            pin_memory=True, # Critical for high throughput
            persistent_workers=(num_workers > 0),
            multiprocessing_context=(py_mp.get_context("spawn") if num_workers > 0 else None),
            prefetch_factor=4 if num_workers > 0 else None,
        )
        
        video_chunks = {} 
        
        # --- INFERENCE LOOP ---
        try:
            for items in loader:
                if not items:
                    continue
                
                # Unpack batch (items is a list due to collate_keep_valid)
                (v_idx, v_id, batch_order, pixel_values) = items[0]
                v_idx = int(v_idx)
                
                # pixel_values is (Batch, C, H, W) - already preprocessed!
                # Non-blocking transfer is faster with pin_memory=True
                pixel_values = pixel_values.to(device=device, dtype=torch.float16, non_blocking=True)
                
                try:
                    with torch.inference_mode(), torch.amp.autocast('cuda'):
                        batch_feats = net(pixel_values).detach().cpu()
                    
                    if v_idx not in video_chunks:
                        video_chunks[v_idx] = {}
                    video_chunks[v_idx][int(batch_order)] = batch_feats
                    
                    # Check if video is complete
                    if len(video_chunks[v_idx]) == video_batch_counts[v_idx]:
                        # Reassemble
                        ordered = [video_chunks[v_idx][k] for k in sorted(video_chunks[v_idx].keys())]
                        feats = torch.cat(ordered, dim=0)
                        
                        save_path = f"{args.save_dir}/{v_id}.pt"
                        
                        # --- ATOMIC SAVE ---
                        atomic_save(feats, save_path)
                        print(f"GPU {gpu_id}: ✓ Saved {v_id} ({feats.shape[0]} frames)")
                        
                        if status_dict is not None:
                            status_dict[v_id] = "succeeded"
                        
                        # Cleanup
                        del video_chunks[v_idx]
                        
                        with processed_count.get_lock():
                            processed_count.value += 1
                            
                except Exception as e:
                    print(f"Error processing batch {batch_order} for {v_id} on GPU {gpu_id}: {e}")
                    continue
                    
        except Exception as e:
            print(f"Loader error on GPU {gpu_id}: {e}")

        # Any video still in video_chunks after the loop had incomplete/missing batches
        if status_dict is not None:
            for v_idx in video_chunks:
                v_id = dataset.get_video_name(v_idx)
                status_dict[v_id] = "failed"

        del video_chunks
        del loader
        gc.collect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos_root", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--video_format", default=None, type=str,
                        help="Video file extension (optional - if not provided, searches for all common video formats)")
    parser.add_argument("--video_metadata_cache", default=None, type=str)
    parser.add_argument("--video_uids_json", default=None, type=str, help="JSON file containing video uids to process")
    parser.add_argument("--encoder", default="pe", choices=["pe", "dinov3"],
                        help="Encoder backbone to use: 'pe' (PerceptionEncoder) or 'dinov3' (DINOv3 via HuggingFace)")
    parser.add_argument("--model_config", default="PE-Core-L14-336", type=str,
                        help="PE model config name (only used when --encoder=pe)")
    parser.add_argument("--dinov3_model", default="facebook/dinov3-vitl16-pretrain-lvd1689m", type=str,
                        help="DINOv3 HuggingFace model name (only used when --encoder=dinov3)")
    parser.add_argument("--read_fps", default=2.0, type=float, help="fps to read from video for feature extraction")
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--decord_threads", default=1, type=int, help="Internal threads per decord reader (keep low)")
    parser.add_argument("--dataloader_num_workers", default=4, type=int, help="CPU workers per GPU")
    parser.add_argument("--shuffle_videos", action="store_true")
    parser.add_argument("--shuffle_seed", default=None, type=int)
    parser.add_argument("--fps", default=30.0, type=float, help="fall back fps if not found in metadata")
    parser.add_argument("--timeout", default=120, type=int, help="Timeout in seconds for reading a single video (0=no timeout)")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Load UIDs
    if args.video_uids_json and os.path.exists(args.video_uids_json):
        with open(args.video_uids_json, "r") as f:
            video_uids = json.load(f)
    else:
        video_uids = None

    # Derive metadata cache path from the video_uids_json filename, saved in video_lengths_cache/
    uid_cache_filename = None
    if args.video_uids_json:
        uid_stem = os.path.splitext(os.path.basename(args.video_uids_json))[0]
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cache_dir = os.path.join(script_dir, "video_lengths_cache")
        os.makedirs(cache_dir, exist_ok=True)
        uid_cache_filename = os.path.join(cache_dir, f"{uid_stem}.json")

    dataset = VideoDataset(
        args.videos_root, 
        args.video_format, 
        video_uids=video_uids, 
        default_fps=args.fps, 
        read_fps=args.read_fps, 
        cache_dir=args.save_dir,
        metadata_cache_file=args.video_metadata_cache,
        cache_filename=uid_cache_filename,
    )
    print('Dataloader Loading Complete')

    # Detect video IDs from the list that have no matching file in videos_root
    not_found = []
    if video_uids is not None:
        found_ids = set(dataset.get_video_name(i) for i in range(len(dataset)))
        not_found = [uid for uid in video_uids if uid not in found_ids]
    
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs available.")
    
    print(f"Found {len(dataset)} videos, using {num_gpus} GPUs")
    
    # --- METADATA PRE-CACHING (Parallel) ---
    uncached_paths = [
        dataset.video_paths[i] 
        for i in range(len(dataset)) 
        if os.path.basename(dataset.video_paths[i]) not in dataset.video_metadata
    ]
    
    if len(uncached_paths) > 0:
        print(f"Pre-caching {len(uncached_paths)} video metadata entries...")
        num_cache_workers = min(32, py_mp.cpu_count() or 8)
        
        with ProcessPoolExecutor(max_workers=num_cache_workers) as executor:
            futures = {executor.submit(_get_video_metadata_worker, path): path for path in uncached_paths}
            with tqdm.tqdm(total=len(uncached_paths), desc="Caching metadata") as pbar:
                for future in as_completed(futures):
                    video_name, metadata = future.result()
                    if metadata is not None:
                        dataset.video_metadata[video_name] = metadata
                    pbar.update(1)
        dataset._save_metadata_cache()
    
    # Shuffle and Split
    video_indices = list(range(len(dataset)))
    if args.shuffle_videos:
        rng = np.random.default_rng(args.shuffle_seed)
        video_indices = rng.permutation(video_indices).tolist()

    mp.set_start_method("spawn", force=True)

    manager = mp.Manager()
    status_dict = manager.dict()

    processed_count = mp.Value("i", 0)
    split_indices = [video_indices[i::num_gpus] for i in range(num_gpus)]
    processes = []
    
    model_desc = args.dinov3_model if args.encoder == "dinov3" else args.model_config
    with tqdm.tqdm(total=len(dataset), desc=f"Processing videos ({model_desc})") as pbar:
        for gpu_id in range(num_gpus):
            p = mp.Process(target=worker, args=(gpu_id, args, dataset, split_indices[gpu_id], processed_count, num_gpus, status_dict))
            p.start()
            processes.append(p)
            time.sleep(2) # Stagger start

        while any(p.is_alive() for p in processes):
            with processed_count.get_lock():
                pbar.n = processed_count.value
            pbar.refresh()
            time.sleep(1)
        
        # Final update
        with processed_count.get_lock():
            pbar.n = processed_count.value
        pbar.refresh()

    for p in processes:
        p.join()

    # --- EXTRACTION SUMMARY ---
    final_status = dict(status_dict)
    manager.shutdown()

    succeeded    = sorted(v for v, s in final_status.items() if s == "succeeded")
    already_done = sorted(v for v, s in final_status.items() if s == "already_done")
    failed       = sorted(v for v, s in final_status.items() if s == "failed")

    print("\n" + "=" * 60)
    print("EXTRACTION SUMMARY")
    print("=" * 60)
    if video_uids is not None:
        print(f"  Total in video_uids list : {len(video_uids)}")
        print(f"  Not found in videos_root : {len(not_found)}")
    print(f"  Already done (skipped)   : {len(already_done)}")
    print(f"  Newly succeeded          : {len(succeeded)}")
    print(f"  Failed                   : {len(failed)}")
    print(f"  Total processed          : {len(already_done) + len(succeeded)}")

    if not_found:
        print(f"\nNot found ({len(not_found)}):")
        for v in not_found:
            print(f"    - {v}")
    if failed:
        print(f"\nFailed ({len(failed)}):")
        for v in failed:
            print(f"    - {v}")
    print("=" * 60)


if __name__ == "__main__":
    main()