from collections import OrderedDict
from copy import deepcopy
import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import time

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

from .core import load_opt
from .dist_utils import get_rank, get_world_size, barrier
from .worker import (
    AUX_LOSS_REGISTRY,
    AverageMeter,
    Logger,
    PtGenerator,
    TrainerMultiDataset,
    fix_random_seed,
    make_dataloader,
    make_models_net,
    make_optimizer,
    make_scheduler,
)


class TrainerMultiDatasetMultiNode(TrainerMultiDataset):
    """
    Additive multi-node-safe variant of TrainerMultiDataset.

    Key difference vs original TrainerMultiDataset._init_multi_dataset:
    - DDP is bound by local CUDA rank (LOCAL_RANK), not global rank.
    """

    def _init_multi_dataset(self, opt):
        """Initialize trainer with combined multi-dataset (multi-node safe DDP)."""
        self.opt = opt
        self._last_external_eval_jobid = None
        self._external_eval_local_submit_disabled = False

        # Set random seed
        rng = fix_random_seed(opt.get('seed', 2022))

        # Build model and EMA
        self.model = make_models_net(opt).cuda()
        self.model_ema = deepcopy(self.model).eval().requires_grad_(False)
        self.pt_gen = PtGenerator(**opt['pt_gen']).cuda()
        self.ema_beta = opt['train'].get('ema_beta', 0.999)

        # Use combined dataset
        self.num_epochs = opt['train']['epochs'] + opt['train']['warmup_epochs']
        self.dataset = self.combined_dataset

        # Create a wrapper to handle set_epoch for ConcatDataset
        class ConcatDatasetWrapper:
            def __init__(self, concat_dataset, individual_datasets):
                self.concat_dataset = concat_dataset
                self.individual_datasets = individual_datasets

            def __len__(self):
                return len(self.concat_dataset)

            def __getitem__(self, idx):
                return self.concat_dataset[idx]

            def set_epoch(self, epoch):
                for ds in self.individual_datasets:
                    if hasattr(ds, 'set_epoch'):
                        ds.set_epoch(epoch)

        self.dataset = ConcatDatasetWrapper(self.combined_dataset, self.individual_train_datasets)
        self.dataset_sample_weights, self.dataset_sampling_report = \
            self._build_multi_dataset_sample_weights(opt)

        self.batch_size = batch_size = opt['train']['batch_size']
        self.dataloader, self.sampler = make_dataloader(
            self.combined_dataset, generator=rng, is_training=True,
            batch_size=batch_size, num_workers=opt['train']['num_workers'],
            world_size=get_world_size(), rank=get_rank(),
            sample_weights=self.dataset_sample_weights,
        )
        self.microbatch_size = opt['train'].get('microbatch_size', batch_size)
        self.num_microbatches = batch_size // self.microbatch_size
        assert batch_size % self.microbatch_size == 0

        # Build training utilities
        self.itrs_per_epoch = opt['train']['scheduler']['itrs_per_epoch'] = len(self.dataloader)
        self.num_itrs = self.num_epochs * self.itrs_per_epoch
        self.epoch = self.itr = 0
        self.optimizer = make_optimizer(self.model, opt['train']['optimizer'])
        self.scheduler = make_scheduler(self.optimizer, opt['train']['scheduler'])
        self.clip_grad_norm = opt['train'].get('clip_grad_norm')

        # Build logging utilities
        self.log_interval = opt['log'].get('log_interval', 100)
        self.checkpoint_epochs = opt['log'].get('checkpoint_epochs', (-1,))
        if get_rank() == 0:
            self.logger = Logger(os.path.join(opt['_root'], 'train.txt'))
            self.tb_writer = SummaryWriter(os.path.join(opt['_root'], 'tb'))
            self.loss_meters = OrderedDict()
            self.timer = AverageMeter()
            if self.dataset_sampling_report:
                self.logger.write("Dataset-weighted sampling enabled:")
                for ds_name, item in self.dataset_sampling_report.items():
                    self.logger.write(
                        f"  - {ds_name}: size={item['size']}, "
                        f"group_weight={item['group_weight']:.6g}, "
                        f"expected_prob={item['expected_prob'] * 100:.2f}%"
                    )
        else:
            self.logger = self.tb_writer = self.loss_meters = self.timer = None

        # Load checkpoint if resuming
        if opt['_resume']:
            self.load()
            barrier()

        # Set up distributed training (LOCAL_RANK-safe)
        if opt['_distributed']:
            local_rank = int(opt.get('_local_rank', os.environ.get('LOCAL_RANK', 0)))
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[local_rank],
                output_device=local_rank,
            )
            self._ema_init()

        # Register model hyperparameters
        self.max_vid_len = opt['model']['max_vid_len']
        self.max_text_len = opt['model']['max_text_len']
        self.vid_stride = opt['model'].get('vid_stride', 1)
        self.input_vid_len = self.max_vid_len * self.vid_stride

        # Register annotation hyperparameters
        self.center_sampling = opt['train'].get('center_sampling', 'radius')
        self.center_sampling_radius = opt['train']['center_sampling_radius']

        # Register optimization hyperparameters
        self.loss_norm_momentum = opt['train'].get('loss_norm_momentum', 0.9)
        self.loss_norm = opt['train']['loss_norm']
        self.loss_weight = opt['train'].get('loss_weight', 1.0)
        self.reg_loss = opt['train'].get('reg_loss', 'diou')

        # Initialize auxiliary losses (from TrainerAuxiliary)
        vid_embd_dim = opt['model']['vid_net']['embd_dim']
        self.ds_contrastive = opt['train']['loss_aux']['ds_contrast']['enable']
        self.gt_contrastive = opt['train']['loss_aux']['gt_contrast']['enable']
        self.early_fusion = opt['model'].get('early_fusion', True)
        self.use_mst = opt['model'].get('use_mst', False)

        if self.early_fusion and self.logger:
            self.logger.write("Early fusion enabled")

        if self.ds_contrastive:
            ds_loss_type = AUX_LOSS_REGISTRY[
                opt['train']['loss_aux']['ds_contrast'].get('type', 'ds_contrastive')
            ]
            self.ds_contrastive_loss = ds_loss_type(
                opt['train']['loss_aux']['ds_contrast'], vid_embd_dim
            ).cuda()
            self.ds_contrastive_weight = opt['train']['loss_aux']['ds_contrast']['weight']
        else:
            self.ds_contrastive_weight = 0.0

        if self.gt_contrastive:
            gt_loss_type = AUX_LOSS_REGISTRY[
                opt['train']['loss_aux']['gt_contrast'].get('type', 'gt_point_contrastive')
            ]
            self.gt_contrastive_loss = gt_loss_type(
                opt['train']['loss_aux']['gt_contrast'], vid_embd_dim
            ).cuda()
            self.gt_contrastive_weight = opt['train']['loss_aux']['gt_contrast']['weight']
            self.loss_aux_gt_type = opt['train']['loss_aux']['gt_contrast'].get('gt_type', 'point')
            self.loss_aux_span_radius = opt['train']['loss_aux']['gt_contrast'].get(
                'span_radius', self.center_sampling_radius
            )
            self.span_contr_gt = opt['train']['loss_aux']['gt_contrast'].get('span_contr_gt', False)
        else:
            self.gt_contrastive_weight = 0.0

        # Log dataset info
        if self.logger:
            self.logger.write("Multi-dataset training enabled (multi-node additive path):")
            for ds_name in self.train_dataset_names:
                self.logger.write(f"  - {ds_name}")
            self.logger.write(f"Combined dataset size: {len(self.combined_dataset)}")

    def _is_external_eval_enabled(self):
        val_cfg = self.opt.get("train", {}).get("validation", {})
        ext_cfg = val_cfg.get("external_eval", {})
        if bool(ext_cfg.get("enable", False)):
            return True
        env_flag = str(os.environ.get("VTG_EXTERNAL_EVAL_ENABLE", "")).lower()
        return env_flag in ("1", "true", "yes", "on")

    def _resolve_external_eval_script(self):
        val_cfg = self.opt.get("train", {}).get("validation", {})
        ext_cfg = val_cfg.get("external_eval", {})
        script = ext_cfg.get("sbatch_script") or os.environ.get("VTG_EXTERNAL_EVAL_SCRIPT")
        if script:
            return os.path.abspath(script)
        return None

    def _resolve_external_eval_submit_mode(self):
        """
        External-eval submit behavior.
        - local: submit sbatch directly from training process (legacy behavior)
        - auto: same as local; if host policy blocks sbatch, disable local submit for later epochs
        - snapshot_only: create snapshots only; submit from login-node watcher
        """
        val_cfg = self.opt.get("train", {}).get("validation", {})
        ext_cfg = val_cfg.get("external_eval", {})
        raw_mode = ext_cfg.get("submit_mode") or os.environ.get("VTG_EXTERNAL_EVAL_SUBMIT_MODE", "local")
        mode = str(raw_mode).strip().lower()

        aliases = {
            "local": "local",
            "sbatch": "local",
            "auto": "auto",
            "snapshot_only": "snapshot_only",
            "none": "snapshot_only",
            "login_watcher": "snapshot_only",
            "watcher": "snapshot_only",
        }
        return aliases.get(mode, "local")

    @staticmethod
    def _is_submit_blocked_on_host(stdout: str, stderr: str):
        msg = f"{stdout}\n{stderr}".lower()
        blocked_signals = (
            "job submission is not allowed from this host",
            "submit through one of the available login resources",
            "not available on compute nodes",
        )
        return any(sig in msg for sig in blocked_signals)

    @staticmethod
    def _extract_sbatch_jobid(output: str):
        lines = [ln.strip() for ln in output.splitlines() if ln.strip()]
        for line in reversed(lines):
            m = re.match(r"^([0-9]+)(?:;[A-Za-z0-9._-]+)?$", line)
            if m:
                return m.group(1)
            m = re.search(r"Submitted batch job\s+([0-9]+)", line, flags=re.IGNORECASE)
            if m:
                return m.group(1)
        return None

    def _link_or_copy(self, src, dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst):
            os.remove(dst)
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)

    def _create_eval_snapshot(self):
        root = os.path.abspath(self.opt["_root"])
        epoch_tag = f"epoch_{self.epoch:03d}"
        ckpt_name = epoch_tag
        snapshot_root = os.path.join(root, "eval_snapshots", epoch_tag)
        models_dir = os.path.join(snapshot_root, "models")
        states_dir = os.path.join(snapshot_root, "states")
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(states_dir, exist_ok=True)

        self._link_or_copy(os.path.join(root, "opt.yaml"), os.path.join(snapshot_root, "opt.yaml"))

        # Create immutable, epoch-named checkpoint copies for async eval jobs.
        # Keep a snapshot-local "last.pth" alias for backward compatibility.
        src_model = os.path.join(root, "models", "last.pth")
        src_state = os.path.join(root, "states", "last.pth")
        dst_model_epoch = os.path.join(models_dir, f"{ckpt_name}.pth")
        dst_state_epoch = os.path.join(states_dir, f"{ckpt_name}.pth")
        shutil.copy2(src_model, dst_model_epoch)
        shutil.copy2(src_state, dst_state_epoch)
        self._link_or_copy(dst_model_epoch, os.path.join(models_dir, "last.pth"))
        self._link_or_copy(dst_state_epoch, os.path.join(states_dir, "last.pth"))

        manifest = {
            "epoch": self.epoch,
            "itr": self.itr,
            "snapshot_root": snapshot_root,
            "created_at": int(time.time()),
            "source_root": root,
            "ckpt_name": ckpt_name,
        }
        with open(os.path.join(snapshot_root, "snapshot_manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

        return snapshot_root, ckpt_name

    def _submit_external_eval_job(self, snapshot_root, ckpt_name="last"):
        if self._external_eval_local_submit_disabled:
            if self.logger:
                self.logger.write(
                    "[external-eval] local sbatch submit disabled; "
                    "snapshot-only mode active for remaining epochs."
                )
            return None

        script = self._resolve_external_eval_script()
        if not script or not os.path.exists(script):
            if self.logger:
                self.logger.write(
                    "[external-eval] skipped: VTG_EXTERNAL_EVAL_SCRIPT is not set "
                    "or script path does not exist."
                )
            return None

        cmd = ["sbatch", "--parsable", "--export=ALL"]
        extra_args = os.environ.get("VTG_EXTERNAL_EVAL_SBATCH_ARGS", "").strip()
        if extra_args:
            cmd.extend(shlex.split(extra_args))

        serial_eval = str(os.environ.get("VTG_EXTERNAL_EVAL_SERIAL", "1")).lower() in ("1", "true", "yes", "on")
        if serial_eval:
            if self._last_external_eval_jobid:
                cmd.extend(["--dependency", f"afterany:{self._last_external_eval_jobid}"])
            else:
                last_jobid_file = os.path.join(self.opt["_root"], "external_eval_last_jobid.txt")
                if os.path.exists(last_jobid_file):
                    with open(last_jobid_file, "r") as f:
                        prev = f.read().strip()
                    if prev:
                        cmd.extend(["--dependency", f"afterany:{prev}"])

        cmd.extend([script, snapshot_root, ckpt_name])

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            if self._is_submit_blocked_on_host(proc.stdout, proc.stderr):
                self._external_eval_local_submit_disabled = True
            if self.logger:
                self.logger.write(
                    "[external-eval] sbatch submission failed:\n"
                    f"  cmd: {' '.join(cmd)}\n"
                    f"  stdout: {proc.stdout.strip()}\n"
                    f"  stderr: {proc.stderr.strip()}"
                )
                if self._external_eval_local_submit_disabled:
                    self.logger.write(
                        "[external-eval] Detected host policy blocking compute-node submission. "
                        "Keep VTG_EXTERNAL_EVAL_ENABLE=1 to keep snapshots, and run "
                        "scripts/slurm/watch_external_eval_login.py from a login node."
                    )
            return None

        jobid = self._extract_sbatch_jobid(proc.stdout)
        if not jobid:
            if self.logger:
                self.logger.write(
                    "[external-eval] sbatch submission succeeded but jobid parse failed:\n"
                    f"  cmd: {' '.join(cmd)}\n"
                    f"  stdout: {proc.stdout.strip()}\n"
                    f"  stderr: {proc.stderr.strip()}"
                )
            return None
        self._last_external_eval_jobid = jobid
        with open(os.path.join(self.opt["_root"], "external_eval_last_jobid.txt"), "w") as f:
            f.write(jobid)
        self._record_external_eval_submission(
            epoch=self.epoch,
            snapshot_root=snapshot_root,
            jobid=jobid,
            script=script,
            ckpt_name=ckpt_name,
        )
        if self.logger:
            self.logger.write(
                f"[external-eval] submitted epoch={self.epoch}, ckpt={ckpt_name}, "
                f"jobid={jobid}, snapshot={snapshot_root}"
            )
        return jobid

    def _record_external_eval_submission(self, epoch, snapshot_root, jobid, script, ckpt_name="last"):
        jobs_path = os.path.join(self.opt["_root"], "external_eval_jobs.json")
        jobs = {"jobs": {}}
        if os.path.exists(jobs_path):
            try:
                with open(jobs_path, "r") as f:
                    jobs = json.load(f)
            except Exception:
                jobs = {"jobs": {}}
        if "jobs" not in jobs:
            jobs["jobs"] = {}

        jobs["jobs"][str(epoch)] = {
            "epoch": int(epoch),
            "jobid": str(jobid),
            "snapshot_root": os.path.abspath(snapshot_root),
            "script": os.path.abspath(script),
            "ckpt_name": str(ckpt_name),
            "submitted_at": int(time.time()),
        }
        with open(jobs_path, "w") as f:
            json.dump(jobs, f, indent=2)

    def _run_validation_epoch(self):
        """
        Override validation hook:
        - Default path: run in-process validation (original behavior).
        - External-eval path: snapshot current `last.pth` and submit a separate
          SLURM eval job while training proceeds to next epoch.
        """
        if not self._is_external_eval_enabled():
            return super()._run_validation_epoch()

        if get_rank() == 0:
            snapshot_root, ckpt_name = self._create_eval_snapshot()
            submit_mode = self._resolve_external_eval_submit_mode()
            if submit_mode == "snapshot_only":
                if self.logger:
                    self.logger.write(
                        f"[external-eval] snapshot-only mode: epoch={self.epoch}, "
                        f"ckpt={ckpt_name}, snapshot={snapshot_root}"
                    )
            else:
                self._submit_external_eval_job(snapshot_root, ckpt_name=ckpt_name)

        barrier()
        torch.cuda.empty_cache()
        # External eval is asynchronous; no early stopping in this mode.
        return False


def _load_eval_checkpoint_into_trainer(trainer, ckpt_name):
    model_path = os.path.join(trainer.opt["_root"], "models", f"{ckpt_name}.pth")
    state_path = os.path.join(trainer.opt["_root"], "states", f"{ckpt_name}.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint model file not found: {model_path}")

    model_ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    model = trainer._unwrap(trainer.model)
    if "model" in model_ckpt:
        model.load_state_dict(model_ckpt["model"])
        trainer.model_ema.load_state_dict(model_ckpt.get("model_ema", model_ckpt["model"]))
    else:
        model.load_state_dict(model_ckpt)
        trainer.model_ema.load_state_dict(model_ckpt)

    if os.path.exists(state_path):
        state_ckpt = torch.load(state_path, map_location="cpu", weights_only=False)
        trainer.epoch = state_ckpt.get("epoch", trainer.epoch)
        trainer.itr = state_ckpt.get("itr", trainer.itr)


def _update_external_eval_artifacts(
    exp_root,
    snapshot_root,
    epoch,
    metric_name,
    tracked_metric,
    all_metrics,
    ckpt_name="last",
):
    models_dir = os.path.join(exp_root, "models")
    states_dir = os.path.join(exp_root, "states")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(states_dir, exist_ok=True)

    history_path = os.path.join(exp_root, "external_eval_history.json")
    tracker_path = os.path.join(exp_root, "external_eval_tracker.json")

    history = {}
    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            history = json.load(f)
    if "history" not in history:
        history["history"] = {}
    history["metric_name"] = metric_name
    history["history"][str(epoch)] = {k: float(v) for k, v in all_metrics.items()}
    history["history"][str(epoch)]["tracked_metric"] = float(tracked_metric)
    history["history"][str(epoch)]["snapshot_root"] = snapshot_root

    tracker = {
        "metric_name": metric_name,
        "best_metric": -1.0,
        "best_epoch": 0,
        "best_snapshot_root": "",
    }
    if os.path.exists(tracker_path):
        with open(tracker_path, "r") as f:
            tracker = json.load(f)

    is_better = float(tracked_metric) > float(tracker.get("best_metric", -1.0))
    if is_better:
        tracker["best_metric"] = float(tracked_metric)
        tracker["best_epoch"] = int(epoch)
        tracker["best_snapshot_root"] = snapshot_root
        src_model = os.path.join(snapshot_root, "models", f"{ckpt_name}.pth")
        src_state = os.path.join(snapshot_root, "states", f"{ckpt_name}.pth")
        if not os.path.exists(src_model):
            src_model = os.path.join(snapshot_root, "models", "last.pth")
        if not os.path.exists(src_state):
            src_state = os.path.join(snapshot_root, "states", "last.pth")
        shutil.copy2(src_model, os.path.join(models_dir, "best.pth"))
        if os.path.exists(src_state):
            shutil.copy2(src_state, os.path.join(states_dir, "best.pth"))
        history["best_epoch"] = int(epoch)
        history["best_metric"] = float(tracked_metric)
    else:
        history["best_epoch"] = int(tracker.get("best_epoch", 0))
        history["best_metric"] = float(tracker.get("best_metric", -1.0))

    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    with open(tracker_path, "w") as f:
        json.dump(tracker, f, indent=2)


def run_snapshot_eval(snapshot_root, ckpt_name="last"):
    snapshot_root = os.path.abspath(snapshot_root)
    opt_path = os.path.join(snapshot_root, "opt.yaml")
    if not os.path.exists(opt_path):
        raise FileNotFoundError(f"Snapshot opt file not found: {opt_path}")

    opt = load_opt(opt_path, is_training=True)
    opt["_root"] = snapshot_root
    opt["_resume"] = False
    opt["_world_size"] = 1
    opt["_distributed"] = False
    opt["_rank"] = 0
    opt["_local_rank"] = 0

    trainer_type = opt.get("meta", {}).get("trainer_type", "TrainerMultiDatasetMultiNode")
    trainer_cls = globals().get(trainer_type)
    if trainer_cls is None:
        raise ValueError(f"Trainer class not found for snapshot eval: {trainer_type}")

    trainer = trainer_cls(opt)
    _load_eval_checkpoint_into_trainer(trainer, ckpt_name)

    if not getattr(trainer, "validation_enabled", False):
        raise RuntimeError(
            "Validation is disabled in opt.yaml; cannot run per-epoch external eval."
        )

    if getattr(trainer, "multi_val_mode", False):
        tracked_metric, all_metrics = trainer._validate_multi_dataset()
    else:
        tracked_metric, all_metrics = trainer._validate_single_dataset(
            trainer.val_dataloader, "single"
        )

    result = {
        "epoch": int(trainer.epoch),
        "metric_name": trainer.metric_name,
        "tracked_metric": float(tracked_metric),
        "metrics": {k: float(v) for k, v in all_metrics.items()},
        "snapshot_root": snapshot_root,
        "ckpt_name": ckpt_name,
    }
    with open(os.path.join(snapshot_root, "external_eval_result.json"), "w") as f:
        json.dump(result, f, indent=2)

    # If this snapshot belongs to experiments/<name>/eval_snapshots/epoch_xxx,
    # update best-checkpoint artifacts under experiments/<name>/.
    eval_snapshots_dir = os.path.dirname(snapshot_root)
    if os.path.basename(eval_snapshots_dir) == "eval_snapshots":
        exp_root = os.path.dirname(eval_snapshots_dir)
        _update_external_eval_artifacts(
            exp_root=exp_root,
            snapshot_root=snapshot_root,
            epoch=trainer.epoch,
            metric_name=trainer.metric_name,
            tracked_metric=tracked_metric,
            all_metrics=all_metrics,
            ckpt_name=ckpt_name,
        )
    print(f"[external-eval] completed epoch={trainer.epoch} metric={trainer.metric_name} value={tracked_metric:.6f}")
    return result


def _parse_cli_args():
    parser = argparse.ArgumentParser(description="Multi-node helper utilities.")
    parser.add_argument("--eval-snapshot", type=str, default=None, help="Path to eval snapshot root.")
    parser.add_argument("--ckpt", type=str, default="last", help="Checkpoint basename under models/states.")
    return parser.parse_args()


def validate_multinode_env_or_raise():
    """Small helper for preflight checks in entrypoints/tests."""
    required = ('RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT')
    missing = [k for k in required if os.environ.get(k) is None]
    if missing:
        raise RuntimeError(
            f"Missing distributed launcher env vars: {missing}. "
            "Use torchrun/srun launcher mode; no single-node fallback in multi-node entrypoint."
        )


if __name__ == "__main__":
    cli_args = _parse_cli_args()
    if cli_args.eval_snapshot:
        run_snapshot_eval(cli_args.eval_snapshot, ckpt_name=cli_args.ckpt)
    else:
        raise SystemExit(
            "No action specified. Use --eval-snapshot <path> [--ckpt last] for external eval."
        )
