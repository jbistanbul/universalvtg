import argparse
import os
import shutil
import warnings

import torch
import torch.distributed as dist

warnings.filterwarnings("ignore", message=r"Detected call of `lr_scheduler.step\(\)` before `optimizer.step\(\)`")
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.*is deprecated.*")

from libs import load_opt
from libs.dist_utils import print0


def _parse_and_validate_launcher_env():
    required = ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT")
    missing = [k for k in required if os.environ.get(k) is None]
    if missing:
        raise RuntimeError(
            "Missing required launcher env vars for multi-node mode: "
            f"{missing}. Launch with torchrun (or srun+torchrun)."
        )

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    master_addr = os.environ["MASTER_ADDR"]
    master_port = os.environ["MASTER_PORT"]

    if rank < 0 or world_size < 1 or local_rank < 0:
        raise RuntimeError(
            f"Invalid distributed env values: rank={rank}, world_size={world_size}, "
            f"local_rank={local_rank}."
        )
    if not master_addr or not master_port:
        raise RuntimeError("MASTER_ADDR and MASTER_PORT must be non-empty.")

    return rank, world_size, local_rank


def _prepare_experiment_root(name: str, opt_relpath: str, rank: int):
    os.makedirs("experiments", exist_ok=True)
    root = os.path.join("experiments", name)
    opt_dst = os.path.join(root, "opt.yaml")

    if rank == 0:
        os.makedirs(root, exist_ok=True)
        if not os.path.exists(opt_dst):
            opt_src = os.path.join("opts", opt_relpath)
            if not os.path.exists(opt_src):
                raise FileNotFoundError(f"Option file not found: {opt_src}")
            shutil.copyfile(opt_src, opt_dst)
        os.makedirs(os.path.join(root, "models"), exist_ok=True)
        os.makedirs(os.path.join(root, "states"), exist_ok=True)

    # Rank-0-only filesystem writes must complete before trainer construction.
    dist.barrier()
    return root, opt_dst


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt", type=str, required=True, help="training options relative path under opts/")
    parser.add_argument("--name", type=str, required=True, help="job name")
    args = parser.parse_args()

    import libs.worker as worker_mod
    import libs.worker_multinode as worker_multinode_mod

    rank, world_size, local_rank = _parse_and_validate_launcher_env()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for train_multinode.py")

    torch.cuda.set_device(local_rank)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    gpu_capability = torch.cuda.get_device_capability(torch.cuda.current_device())
    if gpu_capability[0] < 8:
        os.environ["TRITON_F32_DEFAULT"] = "ieee"

    from datetime import timedelta
    # Process-group timeout (seconds). Keep finite; default 30 minutes.
    # Backward compatible with previous NCCL_TIMEOUT env.
    nccl_timeout = int(
        os.environ.get("VTG_PG_TIMEOUT_SEC", os.environ.get("NCCL_TIMEOUT", 1800))
    )
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=nccl_timeout),
        device_id=torch.device(f"cuda:{local_rank}"),
    )

    try:
        root, opt_path = _prepare_experiment_root(args.name, args.opt, rank)
        opt = load_opt(opt_path, is_training=True)

        opt["_root"] = root
        opt["_resume"] = (
            os.path.exists(os.path.join(root, "models", "last.pth"))
            and os.path.exists(os.path.join(root, "states", "last.pth"))
        )
        opt["_world_size"] = world_size
        opt["_distributed"] = world_size > 1
        opt["_rank"] = rank
        opt["_local_rank"] = local_rank

        trainer_type = opt.get("meta", {}).get("trainer_type", "TrainerMultiDatasetMultiNode")
        trainer_cls = getattr(worker_multinode_mod, trainer_type, None)
        if trainer_cls is None:
            trainer_cls = getattr(worker_mod, trainer_type, None)
        if trainer_cls is None:
            raise ValueError(
                f"Trainer class '{trainer_type}' not found. "
                "Ensure it is imported in train_multinode.py"
            )

        print0(
            f"[multinode] rank={rank}/{world_size}, local_rank={local_rank}, "
            f"trainer={trainer_cls.__name__}, root={root}"
        )
        if rank == 0:
            print0(
                "[multinode] external_eval: "
                f"enable={os.environ.get('VTG_EXTERNAL_EVAL_ENABLE', '0')}, "
                f"script={os.environ.get('VTG_EXTERNAL_EVAL_SCRIPT', '<unset>')}, "
                f"serial={os.environ.get('VTG_EXTERNAL_EVAL_SERIAL', '1')}, "
                f"submit_mode={os.environ.get('VTG_EXTERNAL_EVAL_SUBMIT_MODE', 'local')}"
            )
        trainer = trainer_cls(opt)
        trainer.run()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
