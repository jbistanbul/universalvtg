"""
Evaluate trained UniversalVTG models using a YAML config for dataset selection.

The evaluator supports either:
- a local experiment name under `experiments/<name>`, or
- an explicit external experiment root containing `opt.yaml` and `models/<ckpt>.pth`.

Examples:
    python eval_from_config.py \
        --name my_release_run \
        --config opts/eval/multidata_evaluation.yaml \
        --ckpt best

    python eval_from_config.py \
        --experiment_root /path/to/experiment_folder \
        --config opts/eval/multidata_evaluation.yaml \
        --ckpt best
"""

import argparse
import math
import os
import copy
import json
import tempfile
import torch
import yaml
import numpy as np
from datetime import datetime

from libs import load_opt

if torch.cuda.is_available():
    gpu_id = torch.cuda.current_device()
    gpu_capability = torch.cuda.get_device_capability(gpu_id)
else:
    gpu_capability = None
if gpu_capability is not None and gpu_capability[0] < 8:
    os.environ["TRITON_F32_DEFAULT"] = "ieee"


def preprocess_experiment_config(opt_path):
    """
    Multi-dataset experiment configs use train.datasets / eval.datasets instead
    of train.data / eval.data.  load_opt's _update_opt expects the latter, so
    we extract the first dataset's data block into train.data / eval.data and
    write a temporary YAML that load_opt can handle.

    Returns (processed_path, is_multidataset).
    """
    with open(opt_path) as f:
        raw = yaml.load(f, Loader=yaml.FullLoader)

    is_multi = (
        'datasets' in raw.get('eval', {})
        or 'datasets' in raw.get('train', {})
    )
    if not is_multi:
        return opt_path, False

    print("[INFO] Multi-dataset experiment config detected, extracting first dataset")

    if 'datasets' in raw.get('train', {}):
        raw['train']['data'] = copy.deepcopy(raw['train']['datasets'][0]['data'])
    if 'datasets' in raw.get('eval', {}):
        raw['eval']['data'] = copy.deepcopy(raw['eval']['datasets'][0]['data'])

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp:
        yaml.dump(raw, tmp)
        return tmp.name, True


def extract_eval_from_config(config_path, dataset_name=None):
    """
    Read a YAML config and return (dataset_entries, eval_config, is_multi_eval).

    - dataset_entries: list of dicts with:
        {
            "name": dataset_name,
            "data": eval.data dict,
            "eval_overrides": per-dataset eval-level overrides
        }
    - eval_config: the full eval section (ranks, nms, etc.)

    For multi-dataset configs:
      - if *dataset_name* is provided, return only that dataset.
      - if omitted, return all datasets in config order.
    """
    with open(config_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    eval_cfg = cfg.get('eval', {})

    # Single-dataset config
    if 'data' in eval_cfg and 'datasets' not in eval_cfg:
        single_name = eval_cfg['data'].get('name', 'default')
        return [
            {
                'name': single_name,
                'data': eval_cfg['data'],
                'eval_overrides': {},
            }
        ], eval_cfg, False

    # Multi-dataset config
    if 'datasets' in eval_cfg:
        datasets = eval_cfg['datasets']
        if not datasets:
            raise ValueError("Config eval.datasets is empty")

        available = [d.get('name', f"dataset_{i}") for i, d in enumerate(datasets)]

        def to_entry(ds, idx):
            if 'data' not in ds:
                raise ValueError(f"Dataset entry at index {idx} is missing 'data'")
            return {
                'name': ds.get('name', f"dataset_{idx}"),
                'data': ds['data'],
                'eval_overrides': {
                    k: v for k, v in ds.items() if k not in ('name', 'data')
                }
            }

        if dataset_name:
            for i, ds in enumerate(datasets):
                if ds.get('name') == dataset_name:
                    print(f"[INFO] Selected dataset '{dataset_name}' from config")
                    return [to_entry(ds, i)], eval_cfg, True
            raise ValueError(
                f"Dataset '{dataset_name}' not found in config. Available: {available}"
            )

        print("[INFO] No --dataset specified; evaluating all datasets sequentially")
        print(f"       Datasets: {available}")
        return [to_entry(ds, i) for i, ds in enumerate(datasets)], eval_cfg, True

    raise ValueError(
        "Config YAML must contain an 'eval' section with 'data' or 'datasets'"
    )


def override_eval_config(opt, data_config, eval_config, dataset_eval_overrides=None):
    """
    Override the loaded experiment opt's eval section with values from the
    user-provided YAML config.
    """
    opt = copy.deepcopy(opt)
    dataset_eval_overrides = dataset_eval_overrides or {}

    # Override eval.data fields from the config's data block
    for key, val in data_config.items():
        opt['eval']['data'][key] = val

    # Override eval-level settings (nms, ranks, thresholds, etc.)
    eval_level_keys = [
        'ranks', 'iou_threshs', 'pre_nms_topk', 'pre_nms_thresh',
        'seg_len_thresh', 'max_text_len', 'batchify_text_queries', 'nms',
    ]
    for key in eval_level_keys:
        if key in eval_config:
            opt['eval'][key] = eval_config[key]
        if key in dataset_eval_overrides:
            opt['eval'][key] = dataset_eval_overrides[key]

    opt['eval']['data'].setdefault('ext_score_dir', None)

    # Recompute pt_gen.max_seq_len if the new eval.data.max_vid_len requires
    # a larger point-generator buffer than what _update_opt originally computed.
    model_max_vid_len = opt['model']['vid_net']['max_seq_len']
    new_eval_max_vid_len = opt['eval']['data'].get('max_vid_len', model_max_vid_len)
    needed_n = math.ceil(new_eval_max_vid_len / model_max_vid_len)
    needed_max_seq = model_max_vid_len * needed_n
    if needed_max_seq > opt['pt_gen']['max_seq_len']:
        print(
            f"[INFO] Expanding pt_gen.max_seq_len: "
            f"{opt['pt_gen']['max_seq_len']} -> {needed_max_seq} "
            f"(eval.data.max_vid_len={new_eval_max_vid_len})"
        )
        opt['pt_gen']['max_seq_len'] = needed_max_seq

    return opt


def run_evaluation(opt, output_suffix="custom", save_predictions=False):
    """Run evaluation and return a results dict."""
    import libs.worker as worker_mod

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    evaluator_type = opt.get('meta', {}).get('evaluator_type', 'EvaluatorOriginal')
    print(f"[INFO] Using evaluator: {evaluator_type}")

    evaluator_cls = getattr(worker_mod, evaluator_type, None)
    if evaluator_cls is None:
        raise ValueError(
            f"Evaluator '{evaluator_type}' not found. "
            "Make sure it is imported from libs.worker."
        )

    evaluator = evaluator_cls(opt)
    evaluator.run()

    results = {
        'experiment': opt.get('_root', 'unknown'),
        'checkpoint': opt.get('_ckpt', 'unknown'),
        'timestamp': datetime.now().isoformat(),
    }

    if hasattr(evaluator, 'counts') and hasattr(evaluator, 'text_cnt'):
        if evaluator.text_cnt > 0:
            recalls = evaluator.counts / evaluator.text_cnt
            ranks = opt['eval'].get('ranks', [1, 5])
            iou_threshs = opt['eval'].get('iou_threshs', [0.3, 0.5])
            results['metrics'] = {}
            for i, rank in enumerate(ranks):
                for j, iou in enumerate(iou_threshs):
                    results['metrics'][f"R{rank}@{iou}"] = float(recalls[i, j])

    if save_predictions and hasattr(evaluator, 'predictions'):
        pred_file = os.path.join(
            opt['_root'],
            f"predictions_{opt['_ckpt']}_{output_suffix}.json",
        )
        with open(pred_file, 'w') as f:
            json.dump(evaluator.predictions, f, indent=2)
        print(f"[INFO] Predictions saved to: {pred_file}")
        results['predictions_file'] = pred_file

    return results

def resolve_experiment_paths(experiment_name=None, experiment_root=None, ckpt='best'):
    if bool(experiment_name) == bool(experiment_root):
        raise ValueError("Provide exactly one of --name or --experiment_root")

    if experiment_root:
        root = os.path.abspath(experiment_root)
        experiment_label = os.path.basename(root.rstrip(os.sep)) or root
    else:
        root = os.path.join('experiments', experiment_name)
        experiment_label = experiment_name

    opt_path = os.path.join(root, 'opt.yaml')
    ckpt_path = os.path.join(root, 'models', f'{ckpt}.pth')
    return root, experiment_label, opt_path, ckpt_path


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained models using a YAML config for dataset specification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--name', type=str, default=None,
        help="Experiment folder name under experiments/",
    )
    parser.add_argument(
        '--experiment_root', type=str, default=None,
        help="Path to an experiment root containing opt.yaml and models/<ckpt>.pth",
    )
    parser.add_argument(
        '--config', type=str, required=True,
        help="Path to YAML config with eval dataset paths and settings",
    )
    parser.add_argument(
        '--ckpt', type=str, default='best',
        help="Checkpoint name to evaluate (default: best)",
    )
    parser.add_argument(
        '--dataset', type=str, default=None,
        help="Dataset name to select from multi-dataset configs",
    )
    parser.add_argument(
        '--output_suffix', type=str, default='custom',
        help="Suffix for output files (default: custom)",
    )
    parser.add_argument(
        '--save_predictions', action='store_true',
        help="Save predictions to JSON file",
    )

    args = parser.parse_args()

    # ── Validate experiment paths ──────────────────────────────────────
    root, experiment_label, opt_path, ckpt_path = resolve_experiment_paths(
        experiment_name=args.name,
        experiment_root=args.experiment_root,
        ckpt=args.ckpt,
    )

    if not os.path.exists(opt_path):
        raise ValueError(f"Experiment opt.yaml not found: {opt_path}")
    if not os.path.exists(ckpt_path):
        raise ValueError(f"Checkpoint not found: {ckpt_path}")
    if not os.path.exists(args.config):
        raise ValueError(f"Config not found: {args.config}")

    # ── Extract eval settings from the provided YAML ──────────────────
    print(f"[INFO] Loading eval config from: {args.config}")
    dataset_entries, eval_config, is_multi_eval = extract_eval_from_config(
        args.config, args.dataset
    )

    for ds in dataset_entries:
        print(f"\n[INFO] Validating paths for dataset '{ds['name']}'")
        for label, path_key in [
            ('Annotation file', 'anno_file'),
            ('Text feature dir', 'text_feat_dir'),
            ('Video feature dir', 'vid_feat_dir'),
        ]:
            p = ds['data'].get(path_key)
            if p and not os.path.exists(p):
                print(f"[WARNING] {label} not found: {p}")

    # ── Load experiment config (model architecture, computed fields) ───
    print(f"[INFO] Loading experiment config from: {opt_path}")
    processed_path, is_multi = preprocess_experiment_config(opt_path)
    try:
        opt = load_opt(processed_path, is_training=False)
    finally:
        if is_multi and processed_path != opt_path:
            try:
                os.unlink(processed_path)
            except OSError:
                pass

    opt['_root'] = root
    opt['_ckpt'] = args.ckpt

    all_results = []

    # ── Run per-dataset evaluation ────────────────────────────────────
    for idx, dataset_entry in enumerate(dataset_entries):
        dataset_name = dataset_entry['name']
        data_config = dataset_entry['data']
        eval_overrides = dataset_entry.get('eval_overrides', {})

        print("\n" + "=" * 60)
        print(
            f"[INFO] Evaluating dataset {idx + 1}/{len(dataset_entries)}: "
            f"{dataset_name}"
        )
        print(f"[INFO] Evaluating checkpoint: {args.ckpt}")

        print("\n[INFO] Original eval data paths (from experiment):")
        print(f"  anno_file:     {opt['eval']['data'].get('anno_file', 'N/A')}")
        print(f"  text_feat_dir: {opt['eval']['data'].get('text_feat_dir', 'N/A')}")
        print(f"  vid_feat_dir:  {opt['eval']['data'].get('vid_feat_dir', 'N/A')}")

        dataset_opt = override_eval_config(opt, data_config, eval_config, eval_overrides)

        print("\n[INFO] Overridden eval data paths (from config):")
        print(f"  anno_file:     {dataset_opt['eval']['data'].get('anno_file', 'N/A')}")
        print(f"  text_feat_dir: {dataset_opt['eval']['data'].get('text_feat_dir', 'N/A')}")
        print(f"  vid_feat_dir:  {dataset_opt['eval']['data'].get('vid_feat_dir', 'N/A')}")
        print(f"  split:         {dataset_opt['eval']['data'].get('split', 'N/A')}")
        print(f"  iou_threshs:   {dataset_opt['eval'].get('iou_threshs', 'N/A')}")
        print(
            f"  batchify_text_queries: "
            f"{dataset_opt['eval'].get('batchify_text_queries', 'N/A')}"
        )

        dataset_suffix = args.output_suffix
        if len(dataset_entries) > 1:
            safe_name = dataset_name.replace('/', '_').replace(' ', '_')
            dataset_suffix = f"{args.output_suffix}_{safe_name}"

        results = run_evaluation(dataset_opt, dataset_suffix, args.save_predictions)
        results['dataset'] = dataset_name
        all_results.append(results)

        print("\n[DATASET SUMMARY]")
        print(f"  Dataset:     {dataset_name}")
        if 'metrics' in results:
            for metric, value in results['metrics'].items():
                print(f"    {metric}: {value:.4f}")

    # ── Aggregate summary + save JSON ─────────────────────────────────
    aggregate_metrics = {}
    metric_totals = {}
    metric_counts = {}
    for res in all_results:
        for metric_name, metric_val in res.get('metrics', {}).items():
            metric_totals[metric_name] = metric_totals.get(metric_name, 0.0) + metric_val
            metric_counts[metric_name] = metric_counts.get(metric_name, 0) + 1
    for metric_name, total in metric_totals.items():
        aggregate_metrics[metric_name] = total / metric_counts[metric_name]

    aggregate_results = {
        'experiment': experiment_label,
        'experiment_root': root,
        'config': args.config,
        'checkpoint': args.ckpt,
        'timestamp': datetime.now().isoformat(),
        'evaluated_datasets': [r['dataset'] for r in all_results],
        'per_dataset_results': all_results,
        'aggregate_metrics': aggregate_metrics,
    }

    aggregate_file = os.path.join(
        root, f"eval_aggregate_{args.ckpt}_{args.output_suffix}.json"
    )
    with open(aggregate_file, 'w') as f:
        json.dump(aggregate_results, f, indent=2)

    print("\n" + "=" * 60)
    print("[AGGREGATE SUMMARY]")
    print(f"  Experiment:  {experiment_label}")
    print(f"  Config:      {args.config}")
    if args.dataset:
        print(f"  Dataset:     {args.dataset}")
    elif is_multi_eval:
        print("  Dataset:     ALL (sequential)")
    print(f"  Checkpoint:  {args.ckpt}")
    print(f"  # Datasets:  {len(all_results)}")
    if aggregate_metrics:
        print("  Aggregate metrics:")
        for metric, value in aggregate_metrics.items():
            print(f"    {metric}: {value:.4f}")
    print(f"  Aggregate JSON: {aggregate_file}")

    return aggregate_results


if __name__ == '__main__':
    main()
