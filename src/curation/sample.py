import json5
import logging
import random

from datasets import load_from_disk, concatenate_datasets

from .arguments import get_args

DATASET_FILTERING_KEYWORDS = ["multimodal", "image", "audio", "video", "speech", "vqa", "modal", "cot", "icl", "average", "mean", "all", "overall", "dev", "test", "valid"]
METRIC_FILTERING_KEYWORDS = ["llm", "judge", 'gpt', 'claude', 'gemini']
SAMPLING_NUMBER = 4

def _apply_filter_and_log(ds_obj, filter_fn, description: str):
    before = len(ds_obj)
    ds_after = ds_obj.filter(filter_fn)
    after = len(ds_after)
    logging.info("%s: %d -> %d", description, before, after)
    return ds_after

def main():
    args = get_args()
    logging.basicConfig(level=getattr(logging, args.loglevel.upper(), logging.INFO), format='%(asctime)s %(levelname)s: %(message)s')
    logging.info("Loading dataset from %s", args.hf_ds)
    ds = load_from_disk(args.hf_ds)

    m = 0
    n = 0
    for i in range(len(ds)):
        if ds[i]['dataset'] is None:
            m += 1
        if ds[i]['subset'] is None:
            n += 1
    
    logging.info("Found %d examples with dataset=None and %d examples with subset=None", m, n)

    def convert_none_to_xx(example):
        if example['dataset'] is None:
            example['dataset'] = "xx"
        if example['subset'] is None:
            example['subset'] = "xx"
        return example

    ds = ds.map(convert_none_to_xx)

    ds = _apply_filter_and_log(ds, lambda example: float(example['metric_value']) > 0.0 and float(example['metric_value']) < 100.0, "Filter: metric value in valid range")
    ds = _apply_filter_and_log(ds, lambda example: example['dataset'] != 'xx', "Filter: dataset != 'xx'")
    ds = _apply_filter_and_log(ds, lambda example: not any(keyword in example['subset'].strip().lower() for keyword in DATASET_FILTERING_KEYWORDS), "Filter: remove blacklisted keywords in subset")
    ds = _apply_filter_and_log(ds, lambda example: not any(keyword in example['dataset'].strip().lower() for keyword in DATASET_FILTERING_KEYWORDS), "Filter: remove blacklisted keywords in dataset")
    ds = _apply_filter_and_log(ds, lambda example: not any(keyword in json5.loads(example['initial_extracted_dict'])['metric'].strip().lower() for keyword in METRIC_FILTERING_KEYWORDS), "Filter: remove blacklisted keywords in metric")
    
    ds_subset_benchmark = _apply_filter_and_log(ds, lambda example: example['table_source_arxiv_id'] == example['data_source_arxiv_id'] , "Filter: benchmark papers")
    ds_subset_experiments = _apply_filter_and_log(ds, lambda example: example['table_source_arxiv_id'] != example['data_source_arxiv_id'] , "Filter: experimental papers")

    rng = random.Random(42)

    ### First, process the benchmark subset
    
    # Group indices by (data_source_arxiv_id)
    groups = {}
    for i, ex in enumerate(ds_subset_benchmark):
        key = (ex['data_source_arxiv_id'])
        groups.setdefault(key, []).append(i)

    total_selected_indices = []
    for key, indices in groups.items():
        # condition 1: model GPT4 family
        selected_indices = []
        for idx in indices:
            # idx refers to index within ds_subset_benchmark
            model = ds_subset_benchmark[idx].get('model_name') if isinstance(ds_subset_benchmark[idx], dict) else None
            if model in ['GPT-4o', 'GPT-4']:
                selected_indices.append(idx)

        if len(selected_indices) == 0:
            continue

        # condition 2: number of shots less than 10
        new_selected_indices = []
        for idx in selected_indices:
            shots = ds_subset_benchmark[idx].get('number_of_shots', None)
            try:
                num_shots = int(shots) if shots is not None else 1
            except Exception:
                num_shots = 1
            if num_shots < 10:
                new_selected_indices.append(idx)

        if len(new_selected_indices) == 0:
            continue

        if len(new_selected_indices) <= SAMPLING_NUMBER:
            total_selected_indices.extend(new_selected_indices)
            continue
        else:
            # try to maximize diversity over the 'subset' field: prefer picking two examples from different subsets
            subset_buckets = {}
            for idx in new_selected_indices:
                s = ds_subset_benchmark[idx].get('subset', 'xx')
                s_key = s.strip().lower() if isinstance(s, str) else str(s)
                subset_buckets.setdefault(s_key, []).append(idx)

            if len(subset_buckets) >= SAMPLING_NUMBER:
                # pick two different subsets at random, then pick one example from each
                chosen_subsets = rng.sample(list(subset_buckets.keys()), k=SAMPLING_NUMBER)
                selected_pair = []
                for s in chosen_subsets:
                    selected_pair.append(rng.choice(subset_buckets[s]))
                new_selected_indices = selected_pair
            else:
                # not enough subset diversity: fall back to uniform sampling
                new_selected_indices = rng.sample(new_selected_indices, k=SAMPLING_NUMBER)
            total_selected_indices.extend(new_selected_indices)
            continue
        
    logging.info("Grouped into %d groups; benchmark size %d -> sampled %d", len(groups), len(ds_subset_benchmark), len(total_selected_indices))
    ds_subset_benchmark_sampled = ds_subset_benchmark.select(total_selected_indices)

    ### Now apply analogous sampling to the experimental subset

    # Group indices by data_source_arxiv_id
    groups_exp = {}
    for i, ex in enumerate(ds_subset_experiments):
        key = ex['data_source_arxiv_id']
        groups_exp.setdefault(key, []).append(i)

    total_selected_indices_exp = []
    # reuse rng from above
    for key, indices in groups_exp.items():
        # condition 1: model GPT4 family
        selected_indices = []
        for idx in indices:
            model = ds_subset_experiments[idx].get('model_name') if isinstance(ds_subset_experiments[idx], dict) else None
            if model in ['GPT-4o', 'GPT-4']:
                selected_indices.append(idx)

        if len(selected_indices) == 0:
            continue

        # condition 2: number of shots less than 10
        new_selected_indices = []
        for idx in selected_indices:
            shots = ds_subset_experiments[idx].get('number_of_shots', None)
            try:
                num_shots = int(shots) if shots is not None else 1
            except Exception:
                num_shots = 1
            if num_shots < 10:
                new_selected_indices.append(idx)

        if len(new_selected_indices) == 0:
            continue

        if len(new_selected_indices) <= SAMPLING_NUMBER:
            total_selected_indices_exp.extend(new_selected_indices)
            continue
        else:
            # hierarchical sampling: first prefer distinct table_source_arxiv_id, then prefer distinct subset
            table_buckets = {}
            for idx in new_selected_indices:
                tid = ds_subset_experiments[idx].get('table_source_arxiv_id', 'xx')
                tid_key = str(tid)
                table_buckets.setdefault(tid_key, []).append(idx)

            if len(table_buckets) >= SAMPLING_NUMBER:
                # pick SAMPLING_NUMBER distinct table ids
                chosen_tids = rng.sample(list(table_buckets.keys()), k=SAMPLING_NUMBER)
                selected_pair = []
                used_subsets = set()
                for tid in chosen_tids:
                    candidates = table_buckets[tid]
                    # prefer a candidate whose subset is not yet used
                    chosen_idx = None
                    for c in candidates:
                        s = ds_subset_experiments[c].get('subset', 'xx')
                        s_key = s.strip().lower() if isinstance(s, str) else str(s)
                        if s_key not in used_subsets:
                            chosen_idx = c
                            used_subsets.add(s_key)
                            break
                    # if none with a new subset, pick a random candidate
                    if chosen_idx is None:
                        chosen_idx = rng.choice(candidates)
                        s = ds_subset_experiments[chosen_idx].get('subset', 'xx')
                        s_key = s.strip().lower() if isinstance(s, str) else str(s)
                        used_subsets.add(s_key)
                    selected_pair.append(chosen_idx)
                new_selected_indices = selected_pair
            else:
                # fallback: try to maximize subset diversity across remaining candidates
                subset_buckets = {}
                for idx in new_selected_indices:
                    s = ds_subset_experiments[idx].get('subset', 'xx')
                    s_key = s.strip().lower() if isinstance(s, str) else str(s)
                    subset_buckets.setdefault(s_key, []).append(idx)

                if len(subset_buckets) >= SAMPLING_NUMBER:
                    chosen_subsets = rng.sample(list(subset_buckets.keys()), k=SAMPLING_NUMBER)
                    selected_pair = []
                    for s in chosen_subsets:
                        selected_pair.append(rng.choice(subset_buckets[s]))
                    new_selected_indices = selected_pair
                else:
                    # ultimate fallback: uniform sampling
                    new_selected_indices = rng.sample(new_selected_indices, k=SAMPLING_NUMBER)

            total_selected_indices_exp.extend(new_selected_indices)
            continue

    logging.info("Grouped experimental into %d groups; experimental size %d -> sampled %d",
                len(groups_exp), len(ds_subset_experiments), len(total_selected_indices_exp))
    ds_subset_experimental_sampled = ds_subset_experiments.select(total_selected_indices_exp)

    ds = concatenate_datasets([ds_subset_benchmark_sampled, ds_subset_experimental_sampled])
    ds = ds.shuffle(seed=args.seed if hasattr(args, 'seed') else 42)
    logging.info("Final sampled dataset size: %d", len(ds))
    ds.save_to_disk(args.processed_data_dir)
    logging.info("Saved dataset with venues to %s", args.processed_data_dir)

if __name__ == "__main__":
    main()
