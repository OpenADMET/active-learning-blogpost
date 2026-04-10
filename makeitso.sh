#!/usr/bin/env bash
CONFIGS=(
    config/pxr_chemeleon_chembl_config.yaml
    config/pxr_chemeleon_config.yaml
    config/pxr_chemprop_chembl_config.yaml
    config/pxr_chemprop_config.yaml
    config/asap_chemeleon_chembl_config.yaml
    config/asap_chemeleon_config.yaml
    config/asap_chemprop_chembl_config.yaml
    config/asap_chemprop_config.yaml
)

# Step 1: setup (GTM embedding + split) — one per config, run serially before dispatching jobs
for cfg in "${CONFIGS[@]}"; do
    python run.py --config "${cfg}" --setup-only
done

# Step 2: one job per (config × strategy × seed)
for cfg in "${CONFIGS[@]}"; do
    name=$(basename "${cfg}" _config.yaml)
    for strat in EI UCB Random Exploitation Exploration Diversity; do
        for seed in 42 43 44 45 46; do
            sbatch --job-name=al_${name}_${strat}_${seed} --gres=gpu:1 --partition=gpu --mem=32G --time=8:00:00 --ntasks-per-node=1 \
                   --wrap="python run.py --config ${cfg} --strategy ${strat} --seed ${seed}"
        done
    done
done

# Step 3: generate figures (after all jobs finish)
# for cfg in "${CONFIGS[@]}"; do
#     python analysis.py --config "${cfg}"
# done
