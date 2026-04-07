# Step 1: scaffold split + GTM embedding (fast, single node)
# python run.py --setup-only

# Step 2: one job per pair
for strat in EI UCB Random Exploitation Exploration Diversity; do
    for seed in 42 43 44 45 46; do
        for split in scaffold random; do
            sbatch --job-name=al_${split}_${strat}_${seed} --gres=gpu:1 --partition=gpu --mem=32G --time=8:00:00 --ntasks-per-node=1 \
                   --wrap="python run.py --strategy ${strat} --seed ${seed} --split ${split}"
        done
    done
done

# Step 3: generate figures (after all jobs finish)
# python analysis.py
