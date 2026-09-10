FIPS=$(python -c "print(*[f'18{i:03d}' for i in range(1,184,2)])")

python run_experiment.py \
  --fips $FIPS \
  --levels tract \
  --replicates 20 \
  --p-values 0.05:0.50:0.05 