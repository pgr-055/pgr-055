#!/bin/bash
set -euo pipefail

# === Load modules and activate environment ===
ITER_NUM=0
MAX_ITERS=5

# === Define directories and configuration ===
ROOT_DIR=/storage/cms/wangyy_lab/pjg5318/nanobody_design/nanobody
SCRIPTS_DIR=/storage/cms/wangyy_lab/pjg5318/nanobody_design/utils

UPDATE_FASTA_SCRIPT="$SCRIPTS_DIR/generate_mutants.py"
MSA_SCRIPT="$SCRIPTS_DIR/get_msa.py"
RELAX_SCRIPT="$SCRIPTS_DIR/relax_nb.py"
ALIGN_SCRIPT="$SCRIPTS_DIR/align_nb.py"
REMARK_SCRIPT="$SCRIPTS_DIR/parse_pdb_remarks.py"
COMBINE_SCRIPT="$SCRIPTS_DIR/combine_csv.py"
PLOT_SCRIPT="$SCRIPTS_DIR/plot_csv.py"

# === Input files ===
INPUT_DIR="$ROOT_DIR/input"
ORIG_FASTA="$INPUT_DIR/original_nanobody.fasta"
MUTATION_LIST="$INPUT_DIR/mutations.txt"
CSV_DIR="$ROOT_DIR/csv"

mkdir -p "$CSV_DIR"

while (( ITER_NUM != MAX_ITERS )); do
  echo "Starting iteration $ITER_NUM"

  ITER_DIR="$ROOT_DIR/iter_${ITER_NUM}"
  FASTA_MUT_DIR="$ITER_DIR/fastas"
  MSA_DIR="$ITER_DIR/msa"
  BOLTZ_DIR="$ITER_DIR/boltz"
  RELAXED_DIR="$ITER_DIR/relaxed"
  ALIGNED_DIR="$ITER_DIR/aligned"
  CLEAN_DIR="$ITER_DIR/clean"

  mkdir -p "$ITER_DIR" "$FASTA_MUT_DIR" "$MSA_DIR" "$BOLTZ_DIR" "$RELAXED_DIR" "$ALIGNED_DIR" "$CLEAN_DIR"

  # === Step 1: Generate mutant FASTAs ===
  echo "Generating mutant FASTA files..."

  module load conda
  conda activate bpy

  python "$UPDATE_FASTA_SCRIPT" \
    --input_fasta "$ORIG_FASTA" \
    --mutations "$MUTATION_LIST" \
    --output_dir "$FASTA_MUT_DIR"

  # Also include the original sequence for reference
  cp "$ORIG_FASTA" "$FASTA_MUT_DIR/original_nanobody.fasta"

  # === Step 2: MSA & Boltz structure prediction ===
  echo "Running MSA and Boltz predictions..."
  job_ids=()
  for fasta in "$FASTA_MUT_DIR"/*.fasta; do
    name=$(basename "$fasta" .fasta)
    out_subdir="$BOLTZ_DIR/$name"
    mkdir -p "$out_subdir"

    conda activate bpy

    python "$MSA_SCRIPT" \
      --fasta "$fasta" \
      --max_seq 4096 \
      --output "$MSA_DIR"

    proc_fasta="$MSA_DIR/${name}_processed.fasta"

    # 2b: Boltz diffusion on GPU via sbatch
    boltz_cmd=$(cat <<EOF
module load cuda/12.5
spack load ninja
module load conda
conda activate boltz
boltz predict "$proc_fasta" \
  --output_format pdb \
  --out_dir "$out_subdir" \
  --diffusion_samples 1 \
  --recycling_steps 10
EOF
  )

    job_id=$(sbatch --parsable \
      --partition=gpu \
      --ntasks=1 \
      --cpus-per-task=1 \
      --gres=gpu:1 \
      --mem=32G \
      --time=01:00:00 \
      --job-name=boltz_$name \
      --wrap="$boltz_cmd")
    job_ids+=($job_id)
  done

  IFS=:
  dependency_list="${job_ids[*]}"
  unset IFS

  # Wait for all Boltz jobs to finish
  sbatch --wait --dependency=afterany:$dependency_list --wrap="echo 'All Boltz predictions complete.'"

  # Recursively find every *_model_0.pdb under BOLTZ_DIR:
  find "$BOLTZ_DIR" -type f -name "*_model_0.pdb" \
       -exec cp {} "$CLEAN_DIR/" \;

  # Verify how many we got:
  count=$(find "$CLEAN_DIR" -maxdepth 1 -type f -name "*.pdb" | wc -l)
  echo "  Found ${count} model_0.pdb files in CLEAN_DIR."

  # Step 3: Relax with MPI (as before)
  echo "Relaxing predicted structures..."
  num_tasks=$(find "$CLEAN_DIR" -name "*.pdb" | wc -l)
  tasks=$((num_tasks + 1))

  relax_job_id=$(sbatch --wait --parsable \
    --nodes=1 \
    --ntasks-per-node="$tasks" \
    --time=04:00:00 \
    --job-name=relax_nb \
    --wrap="\
module load mpi; \
module load conda; \
conda activate pyros; \
export I_MPI_PMI_LIBRARY=/usr/lib64/libmi2.so; \
srun --mpi=pmi2 python \"$RELAX_SCRIPT\" \
  --struct_parent_dir \"$CLEAN_DIR\" \
  --relaxed_parent_dir \"$RELAXED_DIR\" \
  --use_cartesian 1 \
  --num_cycles 5")

  # === Step 4: Align relaxed mutants to reference ===
  echo "Aligning relaxed mutants to original nanobody..."
  ref_pdb=$(ls "$RELAXED_DIR"/original_nanobody_processed_model_0*.pdb | head -n1)
  echo " Ref PDB: $ref_pdb"
  for relaxed in "$RELAXED_DIR"/*.pdb; do
    name=$(basename "$relaxed" .pdb)
    python "$ALIGN_SCRIPT" \
      --native_pdb "$ref_pdb" \
      --pred_pdb "$relaxed" \
      --chain_id A \
      --res_start 1 \
      --res_end 140 \
      --out_pdb "$ALIGNED_DIR/${name}.pdb"
  done


  python "$REMARK_SCRIPT" \
    --relaxed_dir "$RELAXED_DIR" \
    --aligned_dir "$ALIGNED_DIR" \
    --out_csv "$CSV_DIR/summary_${ITER_NUM}.csv"

  #cat "$ROOT_DIR/slurm-*.out" > log.out
  #cat "$ROOT_DIR/slurm-*.stats" > log.stats
  #rm -f "$ROOT_DIR/slurm-*.out"
  #rm -f "$ROOT_DIR/slurm-*.stats"
  #mv "$ROOT_DIR/log.*" "$ITER_DIR"

  echo "Iteration ${ITER_NUM} completed."

  ((ITER_NUM+=1))

done

module load conda
conda activate bpy

python "$COMBINE_SCIRPT" \
  --summary_dir "$CSV_DIR" \
  --output_csv "$CSV_DIR/summary_combined.csv"

python "$PLOT_SCRIPT" \
  --combined_csv "$CSV_DIR/summary_combined.csv" \
  --out_dir "$CSV_DIR"

echo "Pipeline complete. Summary CSV and plots written to $CSV_DIR"
