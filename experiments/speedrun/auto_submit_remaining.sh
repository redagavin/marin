#!/bin/bash

# Auto-submit remaining jobs when slots are available
# Jobs to submit: llama_30m_soap_1x, qwen3_30m_muon_1x

set -e

SPEEDRUN_DIR="/scratch/yang.zih/marin_speedrun/marin/experiments/speedrun"
LOG_FILE="$SPEEDRUN_DIR/auto_submit.log"

# Jobs to submit
REMAINING_JOBS=(
    "llama_30m_soap_1x"
    "qwen3_30m_muon_1x"
)

# Track which jobs have been submitted
SUBMITTED=()

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

get_gpu_partition_job_count() {
    # Count only jobs in the GPU partition
    squeue -u yang.zih --partition=gpu --format="%T" --noheader | wc -l
}

get_running_job_count() {
    squeue -u yang.zih --partition=gpu --format="%T" --noheader | grep -c "RUNNING" || echo "0"
}

get_pending_job_count() {
    squeue -u yang.zih --partition=gpu --format="%T" --noheader | grep -c "PENDING" || echo "0"
}

is_job_submitted() {
    local job_name="$1"
    for submitted_job in "${SUBMITTED[@]}"; do
        if [[ "$submitted_job" == "$job_name" ]]; then
            return 0
        fi
    done
    return 1
}

submit_job() {
    local job_name="$1"
    local submit_script="$SPEEDRUN_DIR/$job_name/submit_slurm.sh"

    if [[ ! -f "$submit_script" ]]; then
        log "ERROR: Submit script not found: $submit_script"
        return 1
    fi

    log "Submitting job: $job_name"
    cd "$SPEEDRUN_DIR"

    if sbatch "$submit_script" 2>&1 | tee -a "$LOG_FILE"; then
        SUBMITTED+=("$job_name")
        log "Successfully submitted: $job_name"
        return 0
    else
        log "ERROR: Failed to submit: $job_name"
        return 1
    fi
}

main() {
    log "========================================"
    log "Auto-submit monitor started"
    log "Remaining jobs to submit: ${REMAINING_JOBS[*]}"
    log "========================================"

    while true; do
        # Check if all jobs have been submitted
        if [[ ${#SUBMITTED[@]} -eq ${#REMAINING_JOBS[@]} ]]; then
            log "All jobs submitted! Exiting."
            log "Submitted jobs: ${SUBMITTED[*]}"
            exit 0
        fi

        # Get current job counts in GPU partition only
        gpu_total=$(get_gpu_partition_job_count)
        running=$(get_running_job_count)
        pending=$(get_pending_job_count)

        log "Current GPU partition status: $running running, $pending pending (total: $gpu_total)"

        # QOSMaxJobsPerUserLimit indicates a limit on total GPU partition jobs
        # Currently showing 8 jobs (4 running + 4 pending) hitting the limit
        # Submit when total GPU jobs drops below limit (conservatively using 8)
        MAX_GPU_JOBS=8

        if [[ $gpu_total -lt $MAX_GPU_JOBS ]]; then
            slots_available=$((MAX_GPU_JOBS - gpu_total))
            log "GPU job slots available: $slots_available (total: $gpu_total / max: $MAX_GPU_JOBS)"

            # Try to submit remaining jobs
            for job_name in "${REMAINING_JOBS[@]}"; do
                if ! is_job_submitted "$job_name"; then
                    if submit_job "$job_name"; then
                        # Wait a bit between submissions
                        sleep 5
                        # Break to recheck status after submission
                        break
                    fi
                fi
            done
        else
            log "No GPU job slots available (total: $gpu_total / max: $MAX_GPU_JOBS). Waiting..."
        fi

        # Wait before next check
        log "Sleeping for 60 seconds before next check..."
        sleep 60
    done
}

# Run main function
main
