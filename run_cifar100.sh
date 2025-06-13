#!/bin/bash

# CIFAR-100 Knowledge Distillation Training Script
# Edit METHOD to switch between different distillation methods

# =============================================================================
# EDIT THESE PARAMETERS
# =============================================================================
METHOD="dkd"                          # Change to: kd, dkd, or rld
MODEL="wrn_40_2_wrn_16_2"             # Model configuration
GPU_ID="0"                            # GPU ID

# RLD specific parameters (only used when METHOD="rld")
BASE_TEMP="2"                         # 2 for hetero models, 5 for homo models  
KD_WEIGHT="9"                         # 9 for hetero models, 6 for homo models

# =============================================================================
# NO NEED TO EDIT BELOW
# =============================================================================

# Execution function
run_experiment() {
    local cmd="$1"
    echo "Running: $cmd"
    eval $cmd
    echo "----------------------------------------"
}

echo "CIFAR-100 Training - Method: $METHOD, Model: $MODEL"
echo "=========================================="

if [[ "$METHOD" == "kd" ]]; then
    # Standard Knowledge Distillation
    run_experiment "python tools/train.py --cfg configs/cifar100/kd/${MODEL}.yaml --M [1] --gpu ${GPU_ID}"
    run_experiment "python tools/train.py --cfg configs/cifar100/kd/${MODEL}.yaml --M [1] --logit-stand --gpu ${GPU_ID}"
    
    # SDD-KD experiments
    run_experiment "python tools/train.py --cfg configs/cifar100/sdd_kd/${MODEL}.yaml --M [1] --gpu ${GPU_ID}"
    run_experiment "python tools/train.py --cfg configs/cifar100/sdd_kd/${MODEL}.yaml --M [1,2] --gpu ${GPU_ID}"
    run_experiment "python tools/train.py --cfg configs/cifar100/sdd_kd/${MODEL}.yaml --M [1,2,4] --gpu ${GPU_ID}"
    run_experiment "python tools/train.py --cfg configs/cifar100/sdd_kd/${MODEL}.yaml --M [1] --logit-stand --gpu ${GPU_ID}"
    run_experiment "python tools/train.py --cfg configs/cifar100/sdd_kd/${MODEL}.yaml --M [1,2] --logit-stand --gpu ${GPU_ID}"
    run_experiment "python tools/train.py --cfg configs/cifar100/sdd_kd/${MODEL}.yaml --M [1,2,4] --logit-stand --gpu ${GPU_ID}"

elif [[ "$METHOD" == "dkd" ]]; then
    # Original DKD experiments
    run_experiment "python tools/train.py --cfg configs/cifar100/dkd/${MODEL}.yaml --M [1] --gpu ${GPU_ID}"
    run_experiment "python tools/train.py --cfg configs/cifar100/dkd/${MODEL}.yaml --M [1] --logit-stand --gpu ${GPU_ID}"
    
    # SDD-DKD experiments
    run_experiment "python tools/train.py --cfg configs/cifar100/sdd_dkd/${MODEL}.yaml --M [1] --gpu ${GPU_ID}"
    run_experiment "python tools/train.py --cfg configs/cifar100/sdd_dkd/${MODEL}.yaml --M [1,2] --gpu ${GPU_ID}"
    run_experiment "python tools/train.py --cfg configs/cifar100/sdd_dkd/${MODEL}.yaml --M [1,2,4] --gpu ${GPU_ID}"
    run_experiment "python tools/train.py --cfg configs/cifar100/sdd_dkd/${MODEL}.yaml --M [1] --logit-stand --gpu ${GPU_ID}"
    run_experiment "python tools/train.py --cfg configs/cifar100/sdd_dkd/${MODEL}.yaml --M [1,2] --logit-stand --gpu ${GPU_ID}"
    run_experiment "python tools/train.py --cfg configs/cifar100/sdd_dkd/${MODEL}.yaml --M [1,2,4] --logit-stand --gpu ${GPU_ID}"

elif [[ "$METHOD" == "rld" ]]; then
    # Original RLD experiments
    run_experiment "python tools/train.py --cfg configs/cifar100/rld/${MODEL}.yaml --M [1] --base-temp $BASE_TEMP --kd-weight $KD_WEIGHT --gpu ${GPU_ID}"
    run_experiment "python tools/train.py --cfg configs/cifar100/rld/${MODEL}.yaml --M [1] --logit-stand --base-temp $BASE_TEMP --kd-weight $KD_WEIGHT --gpu ${GPU_ID}"
    
    # SDD-RLD experiments
    run_experiment "python tools/train.py --cfg configs/cifar100/sdd_rld/${MODEL}.yaml --M [1] --base-temp $BASE_TEMP --kd-weight $KD_WEIGHT --gpu ${GPU_ID}"
    run_experiment "python tools/train.py --cfg configs/cifar100/sdd_rld/${MODEL}.yaml --M [1,2] --base-temp $BASE_TEMP --kd-weight $KD_WEIGHT --gpu ${GPU_ID}"
    run_experiment "python tools/train.py --cfg configs/cifar100/sdd_rld/${MODEL}.yaml --M [1,2,4] --base-temp $BASE_TEMP --kd-weight $KD_WEIGHT --gpu ${GPU_ID}"
    run_experiment "python tools/train.py --cfg configs/cifar100/sdd_rld/${MODEL}.yaml --M [1] --logit-stand --base-temp $BASE_TEMP --kd-weight $KD_WEIGHT --gpu ${GPU_ID}"
    run_experiment "python tools/train.py --cfg configs/cifar100/sdd_rld/${MODEL}.yaml --M [1,2] --logit-stand --base-temp $BASE_TEMP --kd-weight $KD_WEIGHT --gpu ${GPU_ID}"
    run_experiment "python tools/train.py --cfg configs/cifar100/sdd_rld/${MODEL}.yaml --M [1,2,4] --logit-stand --base-temp $BASE_TEMP --kd-weight $KD_WEIGHT --gpu ${GPU_ID}"

else
    echo "Error: Unknown method $METHOD. Use: kd, dkd, or rld"
    exit 1
fi

echo "All experiments completed!"