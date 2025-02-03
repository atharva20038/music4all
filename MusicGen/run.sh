#!/bin/bash
 
# Set NCCL debug mode to gather more information about NCCL errors
export NCCL_DEBUG=INFO
 
# Disable InfiniBand for NCCL (might help if there are networking issues)
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1  # Disable peer-to-peer communication
 
# Set the network interface for NCCL communication
export NCCL_SOCKET_IFNAME=eno2
 
# Set OpenMP threads to avoid system overload
export OMP_NUM_THREADS=2
export NCCL_TIMEOUT=300  # Increase NCCL timeout to handle slow network initialization
 
torchrun --nproc_per_node=2 Training.py
