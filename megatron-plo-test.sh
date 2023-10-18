#!/bin/bash -l
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=00:59:59
#PBS -q debug
#PBS -l filesystems=home
#PBS -A CSC250STPM09
#PBS -k doe
#PBS -N megatron-test
#PBS -o megatron-test.out
#PBS -e megatron-test.error

sleep 100000000000 