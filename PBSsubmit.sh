#!/bin/bash
#PBS -l select=5:system=polaris
#PBS -l place=scatter
#PBS -l walltime=00:09:59
#PBS -q debug-scaling
#PBS -l filesystems=home
#PBS -A CSC250STPM09
#PBS -k doe
#PBS -N megatron
#PBS -o megatron.out
#PBS -e megatron.error

#---- USER CONFIG PARAMS----
export MPI_HOME=/opt/cray/pe/mpich/8.1.16/ofi/gnu/9.1
# MPI_RUN=/opt/pbs/bin/mpiexec
#-------------------------
#module purge
module load nvhpc/23.1
ml cudatoolkit-standalone/11.8.0
ml gcc

echo "Current(master) node:$(hostname)"
cat $PBS_NODEFILE
export WORKDIR=~/lyd
cat $PBS_NODEFILE > $WORKDIR/myhostnames # store the hostnames
HOST_NUM=$(wc -l < $WORKDIR/myhostnames)
echo "Number of nodes: $HOST_NUM"
sed -i 's/.hsn.*//' $WORKDIR/myhostnames
cat $WORKDIR/myhostnames

# Clean up Python and mpiexec processes on each host
for host in $(cat $WORKDIR/myhostnames); do
    echo "Cleaning up on node: $host"
    ssh $host 'killall -9 python; killall -9 mpiexec'
done

echo "Cleanup completed."


export MASTER_ADD=$(hostname -I | awk '{print $NF}')
echo "Master Addr is: $MASTER_ADD"

#$MPI_RUN -f $WORKDIR/myhostnames -np $HOST_NUM $WORKDIR/run_megatron.sh gpt2large 12 192 $MASTER_ADD $HOST_NUM
#mpiexec -f $WORKDIR/myhostnames -np $HOST_NUM $WORKDIR/run_megatron.sh gpt2large 12 192 $MASTER_ADD $HOST_NUM
#mpiexec -np $HOST_NUM --ppn 1 sh -c 'echo "hello from $(hostname)"'
mpiexec -np $HOST_NUM -ppn 1 sh $WORKDIR/megatron_run_scripts/run_megatron.sh gpt2large 12 240 $MASTER_ADD $HOST_NUM 1 1
# $MPI_RUN -hostfile  ~/myhostnames -np $HOST_NUM $WORKDIR/tf_cnn_bench.sh
echo "Done on PBSjob"
