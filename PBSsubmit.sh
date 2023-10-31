#!/bin/bash
#PBS -l select=4:ngpus=2:system=polaris
#PBS -l place=scatter
#PBS -l walltime=00:09:59
#PBS -q debug
#PBS -l filesystems=home
#PBS -A CSC250STPM09
#PBS -k doe
#PBS -N megatron-test
#PBS -o megatron-test.out
#PBS -e megatron-test.error

#---- USER CONFIG PARAMS----
MPI_RUN=/home/biswas.91/projects/mvapich2-2.3b/install/bin/mpirun_rsh
#-------------------------
module purge
ml cudatoolkit-standalone/11.8.0
ml gcc

echo "Current(master) node:$(hostname)"

export WORKDIR=`pwd`
cat $PBS_NODEFILE > $WORKDIR/myhostnames # store the hostnames
HOST_NUM=$(wc -l < $WORKDIR/myhostnames)
echo "Number of nodes: $HOST_NUM"

# Clean up Python and mpiexec processes on each host
for host in $(cat $WORKDIR/myhostnames); do
    echo "Cleaning up on node: $host"
    ssh $host 'killall -9 python; killall -9 mpiexec'
done

echo "Cleanup completed."


export MASTER_ADD=$(hostname -I | awk '{print $NF}')
echo "Master Addr is: $MASTER_ADD"

$MPI_RUN -f $WORKDIR/myhostnames -np $HOST_NUM $WORKDIR/run_megatron.sh gpt2large 12 192 $MASTER_ADD $HOST_NUM

# $MPI_RUN -hostfile  ~/myhostnames -np $HOST_NUM $WORKDIR/tf_cnn_bench.sh