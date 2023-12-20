#!/bin/bash
#SBATCH -J megatron           # Job name
#SBATCH -o megatron.out%j       # Name of stdout output file
#SBATCH -e megatron.e%j       # Name of stderr error file
#SBATCH -p rtx           # Queue (partition) name
#SBATCH -N 2               # Total # of nodes (must be 1 for serial)
#SBATCH -n 2               # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 02:30:00        # Run time (hh:mm:ss)
##SBATCH --mail-type=all    # Send email at begin and end of job
##SBATCH -A myproject       # Project/Allocation name (req'd if you have more than 1)
##SBATCH --mail-user=username@tacc.utexas.edu

#---- USER CONFIG PARAMS----
#export MPI_HOME=/opt/cray/pe/mpich/8.1.16/ofi/gnu/9.1
export MPI_HOME=/opt/intel/compilers_and_libraries_2020.4.304/linux/mpi/intel64
# MPI_RUN=/opt/pbs/bin/mpiexec
#-------------------------
# #module purge
# module load nvhpc/23.1
# ml cudatoolkit-standalone/11.8.0
# ml gcc

module reset
# module swap PrgEnv-nvhpc PrgEnv-gnu
# #ml nvhpc-mixed/22.11
# ml gcc/10.3.0
# ml cudatoolkit-standalone/11.8.0
cd /work2/08664/tg879638/frontera
source cuda.sh


echo "Current(master) node:$(hostname)"
scontrol show hostname $SLURM_JOB_NODELIST
export WORKDIR=/work2/08664/tg879638/frontera/lyd
scontrol show hostname $SLURM_JOB_NODELIST > $WORKDIR/myhostnames
HOST_NUM=$(wc -l < $WORKDIR/myhostnames)
echo "Number of nodes: $HOST_NUM"
sed -i 's/.hsn.*//' $WORKDIR/myhostnames
cat $WORKDIR/myhostnames

# Clean up Python and mpiexec processes on each host
for host in $(cat $WORKDIR/myhostnames); do
    echo "Cleaning up on node: $host"
    ssh $host 'killall -9 python; killall -9 ibrun'
done

echo "Cleanup completed."




export MASTER_ADD=$(hostname -I | awk '{print $NF}')
echo "Master Addr is: $MASTER_ADD"



sed -i 's/$/.frontera.tacc.utexas.edu/' $WORKDIR/myhostnames


#$MPI_RUN -f $WORKDIR/myhostnames -np $HOST_NUM $WORKDIR/run_megatron.sh gpt2large 12 192 $MASTER_ADD $HOST_NUM
#mpiexec -f $WORKDIR/myhostnames -np $HOST_NUM $WORKDIR/run_megatron.sh gpt2large 12 192 $MASTER_ADD $HOST_NUM
#mpiexec -np $HOST_NUM --ppn 1 sh -c 'echo "hello from $(hostname)"'
#ibrun -np $HOST_NUM -ppn 1 sh $WORKDIR/megatron_running_scripts/run_megatron.sh gpt2large 8 64 $MASTER_ADD $HOST_NUM 1 1
ibrun sh $WORKDIR/megatron_running_scripts/run_megatron.sh bert 8 8 $MASTER_ADD $HOST_NUM 1 8
# $MPI_RUN -hostfile  ~/myhostnames -np $HOST_NUM $WORKDIR/tf_cnn_bench.sh
echo "Done on Frontera job"
