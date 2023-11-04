#!/bin/bash 


source /home/yuke/lyd/conda.sh
conda activate pytorchNCCL-hao

module purge
ml cudatoolkit-standalone/11.8.0
ml gcc

# export $PROTOCOL=RDMA/tcpip/ipoib
export PROTOCOL=RDMA
# export $MODEL=gpt2/bert/t5

export MODEL=$1
export MICRO_BATCH_SIZE=$2
export GLOBAL_BATCH_SIZE=$3

export TP=$6
export PP=$7

export WORKDIR=~/lyd
export LD_PRELOAD=/home/yuke/lyd/nccl/build/lib/libnccl.so.2


export MICRO_BATCH_SIZE_GPT2=$MICRO_BATCH_SIZE

export GLOBAL_BATCH_SIZE_GPT2=$GLOBAL_BATCH_SIZE

export MICRO_BATCH_SIZE_BERT=$MICRO_BATCH_SIZE

export GLOBAL_BATCH_SIZE_BERT=$GLOBAL_BATCH_SIZE

export MICRO_BATCH_SIZE_GPT2_L=$MICRO_BATCH_SIZE

export GLOBAL_BATCH_SIZE_GPT2_L=$GLOBAL_BATCH_SIZE

export MICRO_BATCH_SIZE_T5=$MICRO_BATCH_SIZE

export GLOBAL_BATCH_SIZE_T5=$GLOBAL_BATCH_SIZE





cd /home/yuke/lyd/Megatron-LM





rm -rf /home/yuke/lyd/Megatron-LM/checkpoints/
rm -rf /local/scratch/checkpoints/
#cd /local/scratch


export GPUS_PER_NODE=4

export NCCL_DEBUG=INFO 
export NCCL_DEBUG_SUBSYS=ALL

export NNODES=$5

# export MODEL_PARALLEL_SIZE=2

# export directory="/home/ldai8/bash/Megatron_data_output_profile/${MODEL}/${MODEL_PARALLEL_SIZE}gpus-mbs${MICRO_BATCH_SIZE_BERT}"

# if [ ! -d "$directory" ]; then
#     mkdir -p "$directory/tcpip" "$directory/ipoib" "$directory/RDMA"
#     echo "Directory created."
# else
#     echo "Directory already exists."
# fi


export WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

export DP=$(($GPUS_PER_NODE*$NNODES/$TP/$PP))

if [ "$MODEL" = "gpt2" ]; then
    export CHECKPOINT_PATH=/home/yuke/lyd/Megatron-LM/checkpoints/gpt2_345m
	export VOCAB_FILE=/home/yuke/lyd/Megatron-LM/model/gpt2-vocab.json
	export MERGE_FILE=/home/yuke/lyd/Megatron-LM/model/gpt2-merges.txt
	export DATA_PATH=/home/yuke/lyd/Megatron-LM/my-gpt2_text_document 
elif [ "$MODEL" = "bert" ]; then
    export CHECKPOINT_PATH=/home/yuke/lyd/Megatron-LM/checkpoints/bert_345m
	export VOCAB_FILE=/home/yuke/lyd/Megatron-LM/model/bert-large-cased-vocab.txt
	export DATA_PATH=/home/yuke/lyd/Megatron-LM/my-bert_text_sentence
elif [ "$MODEL" = "gpt2large" ]; then
    export CHECKPOINT_PATH=/home/yuke/lyd/Megatron-LM/checkpoints/gpt2_774m
	#cp /home/yuke/lyd/Megatron-LM/model/gpt2-vocab.json /local/scratch/gpt2-vocab.json
    export VOCAB_FILE=/home/yuke/lyd/Megatron-LM/model/gpt2-vocab.json
	#cp /home/yuke/lyd/Megatron-LM/model/gpt2-merges.txt /local/scratch/gpt2-merges.txt
    export MERGE_FILE=/home/yuke/lyd/Megatron-LM/model/gpt2-merges.txt
	#cp /home/yuke/lyd/Megatron-LM/my-gpt2_text_document* /local/scratch
    export DATA_PATH=/home/yuke/lyd/Megatron-LM/my-gpt2_text_document  
else
    export CHECKPOINT_PATH=/home/yuke/lyd/Megatron-LM/checkpoints/t5_base
	export VOCAB_FILE=/home/yuke/lyd/Megatron-LM/model/bert-large-cased-vocab.txt
	export DATA_PATH=/home/yuke/lyd/Megatron-LM/my-t5_text_sentence
fi


# hostname -I

# # # notes for hostname tests
# # hsn0 10.201.2.27 ### 100gbps (ib_send_bw 10.201.2.27)
# # hsn1 10.201.2.7 ### 100gbps
# # bond0 10.140.57.108 ### 100gbps

# # hostname -I: all ip address ### 10.140.57.108 10.201.2.7 10.201.2.27

# export NAME_ADD=$(hostname -I | awk '{print $NF}')
# echo $NAME_ADD

# if [ "$PROTOCOL" = "RDMA" ]; then
export MASTER_ADDR=$4
export NCCL_SOCKET_IFNAME=hsn0
export NCCL_NET=IB
# elif [ "$PROTOCOL" = "ipoib" ]; then
#     export MASTER_ADDR="10.3.1.153"
# 	export NCCL_SOCKET_IFNAME=ib0
# 	export NCCL_NET=Socket
# else
#     export MASTER_ADDR="10.1.1.153"
# 	export NCCL_SOCKET_IFNAME=en0
# 	export NCCL_NET=Socket
# fi


# export MASTER_ADDR="10.3.1.158"
# export NCCL_SOCKET_IFNAME=ib0
# export NCCL_NET=IB




hostnames_file="$WORKDIR/myhostnames"

# Get the current node's hostname
current_hostname=$(hostname)

# Initialize line number counter
lineno=0

# Read the hostnames file line by line
while read -r line; do
  # Increment line number counter
  ((lineno++))
  
  # Compare the current line with the current hostname
  if [ "$line" == "$current_hostname" ]; then
    # If there's a match, set NODE_RANK and break out of the loop
    export NODE_RANK=$((lineno - 1))
    echo "NODE_RANK set to $NODE_RANK"
    break
  fi
done < "$hostnames_file"

# Check if NODE_RANK was set
if [ -z "$NODE_RANK" ]; then
  echo "No match found for hostname $current_hostname in file $hostnames_file"
fi





export MASTER_PORT=21242



export DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
--nnodes $NNODES \
--node_rank $NODE_RANK \
--master_addr $MASTER_ADDR \
--master_port $MASTER_PORT"



iostat -x 1 /dev/nvme0n1 /dev/nvme1n1 > /home/yuke/lyd/tmp-file/logs/iostat-${MODEL}-worldsize${WORLD_SIZE}-mbs${MICRO_BATCH_SIZE_GPT2_L}-noderank${NODE_RANK}-gbs${GLOBAL_BATCH_SIZE_GPT2_L}-DP${DP}-TP${TP}-PP${PP}.csv &
IO_PID=$!

dool --time --mem --cpu --net -N hsn0,hsn1,lo,total --output /home/yuke/lyd/tmp-file/logs/dool-${MODEL}-worldsize${WORLD_SIZE}-mbs${MICRO_BATCH_SIZE_GPT2_L}-noderank${NODE_RANK}-gbs${GLOBAL_BATCH_SIZE_GPT2_L}-DP${DP}-TP${TP}-PP${PP}.csv 1 &
DOOL_PID=$!

nvidia-smi --query-gpu=name,timestamp,uuid,utilization.gpu,memory.total,utilization.memory,power.draw --format=csv -l 1 > /home/yuke/lyd/tmp-file/logs/nvidiasmi-${MODEL}-worldsize${WORLD_SIZE}-mbs${MICRO_BATCH_SIZE_GPT2_L}-noderank${NODE_RANK}-gbs${GLOBAL_BATCH_SIZE_GPT2_L}-DP${DP}-TP${TP}-PP${PP}.csv &
NVIDIA_PID=$!

sh /home/yuke/lyd/megatron_run_scripts/rtop.sh -d hsn0 > /home/yuke/lyd/tmp-file/logs/hsn0-${MODEL}-worldsize${WORLD_SIZE}-mbs${MICRO_BATCH_SIZE_GPT2_L}-noderank${NODE_RANK}-gbs${GLOBAL_BATCH_SIZE_GPT2_L}-DP${DP}-TP${TP}-PP${PP}.csv &
RTOP1_PID=$!

sh /home/yuke/lyd/megatron_run_scripts/rtop.sh -d hsn1 > /home/yuke/lyd/tmp-file/logs/hsn1-${MODEL}-worldsize${WORLD_SIZE}-mbs${MICRO_BATCH_SIZE_GPT2_L}-noderank${NODE_RANK}-gbs${GLOBAL_BATCH_SIZE_GPT2_L}-DP${DP}-TP${TP}-PP${PP}.csv &
RTOP2_PID=$!

if [ "$MODEL" = "gpt2" ]; then
    /home/yuke/lyd/conda3/envs/pytorchNCCL-hao/bin/python -m torch.distributed.launch $DISTRIBUTED_ARGS /home/yuke/lyd/Megatron-LM/pretrain_gpt.py \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --num-layers 24 --hidden-size 1024 --num-attention-heads 16 --seq-length 512 --max-position-embeddings 512 \
    --micro-batch-size $MICRO_BATCH_SIZE_GPT2 --global-batch-size $GLOBAL_BATCH_SIZE_GPT2 --lr 0.00015 --train-iters 100 --lr-decay-iters 64 \
    --lr-decay-style cosine --vocab-file $VOCAB_FILE --merge-file $MERGE_FILE --lr-warmup-fraction .01 --fp16 \
    --log-interval 1 --save-interval 50 --eval-interval 10 --eval-iters 1 --save $CHECKPOINT_PATH --load $CHECKPOINT_PATH \
    --data-path $DATA_PATH > /home/yuke/lyd/tmp-file/logs/megatron-${MODEL}-worldsize${WORLD_SIZE}-mbs${MICRO_BATCH_SIZE}-noderank${NODE_RANK}-gbs${GLOBAL_BATCH_SIZE}-DP${DP}-TP${TP}-PP${PP}.out
elif [ "$MODEL" = "bert" ]; then
    /home/yuke/lyd/conda3/envs/pytorchNCCL-hao/bin/python -m torch.distributed.launch $DISTRIBUTED_ARGS /home/yuke/lyd/Megatron-LM/pretrain_bert.py \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --num-layers 24 --hidden-size 1024 --num-attention-heads 16 --seq-length 512 --max-position-embeddings 512 \
    --lr 0.0001 --lr-decay-iters 49 --train-iters 100 --min-lr 0.00001 --lr-warmup-fraction 0.01 \
    --micro-batch-size $MICRO_BATCH_SIZE_BERT --global-batch-size $GLOBAL_BATCH_SIZE_BERT \
    --vocab-file $VOCAB_FILE --split 949,50,1 --fp16 --log-interval 1 --save-interval 50 --eval-interval 10 --eval-iters 1 --recompute-method uniform \
    --save $CHECKPOINT_PATH --load $CHECKPOINT_PATH \
    --data-path $DATA_PATH > /home/yuke/lyd/tmp-file/logs/megatron-${MODEL}-worldsize${WORLD_SIZE}-mbs${MICRO_BATCH_SIZE}-noderank${NODE_RANK}-gbs${GLOBAL_BATCH_SIZE}-DP${DP}-TP${TP}-PP${PP}.out
elif [ "$MODEL" = "gpt2large" ]; then
    /home/yuke/lyd/conda3/envs/pytorchNCCL-hao/bin/python -m torch.distributed.launch $DISTRIBUTED_ARGS /home/yuke/lyd/Megatron-LM/pretrain_gpt.py \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --num-layers 40 --hidden-size 1280 --num-attention-heads 20 --seq-length 512 --max-position-embeddings 512 \
    --micro-batch-size $MICRO_BATCH_SIZE_GPT2_L --global-batch-size $GLOBAL_BATCH_SIZE_GPT2_L --lr 0.00015 --train-iters 100 --lr-decay-iters 64 \
    --lr-decay-style cosine --vocab-file $VOCAB_FILE --merge-file $MERGE_FILE --lr-warmup-fraction .01 --fp16 \
    --log-interval 1 --save-interval 50 --eval-interval 10 --eval-iters 1 --save $CHECKPOINT_PATH --load $CHECKPOINT_PATH \
    --data-path $DATA_PATH > /home/yuke/lyd/tmp-file/logs/megatron-${MODEL}-worldsize${WORLD_SIZE}-mbs${MICRO_BATCH_SIZE}-noderank${NODE_RANK}-gbs${GLOBAL_BATCH_SIZE}-DP${DP}-TP${TP}-PP${PP}.out
else
    /home/yuke/lyd/conda3/envs/pytorchNCCL-hao/bin/python -m torch.distributed.launch $DISTRIBUTED_ARGS /home/yuke/lyd/Megatron-LM/pretrain_t5.py --num-layers 24 --hidden-size 1024 \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --pipeline-model-parallel-split-rank $(($PP/2)) \
    --num-attention-heads 16 --kv-channels 64 --ffn-hidden-size 3072 --encoder-seq-length 512 --decoder-seq-length 128 --max-position-embeddings 512 \
    --lr 0.0001 --lr-decay-iters 49 --train-iters 100 --min-lr 0.00001 --lr-warmup-fraction 0.01 \
    --micro-batch-size $MICRO_BATCH_SIZE_T5 --global-batch-size $GLOBAL_BATCH_SIZE_T5 \
    --vocab-file $VOCAB_FILE --vocab-extra-ids 100 --split 949,50,1 --fp16 --log-interval 1 --save-interval 50 --eval-interval 10 --eval-iters 1 \
    --recompute-method uniform --save $CHECKPOINT_PATH --load $CHECKPOINT_PATH \
    --data-path $DATA_PATH > /home/yuke/lyd/tmp-file/logs/megatron-${MODEL}-worldsize${WORLD_SIZE}-mbs${MICRO_BATCH_SIZE}-noderank${NODE_RANK}-gbs${GLOBAL_BATCH_SIZE}-DP${DP}-TP${TP}-PP${PP}.out
fi


cd /home/yuke/lyd/Megatron-LM

rm -rf /home/yuke/lyd/Megatron-LM/checkpoints/
rm -rf /local/scratch/checkpoints/

echo "Training done on Node$NODE_RANK"

kill $IO_PID
kill $DOOL_PID
kill $NVIDIA_PID
kill $RTOP1_PID
kill $RTOP2_PID

echo "Kill monitors done"

exit



# /home/yuke/lyd/conda3/envs/pytorchNCCL-hao/bin/python -m torch.distributed.launch $DISTRIBUTED_ARGS pretrain_gpt.py \
# --num-layers 36 --hidden-size 1280 --num-attention-heads 20 --seq-length 512 --max-position-embeddings 512 \
# --micro-batch-size $MICRO_BATCH_SIZE_GPT2_L --global-batch-size $GLOBAL_BATCH_SIZE_GPT2_L --lr 0.00015 --train-iters 100 --lr-decay-iters 64 \
# --lr-decay-style cosine --vocab-file $VOCAB_FILE --merge-file $MERGE_FILE --lr-warmup-fraction .01 --fp16 \
# --log-interval 1 --save-interval 50 --eval-interval 10 --eval-iters 1 --save $CHECKPOINT_PATH --load $CHECKPOINT_PATH \
# --data-path $DATA_PATH > ~/lyd/logs/megatron-${MODEL}-worldsize${WORLD_SIZE}-mbs${MICRO_BATCH_SIZE}-noderank${NODE_RANK}-gbs-${GLOBAL_BATCH_SIZE}.out







# kill %1
# kill %2
# kill %3



# echo "debug 10"

# dool --time --mem --cpu --net -N en0,en1,ib0,lo,total 1
# sh rtop/rtop.sh -d ib0 -all
# NCCL_IB_DISABLE=1


# python tools/preprocess_data.py \ 
# --input /home/hqi6/data/LLM/Megatron-LM/text/AA/wiki_00 \ 
# --output-prefix my-gpt2 \    
# --vocab ./model/gpt2-vocab.json \       
# --dataset-impl mmap \       
# --tokenizer-type GPT2BPETokenizer \       
# --merge-file ./model/gpt2-merges.txt \  
# --workers 1 \ 
# --chunk-size 1024 \     
# --append-eod 

# python tools/preprocess_data.py --input /home/hqi6/data/LLM/Megatron-LM/text/AA/wiki_00 --output-prefix my-bert --vocab ./model/bert-large-cased-vocab.txt --dataset-impl mmap --tokenizer-type BertWordPieceLowerCase --workers 1 --chunk-size 1024 --split-sentences




# nohup dool --time --mem --cpu --net -N hsn0,hsn1,lo,total 1 > dool.csv 2>&1 &
# DOOL_PID=$!

# nohup nvidia-smi --query-gpu=name,timestamp,uuid,utilization.gpu,memory.total,utilization.memory,power.draw --format=csv -l 1 > nvidia-smi.csv 2>&1 &
# NVIDIA_PID=$!

# nohup sh /home/yuke/lyd/megatron_run_scripts/rtop.sh -d hsn0 > hsn0.csv 2>&1 &
# RTOP1_PID=$!

# nohup sh /home/yuke/lyd/megatron_run_scripts/rtop.sh -d hsn1 > hsn1.csv 2>&1 &
# RTOP2_PID=$!

# # Execute your Python script
# python train.py

# # After the Python script finishes, stop the background processes
# kill $DOOL_PID
# kill $NVIDIA_PID
# kill $RTOP1_PID
# kill $RTOP2_PID
