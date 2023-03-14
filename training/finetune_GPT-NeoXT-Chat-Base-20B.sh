DIR=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)

netif=lo
export GLOO_SOCKET_IFNAME=${netif}
export NCCL_SOCKET_IFNAME=${netif}
export MODEL_NAME=Flux-GPT-20B

export SHOW_DATA=0

BASE_MODEL="${DIR}/../pretrained/GPT-NeoX-20B/togethercomputer_GPT-NeoXT-Chat-Base-20B/"

CHECKPOINT_STEPS=100

DATASETS="\
${DIR}/../data/flux-gpt/flux-gpt-ds-0001.jsonl:0.1 \
"

ARGS="--model-name ${BASE_MODEL} \
--tokenizer-name ${BASE_MODEL} \
--project-name together \
--model-type gptneox \
--optimizer adam \
--seed 42 \
--load-pretrained-model true \
--task-name \
"${DATASETS}" \
--checkpoint-path ${DIR}/../model_ckpts/${MODEL_NAME} \
--total-steps 20000 --warmup-steps 10 --train-warmup-steps 0 \
--checkpoint-steps ${CHECKPOINT_STEPS} \
--lr 1e-6 --seq-length 2048 --batch-size 64 --micro-batch-size 1 --gradient-accumulate-step 1 \
--dist-url tcp://127.0.0.1:7033 \
--num-layers 6 --embedding-dim 6144 \
--world-size 2 --pipeline-group-size 2 --data-group-size 1 \
--job-id 0 --net-interface ${netif} \
--fp16 \
--dp-backend nccl \
--dp-mode allreduce \
--pp-mode gpipe --profiling no-profiling"


(trap 'kill 0' SIGINT; \
python ${DIR}/dist_clm_train.py $(echo ${ARGS}) --cuda-id 0 --rank 0 \
    & \
python ${DIR}/dist_clm_train.py $(echo ${ARGS}) --cuda-id 1 --rank 1 \
    & \
wait)
