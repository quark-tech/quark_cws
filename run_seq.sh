MODE=$1
DO_TRAIN=False
DO_EVAL=False
DO_PRED=False

if [ $MODE == 'train' ];then
    DO_TRAIN=True
elif [ $MODE == 'predict' ]; then
    DO_PRED=True
fi
WARMUP_PROPEORTION=0.1
DATASET_DIR="./dataset/segment/pku"
TASK_NAME="seq"
VOCAB_FILE='./models/N-Gram-Pretrain-Model/vocab.txt'
BERT_CONFIG="./config/bert_config.json"
OUTPUT_DIR="./checkpoint/seq/"
INIT_CKPT="./models/N-Gram-Pretrain-Model"
LOWER=True
MAX_SEQ_LEN=128
TRAIN_BATCH_SIZE=16
LR=0.00001
EPOCHS=12
WARMUP_PROPEORTION=0.1
SAVE_CKPT_STEPS=1000

CUDA_VISIBLE_DEVICES=$2 python2.7 run_seq.py  \
    --data_dir $DATASET_DIR \
    --bert_config_file $BERT_CONFIG \
    --task_name $TASK_NAME \
    --vocab_file $VOCAB_FILE \
    --output_dir $OUTPUT_DIR \
    --init_checkpoint $INIT_CKPT \
    --do_lower_case $LOWER \
    --max_seq_length $MAX_SEQ_LEN \
    --do_train=$DO_TRAIN --do_eval=$DO_EVAL --do_predict=$DO_PRED \
    --train_batch_size $TRAIN_BATCH_SIZE --learning_rate $LR \
    --num_train_epochs $EPOCHS --warmup_proportion $WARMUP_PROPEORTION \
    --save_checkpoints_steps $SAVE_CKPT_STEPS \
