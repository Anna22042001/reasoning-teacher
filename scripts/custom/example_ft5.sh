TARGETS=("single_eq" "addsub" "multiarith")
MODELS=("flan_t5_small" "flan_t5_base")
DEVICES="0"


for MODEL in ${MODELS[@]}; do
  for TARGET in ${TARGETS[@]}; do
    python custom_train.py --dataset_key $TARGET --model_key $MODEL --train_key "ft_cot" --devices $DEVICES --batch_size 8 --inference_batch_size 32
  done
done
