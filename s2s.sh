export CUDA_VISIBLE_DEVICES=0
python3 -u run_seq2seq.py \
    --data_dir ./data/weibo \
    --src_file train_data.json \
    --model_type unilm \
    --model_name_or_path ./bert-model-pt/torch_unilm_model/ \
    --output_dir ./output_dir/ \
    --max_seq_length 256 \
    --max_position_embeddings 512 \
    --do_train \
    --do_lower_case \
    --train_batch_size 64 \
    --learning_rate 1e-5 \
    --num_train_epochs 5
