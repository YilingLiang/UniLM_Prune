export CUDA_VISIBLE_DEVICES=2
python3 -u dae.py \
--model_type unilm \
--model_name_or_path ./bert-model-pt/torch_unilm_model \
--model_recover_path ./output_dir/epoch5/model.5.bin \
--max_seq_length 512 \
--input_file data/weibo/test_data.json \
--output_file data/weibo/predict.json \
--do_lower_case \
--batch_size 32 \
--beam_size 1 \
--max_tgt_length 128 \
