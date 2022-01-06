nohup python3 -u decode_seq2seq.py \
--model_type unilm \
--model_name_or_path /torch_unilm_model \
--model_recover_path /output_dir/model.5.bin \
--max_seq_length 512 \
--input_file /data/weibo/test_data.json \
--output_file /data/weibo/predict.json \
--do_lower_case \
--batch_size 32 \
--beam_size 5 \
--max_tgt_length 128 \
> eval_log.log 2>&1 &