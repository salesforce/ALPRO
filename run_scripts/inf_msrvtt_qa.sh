cd .. 

export PYTHONPATH="$PYTHONPATH:$PWD"
echo $PYTHONPATH

STEP='best'

CONFIG_PATH='config_release/msrvtt_qa.json'

TXT_DB='data/msrvtt_qa/txt/test.jsonl'
IMG_DB='data/msrvtt_qa/videos'

horovodrun -np 8 python src/tasks/run_video_qa.py \
      --do_inference 1 \
      --inference_split test \
      --inference_model_step $STEP \
      --inference_txt_db $TXT_DB \
      --inference_img_db $IMG_DB \
      --inference_batch_size 64 \
      --output_dir output/downstreams/msrvtt_qa/public \
      --config $CONFIG_PATH