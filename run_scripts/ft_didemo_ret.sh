cd .. 

export PYTHONPATH="$PYTHONPATH:$PWD"
echo $PYTHONPATH

CONFIG_PATH='config_release/didemo_ret.json'

horovodrun -np 8 python src/tasks/run_video_retrieval.py \
      --config $CONFIG_PATH \
      --output_dir /export/home/workspace/experiments/alpro/finetune/didemo_ret/$(date '+%Y%m%d%H%M%S')
