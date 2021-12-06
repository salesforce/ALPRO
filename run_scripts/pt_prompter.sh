cd .. 

export PYTHONPATH="$PYTHONPATH:$PWD"
echo $PYTHONPATH

CONFIG_PATH='config_release/pretrain_prompter.json'

horovodrun -np 8 python src/pretrain/run_pretrain_contrastive_only.py \
      --config $CONFIG_PATH \
      --output_dir /export/home/workspace/experiments/alpro/prompter/$(date '+%Y%m%d%H%M%S')