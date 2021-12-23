# ALPRO

## Align and Prompt: Video-and-Language Pre-training with Entity Prompts [[Paper](https://arxiv.org/abs/2112.09583)]

[Dongxu Li](https://www.linkedin.com/in/dongxu-li-a8a035110/), [Junnan Li](https://sites.google.com/site/junnanlics), [Hongdong Li](http://users.cecs.anu.edu.au/~hongdong/), [Juan Carlos Niebles](http://www.niebles.net/), [Steven C.H. Hoi](https://sites.google.com/view/stevenhoi/home)

<img src="pics/teaser.jpg" width="500">

Official PyTorch code for ALPRO. This repository supports pre-training as well as finetuning on 
- Text-Video Retrieval on MSRVTT and DiDeMo.
- Video Question Anwsering on MSRVTT and MSVD.

## Requirements
Our implementation is tested on Ubuntu 20.04.1 with NVIDIA A100 GPUs. Supports for other platforms and hardwares are possible with no warrant. To install the required packages:

```bash
cd env && bash install_pkg.sh
```

## Data Preparation 
1. Download Annotations and Pre-trained Checkpoints
    - [Text annotations](https://storage.googleapis.com/sfr-vision-language-research/ALPRO/data.zip)
    - [Checkpoints of pre-trained model and finetuned model](https://storage.googleapis.com/sfr-vision-language-research/ALPRO/output.zip)
    - [Externel resources](https://storage.googleapis.com/sfr-vision-language-research/ALPRO/ext.zip)
    - unzip `data.zip`, `output.zip`, `ext.zip` under `ALPRO/`.
 
2. Download raw videos of downstream datasets.
   - MSRVTT:
     - download train_val_videos.zip and test_videos.zip from e.g. [here](https://www.mediafire.com/folder/h14iarbs62e7p/shared).
     - check md5sum:

        ```bash
        51f2394d279cf84f1642defd9a651e6f  train_val_videos.zip
        0af68454cec9d586e92805739f3911d0  test_videos.zip
        ```
     - unzip all the videos into `data/msrvtt_ret/videos` (10k in total).
     - create the following soft link:

        ```bash
        ln -s data/msrvtt_ret/videos data/msrvtt_qa/videos```
    - MSVD:
      - download from official release:
  
        ```bash
        wget -nc https://www.cs.utexas.edu/users/ml/clamp/videoDescription/YouTubeClips.tar
        ```
      - check md5sum:
      
        ```bash
        9bdb20fcf14d59524a6febca9f6a8d89  YouTubeClips.tar
        ```
      - unzip all the videos to `data/msvd_qa/videos` (1,970 videos in total).
        
        ```bash
        mkdir data/msvd_qa/videos/ 
        tar xvf YouTubeClips.tar -C data/msvd_qa/videos --strip-components=1
        ```
    - DiDeMo:
       - Following [instructions](https://github.com/LisaAnne/LocalizingMoments/blob/master/README.md) and download from the official release [here](https://drive.google.com/drive/u/1/folders/1_oyJ5rQiZboipbMl6tkhY8v0s9zDkvJc);
       - unzip all the videos into `data/didemo_ret/videos`.
       - Note there might be a couple videos missing. See [here](https://github.com/LisaAnne/LocalizingMoments/blob/master/README.md#getting-the-videos) to download. However, as they account for a small portion of training set, you may feel safe to ignore.
       - Convert all the DiDeMo videos into `*.mp4` format using e.g. [`ffmpeg`](https://askubuntu.com/questions/396883/how-to-simply-convert-video-files-i-e-mkv-to-mp4).
       - We obtained 10,463 videos following these steps (with one video `77807177@N00_5753455690_1e04ccb364` missing).



  3. The directory is expected to be in the structure below:
      ```bash
      .
      |-config_release  # configuration files
      |-data  # text annotations and raw videos
      |---didemo_ret
      |-----txt
      |-----videos
      |---msrvtt_qa/...
      |---msrvtt_ret/...
      |---msvd_qa/...
      |-env  # scripts to install packages
      |-ext  # external resources, e.g. bert tokenizer
      |-output  # checkpoints for pre-trained/finetuned models
      |---downstreams
      |-----didemo_ret
      |-------public
      |---------ckpt # official finetuned checkpoints
      |---------log # inference log
      |---------results_test
      |-----------step_best_1_mean
      |-----msrvtt_qa/...
      |-----msrvtt_ret/...
      |-----msvd_qa/...
      |-run_scripts  # bash scripts to launch experiments
      |-src  # source code
      ```

## Inference with Official Checkpoints

  ```bash
  cd run_scripts
  bash inf_msrvtt_ret.sh
  # {'text2video': {'r1': 33.9, 'r5': 60.7, 'r10': 73.2, 'medianR': 3.0, 'meanR': 27.404}}
  bash inf_didemo_ret.sh
  # {'text2video': {'r1': 35.9, 'r5': 67.5, 'r10': 78.8, 'medianR': 3.0, 'meanR': 19.125}}
  bash inf_msrvtt_qa.sh
  # {'ratios': {'what_ratio': [68.48, 49872], 'who_ratio': [27.99, 20385], 'how_ratio': [2.25, 1640], 'where_ratio': [0.34, 250], 'when_ratio': [0.93, 677]}, 'overall_acc': 42.12, 'what_acc': 36.05, 'who_acc': 52.24, 'how_acc': 85.67, 'where_acc': 42.8, 'when_acc': 78.88}
  bash inf_msvd_qa.sh
  # {'ratios': {'what_ratio': [61.93, 8150], 'who_ratio': [34.6, 4554], 'how_ratio': [2.81, 370], 'where_ratio': [0.21, 28], 'when_ratio': [0.44, 58]}, 'overall_acc': 45.91, 'what_acc': 37.02, 'who_acc': 58.59, 'how_acc': 81.62, 'where_acc': 46.43, 'when_acc': 72.41}
  ```


## Downstream Task Finetuning
  - To finetune on downstream tasks with the pre-trained checkpoint `output/pretrain/alpro_pretrained_ckpt.pt`

    ```bash
    cd run_scripts
    bash ft_msrvtt_ret.sh
    bash ft_didemo_ret.sh
    bash ft_msrvtt_qa.sh
    bash ft_msvd_qa.sh
    ```
  
    For example, with MSRVTT retrieval:
    ```bash
    cd ALPRO/

    export PYTHONPATH="$PYTHONPATH:$PWD"
    echo $PYTHONPATH

    CONFIG_PATH='config_release/msrvtt_ret.json'

    horovodrun -np 8 python src/tasks/run_video_retrieval.py \ # change -np to GPUs numbers.
        --config $CONFIG_PATH \
        --output_dir /export/home/workspace/experiments/alpro/finetune/msrvtt_ret/$(date '+%Y%m%d%H%M%S')  # change to your local path to store finetuning ckpts and logs 
    ``` 
 - Run inference with locally-finetuned checkpoints.
   ```bash
    cd ALPRO/

    export PYTHONPATH="$PYTHONPATH:$PWD"
    echo $PYTHONPATH

    STEP='best'

    CONFIG_PATH='config_release/msrvtt_ret.json'
    OUTPUT_DIR='[INPUT_YOUR_OUTPUT_PATH_HERE]'

    TXT_DB='data/msrvtt_ret/txt/test.jsonl'
    IMG_DB='data/msrvtt_ret/videos'

    horovodrun -np 8 python src/tasks/run_video_retrieval.py \
        --do_inference 1 \
        --inference_split test \
        --inference_model_step $STEP \
        --inference_txt_db $TXT_DB \
        --inference_img_db $IMG_DB \
        --inference_batch_size 64 \
        --output_dir $OUTPUT_DIR \
        --config $CONFIG_PATH
   ```  
   - `OUTPUT_DIR` is the path after the `--output_dir` option in the finetuning script.
   - `$STEP` is a string, which tells the script to use the checkpoint `$OUTPUT_DIR/ckpt/model_step_$STEP.pt` for inference. 


## Pretraining
1. Download [WebVid2M](https://github.com/m-bain/frozen-in-time) and [CC-3M](https://github.com/igorbrigadir/DownloadConceptualCaptions).
  
    - Put WebVid2M videos under `data/webvid2m`;
    - 💡 we downsample webvid2m videos to 10% of the original FPS to speed-up video loading;
    - change `data/cc3m/txt/cc3m.json` with local image paths.

2. Training Prompter:
    ```bash
    cd run_scripts && bash pt_prompter.sh
    ```   

3. Training video-language model: 
    ```bash
    cd run_scripts && bash pt_alpro.sh
    ```
    If you would like to use custom prompter weight, please change `teacher_weights_path` in `config_release/pretrain_alpro.json`
4. To finetune with pre-trained checkpoints, please change `e2e_weights_path` in the finetuning config files, e.g. `config_release/msrvtt_ret.json`.


## Citation

If you find ALPRO useful for your research, please consider citing:
```bibtex
  @inproceedings{li2021align,
    title={Align and Prompt: Video-and-Language Pre-training with Entity Prompts},
    author={Dongxu Li, Junnan Li, Hongdong Li, Juan Carlos Niebles, Steven C.H. Hoi},
    booktitle={arxiv},
    year={2021}
  }
```

## Acknowledgement
We thank members at Salesforce Research for their helpful discussions.

The implementation of ALPRO relies on resources from [ClipBERT](https://github.com/jayleicn/ClipBERT),
[transformers](https://github.com/huggingface/transformers), 
[TimeSformer](https://github.com/facebookresearch/TimeSformer/tree/main/timesformer/models), 
The code is implemented using [PyTorch](https://github.com/pytorch/pytorch), 
with multi-GPU support from [Horovod](https://github.com/horovod/horovod) and [gradient-checkpoint](https://github.com/csrhddlam/pytorch-checkpoint).  We thank the original authors for their open-sourcing and encourage ALPRO users to cite their works when applicable.

