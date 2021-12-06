apt update
apt install lsof

# horovod
HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_NCCL_LINK=SHARED HOROVOD_WITH_PYTORCH=1 \
    pip install --no-cache-dir horovod==0.19.4 &&\
    ldconfig

# use the faster pillow-simd instead of the original pillow
# https://github.com/uploadcare/pillow-simd
pip uninstall pillow && \
CC="cc -mavx2" pip install -U --force-reinstall pillow-simd

spacy download en

pip install -r requirements.txt

git clone https://github.com/NVIDIA/apex.git &&\
    cd apex &&\
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" . &&\
    rm -rf ../apex

