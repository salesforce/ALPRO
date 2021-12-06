import os
import json
import random

import torch
import spacy
from torch.utils.data.dataloader import default_collate
from src.utils.logger import LOGGER
from src.utils.basic_utils import flat_list_of_lists, save_frames_grid
from src.datasets.data_utils import VideoRandomSquareCrop, VideoResizeSquare, mask_batch_text_tokens, select_batch_text_pivots
from src.datasets.dataset_base import AlproBaseDataset, img_collate

from src.datasets.randaugment import TemporalConsistentRandomAugment, RandomAugment

from torch.utils.data import Dataset

from torchvision import transforms
from PIL import Image
import numpy as np


class AlproPretrainSparseDataset(AlproBaseDataset):
    """
    datalist: list(tuples)  each tuple is (img_id, list(dicts)),
        each dict {
            "type": "image",
            "filepath": "/abs/path/to/COCO_val2014_000000401092.jpg",
            "text": "A plate of food and a beverage are on a table.",  # should be tokenized and digitized first?
            ...
            }
    tokenizer:
    max_img_size: int,
    max_txt_len: int, max text sequence length, including special tokens.
    vis_format: str, image or video, used to decide data loading method.
    """
    def __init__(self, datalist, tokenizer, img_lmdb_dir, img_db_type, txt_dir,
                video_fmt='.mp4', crop_size=256, resize_size=288, fps=3, num_frm=3, frm_sampling_strategy="rand",
                max_img_size=1000, max_txt_len=20,
                use_itm=True, is_train=True):
        super(AlproPretrainSparseDataset, self).__init__(
            datalist, tokenizer, img_lmdb_dir, 
            img_db_type=img_db_type,
            fps=fps, 
            num_frm=num_frm, 
            frm_sampling_strategy=frm_sampling_strategy,
            max_img_size=max_img_size, 
            max_txt_len=max_txt_len)
        self.use_itm = use_itm

        self.txt_dir = txt_dir
        self.video_fmt = video_fmt

        self.crop_size = crop_size
        self.video_random_cropper = VideoRandomSquareCrop(crop_size)

        self.resize_size = resize_size

        self.is_train = is_train

        if self.is_train:
            self.randaug = TemporalConsistentRandomAugment(N=2, M=5, augs=['Identity', 'Contrast','Brightness','Sharpness', 'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate', 'HorizontalFlip'])     
        else:
            self.randaug = None

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):
        start_time = None
        end_time = None

        # fetch video
        num_retries = 10  # skip error videos

        for _ in range(num_retries):
            data_sample = self.datalist.iloc[index]

            video_id = str(data_sample.video_id)
            txt_len = int(data_sample.txt_len)

            if hasattr(data_sample, 'text'):
                text = data_sample.text.strip()
            else:
                raise NotImplementedError("Un-supported text annotation format.")

            # fetch video
            video_path = os.path.join(self.img_db_dir, video_id + self.video_fmt) 

            # read with retries
            for i in range(3):
                img_array = self._load_video_from_path_decord(video_path, height=self.resize_size, width=self.resize_size)

                if img_array is not None:
                    break

            if img_array is not None:
                t, c, h, w = img_array.shape

            # Select a random video if the current video was not able to access.
            if img_array is None:
                LOGGER.info(f"Failed to load examples with video: {video_path}. "
                            f"Will randomly sample an example as a replacement.")
                index = random.randint(0, len(self) - 1)
                continue
            else:
                # square crop
                img_array = self.video_random_cropper(img_array)

                if self.randaug:
                    img_array = self.randaug(img_array.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

                break
        else:
            raise RuntimeError(f"Failed to fetch video after {num_retries} retries.")
        
        examples = [{'text_str': text, 'itm_label': 1}]

        return dict(
            img=img_array,  # (T, C, H, W)
            examples=examples,
            n_examples=len(examples),  # used to create image feature copies.
            type='video'
        )

class PretrainImageTextDataset(Dataset):
    def __init__(self, datalist, tokenizer, is_train=True, crop_size=256, resize_size=288, num_frm=4, max_txt_len=40):
        self.datalist = datalist
        self.max_txt_len = max_txt_len

        self.crop_size = crop_size
        self.resize_size = resize_size
        self.num_frms = num_frm

        self.is_train = is_train

        self.transform = transforms.Compose([                        
                transforms.RandomResizedCrop(self.crop_size, scale=(0.2, 1.0), interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                RandomAugment(2,7,isPIL=True,augs=['Identity','Brightness','Sharpness',
                                                'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate'])     
            ])    
        
    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):
        start_time = None
        end_time = None

        # fetch video
        num_retries = 10  # skip error videos

        for _ in range(num_retries):
            data_sample = self.datalist[index]

            try:
                if type(data_sample['caption']) == list:
                    text = random.choice(data_sample['caption'])
                else:
                    text = data_sample['caption']
            
                img_path = data_sample['image']
                img_arr = Image.open(img_path).convert('RGB')   
                img_arr = self.transform(img_arr)
                img_arr = np.asarray(img_arr, dtype=np.float32).transpose(2, 0, 1)
                img_arr = torch.from_numpy(img_arr).unsqueeze(0)
                img_arr = img_arr.repeat(self.num_frms, 1, 1, 1)

            except Exception as e:
                img_arr = None

            if img_arr is not None:
                t, c, h, w = img_arr.shape

            # Select a random video if the current video was not able to access.
            if img_arr is None:
                LOGGER.info(f"Failed to load examples with image: {img_path}. "
                            f"Will randomly sample an example as a replacement.")
                index = random.randint(0, len(self) - 1)
                continue
            else:
                break
        else:
            raise RuntimeError(f"Failed to fetch image after {num_retries} retries.")
        
        examples = [{'text_str': text, 'itm_label': 1}]

        return dict(
            img=img_arr,  # (T, C, H, W)
            examples=examples,
            n_examples=len(examples),  # used to create image feature copies.
            type='img'
        )


class PretrainCollator(object):
    """is_train is kept here if we want to remove
    the randomness during validation of MLM accuracy.
    In that case, instantiate two PretrainCollator"""
    def __init__(self, tokenizer, 
                 mlm=True, mlm_probability=0.15,
                 patch_size=16,
                 mpm=True,
                 max_length=20, is_train=True):
        self.tokenizer = tokenizer
        self.mlm = mlm
        self.mlm_probability = mlm_probability
        self.max_length = max_length
        self.is_train = is_train

        self.mpm = mpm
        self.patch_size = patch_size

    def collate_batch(self, batch):
        if isinstance(batch[0]["img"], torch.Tensor):
            v_collate = default_collate
        else:
            v_collate = img_collate
        visual_inputs = v_collate([d["img"] for d in batch])  # (B, #frm=1 or T, 3, H, W)
        # group data
        text_examples = flat_list_of_lists([d["examples"] for d in batch])
        n_examples_list = [d["n_examples"] for d in batch]  # (B, )
        # group elements data
        batch_enc = self.tokenizer.batch_encode_plus(
            [d["text_str"] for d in text_examples],
            max_length=self.max_length,
            padding='max_length',
            return_tensors="pt",
            truncation=True
        )
        text_input_ids = batch_enc.input_ids  # (B, L)
        text_input_ids_no_mask = text_input_ids.clone()

        if self.mlm:
            text_input_ids, mlm_labels = mask_batch_text_tokens(
                text_input_ids, self.tokenizer,
                is_train=self.is_train)  # make mlm data
        else:
            text_input_ids, mlm_labels = text_input_ids, None
        
        text_input_mask = batch_enc.attention_mask  # (B, L)
        itm_labels = default_collate(
            [d["itm_label"] for d in text_examples])  # (B, )
        
        erase_elems = [random_erase(e, patch_size=self.patch_size) for e in visual_inputs.clone()]

        if self.mpm:
            crop_visual_inputs = v_collate([elems[0] for elems in erase_elems])
            mpm_masks = v_collate([elems[1] for elems in erase_elems])
            context_visual_inputs = v_collate([elems[2] for elems in erase_elems])

            return dict(
                visual_inputs=visual_inputs,  # (B, #frm=1 or T, H, W, C)
                crop_visual_inputs=crop_visual_inputs,
                context_visual_inputs=context_visual_inputs,
                mpm_mask=mpm_masks,
                text_input_ids=text_input_ids_no_mask,
                mlm_text_input_ids=text_input_ids,
                mlm_labels=mlm_labels,
                text_input_mask=text_input_mask, # used to exclude [PAD] token
                itm_labels=itm_labels,
                n_examples_list=n_examples_list,  # used to create image feature copies.
                type=batch[0]['type']
            )
        else:
            return dict(
                visual_inputs=visual_inputs,  # (B, #frm=1 or T, H, W, C)
                text_input_ids=text_input_ids_no_mask,
                mlm_text_input_ids=text_input_ids,
                mlm_labels=mlm_labels,
                text_input_mask=text_input_mask, # used to exclude [PAD] token
                itm_labels=itm_labels,
                n_examples_list=n_examples_list,  # used to create image feature copies.
                type=batch[0]['type']
            )

def random_erase(input_img, patch_size, s_l=0.3, s_h=0.5, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255):
    assert input_img.ndim == 4
    img_t, img_c, img_h, img_w = input_img.shape

    while True:
        s = np.random.uniform(s_l, s_h) * img_h * img_w
        r = np.random.uniform(r_1, r_2)
        w = int(np.sqrt(s / r))
        h = int(np.sqrt(s * r))
        left = np.random.randint(0, img_w)
        top = np.random.randint(0, img_h)

        w = w - w % patch_size
        h = h - h % patch_size

        left = left - left % patch_size
        top = top - top % patch_size

        if left + w <= img_w and top + h <= img_h:
            break

    context_img = input_img.clone()
    context_img[:, :, top: top + h, left: left + w] = 0

    input_img = input_img[:, :, top: top + h, left: left + w]
    pad = (left, img_w - left - w, top, img_h - top - h)
    input_img = torch.nn.functional.pad(input_img, pad=pad, mode='constant', value=0.0)

    img_masks = torch.ones_like(input_img)
    img_masks[:, :, top: top+h, left: left+w] = 0

    img_masks = torch.nn.functional.avg_pool2d(img_masks.float(), kernel_size=(patch_size, patch_size), stride=patch_size)
    img_masks = torch.mean(img_masks, dim=(0, 1))

    return input_img, img_masks, context_img