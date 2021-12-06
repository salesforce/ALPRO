import random
import copy
import os
import torch
import numpy as np
from torch.utils.data.dataloader import default_collate
from src.utils.basic_utils import flat_list_of_lists
from src.utils.load_save import LOGGER
from src.datasets.dataset_base import AlproBaseDataset
from src.datasets.randaugment import TemporalConsistentRandomAugment


class AlproVideoRetrievalDataset(AlproBaseDataset):
    """ This should work for both train and test (where labels are not available).
    datalist: list(tuples)  each tuple is (img_id, list(dicts)),
        each dict
    tokenizer:
    max_img_size: int,
    max_txt_len: int, max text sequence length, including special tokens.
    random_sample_clips: bool, whether using randomly sampled N clips or always use uniformly sampled N clips
    """
    def __init__(self, datalist, tokenizer, img_lmdb_dir,
                 fps=3, num_frm=3, frm_sampling_strategy="rand",
                 max_img_size=1000, max_txt_len=40, itm_neg_size=1,
                 ensemble_n_clips=1, random_sample_clips=True,
                 video_fmt='.mp4', img_db_type='lmdb', is_train=False):
        super(AlproVideoRetrievalDataset, self).__init__(
            datalist, tokenizer, img_lmdb_dir, img_db_type=img_db_type,
            fps=fps, num_frm=num_frm,
            frm_sampling_strategy=frm_sampling_strategy,
            max_img_size=max_img_size, max_txt_len=max_txt_len)
        self.ensemble_n_clips = ensemble_n_clips
        self.num_labels = 2
        self.itm_neg_size = itm_neg_size
        self.random_sample_clips = random_sample_clips
        self.id2data = {
            d["id"]: d for group in datalist for d in group[1]}

        self.is_train = is_train
        self.video_fmt = video_fmt

        if self.is_train:
            self.randaug = TemporalConsistentRandomAugment(N=2, M=5, augs=['Identity', 'Contrast','Brightness','Sharpness', 'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate', 'HorizontalFlip'])     
        else:
            self.randaug = None

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):
        # skip error videos:
        num_retries = 5
        for _ in range(num_retries):
            vid_id, examples = self.datalist[index]  # one video with multiple examples
            if self.ensemble_n_clips > 1:
                raise NotImplementedError('Do not support multiple clips for now.')
            else:
                video_path = os.path.join(self.img_db_dir, vid_id + self.video_fmt) 
                vid_frm_array = self._load_video_from_path_decord(video_path, height=self.max_img_size, width=self.max_img_size)

            # Select a random video if the current video was not able to access.
            if vid_frm_array is None:
                LOGGER.info(f"Failed to load examples with video: {vid_id}. "
                            f"Will randomly sample an example as a replacement.")
                index = random.randint(0, len(self) - 1)
                continue
            sampled_examples = []
            for e in examples:
                s = self._get_single_example(e, index)
                if isinstance(s, dict):
                    sampled_examples.append(s)
                else:
                    sampled_examples.extend(s)
            return dict(
                vid=vid_frm_array,
                examples=sampled_examples,
                n_examples=len(sampled_examples)  # used to create image feature copies.
            )
        else:
            raise RuntimeError(
             f"Failed to fetch video after {num_retries} retries.")

    def _get_single_example(self, data, index):
        examples = []

        text_str = data["txt"]
        itm_label = 1  # positive pair
        examples.append(dict(
            text_str=text_str,
            itm_label=itm_label
        ))
        return examples


class VideoRetrievalCollator(object):
    def __init__(self, tokenizer, max_length=40):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def collate_batch(self, batch):
        # FIXME there is a chance that two captions associated with the same video are batched together. Might need to fix.
        v_collate = default_collate
        visual_inputs = v_collate([d["vid"] for d in batch])  # (B, T, 3, H, W)
        # group data
        text_examples = flat_list_of_lists([d["examples"] for d in batch])
        n_examples_list = [d["n_examples"] for d in batch]  # (B, )
        # group elements data
        # directly concatenate question and option as a single seq.
        text_str_list = [d["text_str"] for d in text_examples]  # (B, )
        batch_enc = self.tokenizer.batch_encode_plus(
            text_str_list,
            max_length=self.max_length,
            padding='max_length',
            return_tensors="pt",
            truncation=True
        )
        text_input_ids = batch_enc.input_ids  # (B, L)
        text_input_mask = batch_enc.attention_mask  # (B, L)

        if "itm_label" in text_examples[0]:
            itm_labels = default_collate(
                [d["itm_label"] for d in text_examples])  # (B, )
        else:
            itm_labels = None

        if "id" in text_examples[0]:
            caption_ids = [d["id"] for d in text_examples]  # (B, )
        else:
            caption_ids = None
        collated_batch = dict(
            visual_inputs=visual_inputs,  # (B, #frm, H, W, C)
            text_input_ids=text_input_ids,
            text_input_mask=text_input_mask,
            caption_ids=caption_ids,  # list(int), example ids,
            labels=itm_labels,
            n_examples_list=n_examples_list  # used to create image feature copies.
        )
        if "vid_id" in batch[0] and len(batch) == 1:
            collated_batch["vid_id"] = batch[0]["vid_id"]
        return collated_batch


class AlproVideoRetrievalEvalDataset(AlproBaseDataset):
    """ Sample by video/image, calculate scores between each video with all the text
    and loop through all the videos. Each batch will only contain a single video,
    but multiple text.

    datalist: list(dict), each dict
    tokenizer:
    max_img_size: int,
    max_txt_len: int, max text sequence length, including special tokens.
    """
    def __init__(self, datalist, tokenizer, img_lmdb_dir,
                 fps=3, num_frm=3, frm_sampling_strategy="rand",
                 max_img_size=1000, max_txt_len=40, ensemble_n_clips=1,
                 video_fmt='.mp4', img_db_type='lmdb'):
        self.ensemble_n_clips = ensemble_n_clips
        super(AlproVideoRetrievalEvalDataset, self).__init__(
            datalist, tokenizer, img_lmdb_dir,
            fps=fps, num_frm=num_frm,
            frm_sampling_strategy=frm_sampling_strategy,
            max_img_size=max_img_size, max_txt_len=max_txt_len,
            img_db_type=img_db_type)
        # id is unique id per caption/example
        for i, d in enumerate(self.datalist):
            assert i == d["id"]
        self.gt_cap_id2vid_id = {d["id"]: d["vid_id"] for d in datalist}
        self.cap_id2data = {d["id"]: d for d in datalist}
        self.batches, self.text_batch = self._prepare_batches_by_video()
        self.id2data = {d["id"]: d for d in self.datalist}

        self.video_fmt = video_fmt

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, index):
        # skip error videos:
        batch = dict()

        batch["vid_id"] = self.batches[index]["vid_id"]  # one video with multiple examples
        batch["examples"] = self.text_batch["examples"]
        batch["n_examples"] = self.text_batch["n_examples"]
        batch["ids"] = self.text_batch["ids"]

        if self.ensemble_n_clips > 1:
            raise NotImplementedError('Do not support multiple clips for now.')
        else:
            # if self.is_train and self.random_sample_clips:
            vid_id = batch["vid_id"]

            video_path = os.path.join(self.img_db_dir, vid_id + self.video_fmt) 
            vid_frm_array = self._load_video_from_path_decord(video_path, height=self.max_img_size, width=self.max_img_size)

        batch["vid"] = vid_frm_array
        return batch

    def _prepare_batches_by_video(self):
        """create batches where each batch contains a single video with multiple text"""
        text_list = []
        for d in self.datalist:
            text_list.append(dict(
                text_str=d["txt"],
                id=d["id"],
            ))
        text_batch = dict(
            vid_id=None,
            examples=text_list,
            n_examples=len(text_list),
            ids=[d["id"] for d in text_list]
        )

        # make 1000 batches for 1000video x 1000text combinations.
        # each batch contains 1video x 1000text
        batches = []
        for idx, d in enumerate(self.datalist):
             #_batch = copy.deepcopy(text_batch)
            _batch = dict()
            _batch["vid_id"] = d["vid_id"]
            batches.append(_batch)
        return batches, text_batch
