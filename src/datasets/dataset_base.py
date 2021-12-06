from torch.utils.data import Dataset
from PIL import Image
import io
import av
import torch
import numpy as np
import lmdb
import random
import decord
from decord import VideoReader
from src.datasets.data_utils import (
    ImageResize, ImagePad, image_to_tensor)
from src.utils.load_save import LOGGER

decord.bridge.set_bridge("torch")


class AlproBaseDataset(Dataset):
    """
    datalist: list(dicts)  # lightly pre-processed
        {
        "type": "image",
        "filepath": "/abs/path/to/COCO_val2014_000000401092.jpg",
        "text": "A plate of food and a beverage are on a table.",
                # should be tokenized and digitized first?
        ...
        }
    tokenizer:
    max_img_size: int,
    max_txt_len: int, max text sequence length, including special tokens.
    fps: float, frame per second
    num_frm: #frames to use as input.
    """

    def __init__(self, datalist, tokenizer, img_lmdb_dir, img_db_type='lmdb', fps=3, num_frm=3,
                 frm_sampling_strategy="rand", max_img_size=-1, max_txt_len=20):
        self.fps = fps
        self.num_frm = num_frm
        self.frm_sampling_strategy = frm_sampling_strategy
        self.datalist = datalist
        self.tokenizer = tokenizer
        self.max_txt_len = max_txt_len
        self.max_img_size = max_img_size
        self.img_resize = ImageResize(
            max_img_size,
            "bilinear")  # longer side will be resized to 1000
        self.img_pad = ImagePad(
            max_img_size, max_img_size)  # pad to 1000 * 1000

        self.img_db_type = img_db_type

        assert img_db_type in ['lmdb', 'rawvideo'], "Invalid type for img_db_type, expected {'lmdb', 'rawvideo'}, found {}.".format(img_db_type)

        if self.img_db_type == 'lmdb':
            self.env = lmdb.open(
                img_lmdb_dir, readonly=True,
                create=False)  # readahead=not _check_distributed()
            self.txn = self.env.begin(buffers=True)
        else:
            self.img_db_dir = img_lmdb_dir

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):
        raise NotImplementedError

    def _load_img(self, img_id):
        """Load and apply transformation to image

        Returns:
            torch.float, in [0, 255], (n_frm=1, c, h, w)
        """
        raw_img = load_decompress_img_from_lmdb_value(
            self.txn.get(str(img_id).encode("utf-8"))
        )
        image_np = np.array(raw_img, dtype=np.uint8)  # (h, w, c)
        raw_img_tensor = image_to_tensor(
            image_np, keepdim=False).float()  # (c, h, w) [0, 255]
        resized_img = self.img_resize(raw_img_tensor)
        transformed_img = self.img_pad(
            resized_img)  # (n_frm=1, c, h, w)
        return transformed_img

    @classmethod
    def _is_extreme_aspect_ratio(cls, tensor, max_ratio=5.):
        """ find extreme aspect ratio, where longer side / shorter side > max_ratio
        Args:
            tensor: (*, H, W)
            max_ratio: float, max ratio (>1).
        """
        h, w = tensor.shape[-2:]
        return h / float(w) > max_ratio or h / float(w) < 1 / max_ratio

    def _load_video(self, video_id, num_clips=None, clip_idx=None,
                    safeguard_duration=False, video_max_pts=None):
        """Load and sample frames from video.
        Apply transformation to the sampled frames.

        Sample a clip:
            - random: set num_clips and clip_idx to be None
            - uniform: set num_clips=N, clip_idx=idx. e.g., num_clips=3
                and clip_idx=1 will first segment the video into 3 clips,
                then sample the 2nd clip.

        Returns:
            torch.float, in [0, 255], (n_frm=T, c, h, w)
        """
        assert (num_clips is None) == (clip_idx is None), "Both None, or both not None"
        # (T, C, H, W) [0, 255]
        io_stream = io.BytesIO(self.txn.get(str(video_id).encode("utf-8")))
        raw_sampled_frms, video_max_pts = extract_frames_from_video_binary(
            io_stream,
            target_fps=self.fps,
            num_frames=self.num_frm,
            multi_thread_decode=False,
            sampling_strategy=self.frm_sampling_strategy,
            num_clips=num_clips,
            clip_idx=clip_idx,
            safeguard_duration=safeguard_duration,
            video_max_pts=video_max_pts
        )

        if raw_sampled_frms is None:
            return None, None
        elif self._is_extreme_aspect_ratio(raw_sampled_frms, max_ratio=5.):
            print(
                f"Found extreme aspect ratio for video id {video_id}. Skip it")
            return None, None

        raw_sampled_frms = raw_sampled_frms.float()
        resized_frms = self.img_resize(raw_sampled_frms)
        padded_frms = self.img_pad(resized_frms)
        return padded_frms, video_max_pts


    def _load_video_from_path_decord(self, video_path, height=None, width=None, start_time=None, end_time=None, fps=-1):
        try:
            if not height or not width:
                vr = VideoReader(video_path)
            else:
                vr = VideoReader(video_path, width=width, height=height)

            vlen = len(vr)

            if start_time or end_time:
                assert fps > 0, 'must provide video fps if specifying start and end time.'

                start_idx = min(int(start_time * fps), vlen)
                end_idx = min(int(end_time * fps), vlen)
            else:
                start_idx, end_idx = 0, vlen

            if self.frm_sampling_strategy == 'uniform':
                frame_indices = np.arange(start_idx, end_idx, vlen / self.num_frm, dtype=int)
            elif self.frm_sampling_strategy == 'nlvl_uniform':
                frame_indices = np.arange(start_idx, end_idx, vlen / self.num_frm).astype(int)
            elif self.frm_sampling_strategy == 'nlvl_rand':
                frame_indices = np.arange(start_idx, end_idx, vlen / self.num_frm).astype(int)

                # generate some random perturbations
                strides = [frame_indices[i] - frame_indices[i-1] for i in range(1, len(frame_indices))] + [vlen - frame_indices[-1]]
                pertube = np.array([np.random.randint(0, stride) for stride in strides])

                frame_indices = frame_indices + pertube

            elif self.frm_sampling_strategy == 'rand':
                frame_indices = sorted(random.sample(range(vlen), self.num_frm))
            elif self.frm_sampling_strategy == 'headtail':
                frame_indices_head = sorted(random.sample(range(vlen // 2), self.num_frm // 2))
                frame_indices_tail = sorted(random.sample(range(vlen // 2, vlen), self.num_frm // 2))
                frame_indices = frame_indices_head + frame_indices_tail
            else:
                raise NotImplementedError('Invalid sampling strategy {} '.format(self.frm_sampling_strategy))

            raw_sample_frms = vr.get_batch(frame_indices)
        except Exception as e:
            return None

        raw_sample_frms = raw_sample_frms.permute(0, 3, 1, 2)

        return raw_sample_frms

def img_collate(imgs):
    """
    Args:
        imgs:

    Returns:
        torch.tensor, (B, 3, H, W)
    """
    w = imgs[0].width
    h = imgs[0].height
    tensor = torch.zeros(
        (len(imgs), 3, h, w), dtype=torch.uint8).contiguous()
    for i, img in enumerate(imgs):
        nump_array = np.array(img, dtype=np.uint8)
        if (nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        # (H, W, 3) --> (3, H, W)
        nump_array = np.rollaxis(nump_array, 2)
        tensor[i] += torch.from_numpy(nump_array)
    return tensor
