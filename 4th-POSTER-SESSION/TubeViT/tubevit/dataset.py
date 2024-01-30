from typing import Callable, Optional, Tuple

from torch import Tensor
from torchvision.datasets import UCF101

import os


class MyUCF101(UCF101):
    def __init__(self, transform: Optional[Callable] = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.transform = transform

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        video, audio, info, video_idx = self.video_clips.get_clip(idx)
        label = self.samples[self.indices[video_idx]][1]

        if self.transform is not None:
            video = self.transform(video)

        return video, label
    
    def _select_fold(self, video_list, annotation_path, fold, train):
        name = "train" if train else "test"
        name = "{}list{:02d}.txt".format(name, fold)
        f = os.path.join(annotation_path, name)
        selected_files = []
        with open(f, "r") as fid:
            data = fid.readlines()
            data = [x.strip().split("\t") for x in data]
            data = [os.path.join(self.root, x[0]) for x in data]
            selected_files.extend(data)
        selected_files = set(selected_files)
        indices = [i for i in range(len(video_list)) if video_list[i] in selected_files]
        return indices