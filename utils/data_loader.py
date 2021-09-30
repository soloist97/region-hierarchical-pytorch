import pickle

import torch
import h5py
from torch.utils.data import Dataset


__all__ = [
    "CaptionDataset"
]


# === For DenseCap features ====
class CaptionDataset(Dataset):

    def __init__(self, mapping_file_path, visual_features_path, encoded_paragraphs_path, dataset_name):
        super(CaptionDataset, self).__init__()

        assert dataset_name in {'train', 'test', 'val'}

        self.dataset_name = dataset_name
        self.mappings = pickle.load(open(mapping_file_path, 'rb'))
        self.visual_features_path = visual_features_path
        self.encoded_paragraphs_path = encoded_paragraphs_path

    def __getitem__(self, i):

        gid = self.mappings['gid_split_dict'][self.dataset_name][i]

        with h5py.File(self.visual_features_path, 'r') as h:
            visual_feature = torch.tensor(h['feats'][gid], dtype=torch.float)

        with h5py.File(self.encoded_paragraphs_path, 'r') as h:
            encoded_paragraph = torch.tensor(h['encoded_paragraph'][gid])
            length = torch.tensor(h['length'][gid])

        return visual_feature, encoded_paragraph, length

    def __len__(self):
        return len(self.mappings['gid_split_dict'][self.dataset_name])


# === For BottomUp features ====
# class CaptionDataset(Dataset):

#     def __init__(self, mapping_file_path, visual_features_path, encoded_paragraphs_path, dataset_name):
#         super(CaptionDataset, self).__init__()

#         assert dataset_name in {'train', 'test', 'val'}

#         self.dataset_name = dataset_name
#         self.mappings = pickle.load(open(mapping_file_path, 'rb'))
#         self.visual_features_path = visual_features_path
#         self.encoded_paragraphs_path = encoded_paragraphs_path

#     def __getitem__(self, i):

#         gid = self.mappings['gid_split_dict'][self.dataset_name][i]
#         iid = self.mappings['gid2iid'][gid]

#         with np.load(self.visual_features_path+'/'+str(iid)+'.npz', 'r') as f:
#             visual_feature = torch.tensor(f['feat'], dtype=torch.float)

#         with h5py.File(self.encoded_paragraphs_path, 'r') as h:
#             encoded_paragraph = torch.tensor(h['encoded_paragraph'][gid], dtype=torch.long)
#             length = torch.tensor(h['length'][gid], dtype=torch.long)

#         return visual_feature, encoded_paragraph, length

#     def __len__(self):
#         return len(self.mappings['gid_split_dict'][self.dataset_name])
