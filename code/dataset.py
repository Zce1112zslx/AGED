from dataclasses import dataclass, field
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from transformers import BertTokenizerFast, PreTrainedTokenizerBase
from tqdm.auto import tqdm
from typing import Optional, Union

class FrameSRLDataset(Dataset):
    def __init__(self, data_file, tokenizer):
        super(FrameSRLDataset, self).__init__()
        # print('load data...')
        data_instance_dic = np.load(data_file, allow_pickle=True).item()
        self.data = []
        self.tokenizer = tokenizer
        cnt = 0
        for k, v in data_instance_dic.items():
            self.tokenize_instance(v)

    def tokenize_instance(self, dic):
        # for k, v in dic.items():
        #     print(k, v)
        data_dic = {}
        context = dic['context']
        query = dic['query']
        encodings = self.tokenizer(context, query, is_split_into_words=True, return_length=True)
        data_dic['input_ids'] = encodings['input_ids']
        data_dic['attention_mask'] = encodings['attention_mask']
        data_dic['token_type_ids'] = encodings['token_type_ids']
        data_dic['length'] = encodings['length']
        data_dic['context_length'] = [data_dic['length'][0] - sum(data_dic['token_type_ids']) - 1]
        data_dic['word_ids'] = [x if x is not None else -1 for x in encodings.word_ids()]
        FE_token_idx_start = [encodings.word_to_tokens(x, sequence_index=1).start for x in dic['FE_word_idx']]
        FE_token_idx_end = [encodings.word_to_tokens(x, sequence_index=1).end - 1 for x in dic['FE_word_idx']]
        FE_token_idx = [[s, t] for s, t in zip(FE_token_idx_start, FE_token_idx_end)]
        data_dic['FE_num'] = [len(FE_token_idx)]
        data_dic['FE_token_idx'] = FE_token_idx
        data_dic['start_positions'] = [encodings.word_to_tokens(x-1, sequence_index=0).start if x > 0 else 0 for x in dic['start_positions']]
        data_dic['end_positions'] = [encodings.word_to_tokens(x-1, sequence_index=0).end - 1 if x > 0 else 0 for x in dic['end_positions']]
        data_dic['gt_FE_word_idx'] = dic['gt_FE_word_idx']
        data_dic['gt_start_positions'] = dic['gt_start_positions']
        data_dic['gt_end_positions'] = dic['gt_end_positions']
        data_dic['FE_core_pts'] = dic['FE_core_pts']
        self.data.append(data_dic)
        # for k, v in data_dic.items():
        #     print(k, v)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def subset(self, indices):
        return Subset(self, indices=indices)


    
@dataclass
class DataCollatorForFrameSRL:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        non_padding_keys = set(['input_ids', 'attention_mask', 'token_type_ids', 'length', 'context_length'])
        # padding_keys = batch[0].keys() - tokeinzer_padding_keys
        # tokenizer_features = {}
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
        )

        for k, v in batch.items():
            if k not in non_padding_keys:
                if k == 'FE_token_idx':
                    batch[k] = self.padding_features(v, [0, 0])
                else:
                    batch[k] = self.padding_features(v, -1)

        for k, v in batch.items():
            if k == 'FE_core_pts':
                batch[k] = torch.FloatTensor(v)
            else:
                batch[k] = torch.LongTensor(v)
        return batch

    def padding_features(self, padding_list, padding_element):
        padding_num = -1
        for lst in padding_list:
            if len(lst) > padding_num:
                padding_num = len(lst)
        for i, lst in enumerate(padding_list):
            l = len(lst)
            while l < padding_num:
                padding_list[i].append(padding_element)
                l += 1
        return padding_list

            

if __name__ == '__main__':
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    tokenizer.add_tokens(['<t>', '</t>', '<f>', '</f>', '<r>', '</r>'])
    d = FrameSRLDataset('../data/dev_instance_dic.npy', tokenizer)
    dd = Subset(d, range(8))
    dc = DataCollatorForFrameSRL(tokenizer)
    dl = DataLoader(dd, 4, shuffle=False, collate_fn=dc)
    for b in dl:
        for k, v in b.items():
            print(k, v)
