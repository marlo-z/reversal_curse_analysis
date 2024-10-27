import numpy as np
import itertools
import torch
from torch.utils.data import Dataset

class ReverseDataset(Dataset):
    def __init__(self, sentences, device):
        self.sentences = sentences
        # for entry in sentences:
        #     if 'forward' in entry:
        #         self.sentences.append(entry['forward'])
        #     if 'backward' in entry:
        #         self.sentences.append(entry['backward'])

        self.device = device

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return torch.tensor(self.sentences[idx]).to(self.device)


def generate_reverse_dataset_ICL(
        vocab_size,            # vocab = {1, ..., n}, where n = vocab_size
        num_x_tokens,
        num_y_tokens,
        device, 
        split_ratio=0.7, 
        SEED=0,
    ):
    ''' (Reverse Datasset with In Context Learning)

    x_tokens: sample 100 tokens from vocab  (based on num_x_tokens)
    y_tokens: sample 10 tokens from vocab   (disjoint from x_tokens)

    for each xi in x_tokens:
        create 10 candiate sentences using each y token
        [
            xi --> y1  <--> y1 <-- xi
                       ...
            xi --> y10 <--> y10 <-- xi
        ]

        sample 3 sentences --> add to val
        sample 7 sentences --> add to train     (based on split_ratio)

    Loss is computed on whole sequence, but probs visualized for last token only
    '''
    np.random.seed(seed=SEED)

    special_tokens = {
        '-->' : 0,
        '<--' : 1,
        '<->' : 2,
    }

    assert num_x_tokens + num_y_tokens + len(special_tokens) <= vocab_size
    sampled_tokens = np.random.choice(vocab_size, num_x_tokens + num_y_tokens, replace=False)
    sampled_tokens += len(special_tokens)

    x_tokens = sampled_tokens[:num_x_tokens]
    y_tokens = sampled_tokens[num_x_tokens:]

    train_sentences = []
    val_sentences = []
    for x in x_tokens:
        num_train = int(num_y_tokens * split_ratio)
        sentences = [
            [x, 0, y, 2, y, 1, x] for y in y_tokens
        ]
        shuffled = np.random.permutation(sentences)
        train_sentences.extend(shuffled[:num_train])
        val_sentences.extend(shuffled[num_train:])


    # print("After moving half of val to train")
    print("--------------- Data: ----------------")
    print("x tokens:", num_x_tokens, "y tokens:", num_y_tokens, "train:", len(train_sentences), "val:", len(val_sentences))

    train_data = ReverseDataset(train_sentences, device)
    val_data = ReverseDataset(val_sentences, device)

    data_modules = dict(
        train_data=train_data, 
        val_data=val_data, 
        num_train=len(train_sentences), 
        num_val=len(val_sentences),
    )

    return data_modules
