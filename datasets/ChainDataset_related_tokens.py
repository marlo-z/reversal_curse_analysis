import numpy as np
import itertools
import torch
from torch.utils.data import Dataset
import json

class ChainDataset_v2(Dataset):
    def __init__(self, sentences, device):
        self.sentences = []
        for entry in sentences:
            if 'x0y' in entry:
                self.sentences.append(entry['x0y'])
            if 'y0z' in entry:
                self.sentences.append(entry['y0z'])
            if 'x1z' in entry:
                self.sentences.append(entry['x1z'])

        self.device = device

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return torch.tensor(self.sentences[idx]).to(self.device)

def generate_chain_dataset_v2(
    vocab_size,
    alphabet_size,              # num tokens sample from vocab
    num_train,                  # number of triples (x0y, y0z, x1z)
    num_val,
    device='cuda', 
    SEED=0,
):
    '''
    Fix 3 tokens x, y, z. Sample random tokens {i...}
    Sentences: xi --> yi, yi --> zi, xi -->> zi
    '''
    np.random.seed(seed=SEED)

    assert vocab_size >= alphabet_size + 3    # 3 additional fixed tokens

    vocab = np.random.choice(vocab_size, alphabet_size + 3, replace=False)
    vocab += 2

    i_tokens = vocab[3:]        # variable tokens {i...}
    x = vocab[0]                 # fixed x, y, z
    y = vocab[1]
    z = vocab[2]

    # Create sentences by taking x,y,z words that line-up at the same index (after shuffling)
    all_sentences = []
    for i in i_tokens:
        x0y = [x, i, 0, y, i]
        y0z = [y, i, 0, z, i]
        x1z = [x, i, 1, z, i]
        all_sentences.append(
            dict(
                x0y=x0y,
                y0z=y0z,
                x1z=x1z,
            )
        )

    # Train val split
    all_sentences = np.array(all_sentences)
    shuffled_idx = np.arange(len(all_sentences))
    np.random.shuffle(shuffled_idx)

    # after shuffling, take first chunck to be train, next chunck to be val
    assert len(all_sentences) >= num_train + num_val
    train_sentences = all_sentences[:num_train].tolist()     # will be appending later
    val_sentences = all_sentences[num_train : num_train + num_val]

    # For each val pairing of (x0y, y0z, x1z), move (x0y, y0z) to training data
    for val_dict in val_sentences:
        x0y = val_dict.pop('x0y')
        y0z = val_dict.pop('y0z')
        train_sentences.append(
            dict(
                x0y=x0y,
                y0z=y0z,
            )
        )
    train_sentences = np.array(train_sentences)

    num_train_sentences = sum([len(entry) for entry in train_sentences])
    num_val_sentences = sum([len(entry) for entry in val_sentences])
    print("--------------- Data: ----------------")
    print("train sentences:", num_train_sentences, "val sentences:", num_val_sentences)        
    # NOTE: num_train_sentence = 3*num_train_triples + 2*num_val_triples
    #       num_val_sentence   = num_val_triples


    # from pprint import pprint
    # print("train:")
    # pprint(train_sentences)
    # print("val:")
    # pprint(val_sentences)
    # input()

    train_data = ChainDataset_v2(train_sentences, device)
    val_data = ChainDataset_v2(val_sentences, device)

    data_dict = dict(
        train_data=train_data, 
        val_data=val_data, 
        num_train_sentences=num_train_sentences,
        num_val_sentences=num_val_sentences,
    )

    return data_dict



