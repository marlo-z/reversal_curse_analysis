import copy
import numpy as np
import itertools
import torch
from torch.utils.data import Dataset
import json

class ChainDataset(Dataset):
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

def generate_chain_dataset(
    vocab_size,            # vocab = {1, ..., n}, where n = vocab_size
    word_size,             # (xi,..., xk) form 1 x_word, where k = word_size, xi from x_alphabet
    device, 
    split_ratio=0.7, 
    SEED=0,
):
    '''
    x --> y represented as: x 0 y
    y --> z represented as: y 0 z
    x -->> z represented as x 1 z
    '''
    alphabet_size=vocab_size//4
    sample_factor=2
    assert vocab_size >= 3*alphabet_size
    np.random.seed(seed=SEED)

    # Sample token ids for vocab, partitioned into:
    #       x_alphabet = {...}, y_alphabet = {...}, z_alphabet = {...}
    # NOTE: if we want to have some un-used tokens in vocab but not in x,y,z alphabets,
    #       then need to make sure 3 * alphabet_size << vocab_size
    #       (so that x,y,z alpahbets don't fully saturate the entire vocab)
    vocab = np.random.choice(vocab_size, (3, alphabet_size), replace=False)
    vocab += 2 
    x_alphabet = vocab[0]
    y_alphabet = vocab[1]
    z_alphabet = vocab[2]

    # Create all possible x,y,z words from cartesian product of x,y,z alphabet
    x_words = np.array(list(itertools.product(x_alphabet, repeat=word_size)))
    y_words = np.array(list(itertools.product(y_alphabet, repeat=word_size)))
    z_words = np.array(list(itertools.product(z_alphabet, repeat=word_size)))

    # Shuffle pairing between x,y,z words (otherwise model will learn this mapping)
    np.random.shuffle(x_words)
    np.random.shuffle(y_words)
    np.random.shuffle(z_words)

    # Sample a subset of all x,y,z word (don't want to take all cartesian products of x,y,z alphabet)
    max_num_words = round(sample_factor * alphabet_size)
    size = min(len(x_words), max_num_words)

    x_words_idx = np.random.choice(len(x_words), size=size, replace=False)
    y_words_idx = np.random.choice(len(y_words), size=size, replace=False)
    z_words_idx = np.random.choice(len(z_words), size=size, replace=False)
    x_words = x_words[x_words_idx]
    y_words = y_words[y_words_idx]
    z_words = z_words[z_words_idx]

    # Create sentences by taking x,y,z words that line-up at the same index (after shuffling)
    all_sentences = []
    for x, y, z in zip(x_words, y_words, z_words):
        x0y = list(x) + [0] + list(y)         # x --> y
        y0z = list(y) + [0] + list(z)         # y --> z
        x1z = list(x) + [1] + list(z)         # x -->> z
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
    cutoff = round(len(all_sentences) * split_ratio)
    train_idx = shuffled_idx[:cutoff]
    val_idx = shuffled_idx[cutoff:]
    train_sentences = all_sentences[train_idx].tolist()     # will be appending later
    val_sentences = all_sentences[val_idx].tolist()

    train_sentences_raw = copy.deepcopy(train_sentences)
    val_sentences_raw = copy.deepcopy(val_sentences)

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

    num_train = sum([len(entry) for entry in train_sentences])
    num_val = sum([len(entry) for entry in val_sentences])
    # print("After moving half of val to train")
    print("--------------- Data: ----------------")
    print("train:", num_train, "val:", num_val, "sampling factor:", sample_factor)


    train_data = ChainDataset(train_sentences, device)
    val_data = ChainDataset(val_sentences, device)

    data_dict = dict(
        train_data=train_data, 
        val_data=val_data, 
        num_train=num_train,
        num_val=num_val,
        # x_words=x_words,
        # y_words=y_words,
        # z_words=z_words,
    )

    if word_size == 1:
        # Split tokens into 3 groups 
        train_tokens_x = []             # (maintain order s.t. train_tokens_{x/y/z}[i] are in the same data)
        train_tokens_y = []
        train_tokens_z = [] 
        val_tokens_x = []
        val_tokens_y = []
        val_tokens_z = []

        # Split train sentences into 3 groups
        train_sentences_x0y = []
        train_sentences_y0z = []
        train_sentences_x1z = []
        for triple in train_sentences_raw:
            x0y = triple['x0y']
            y0z = triple['y0z']
            x1z = triple['x1z']
            xy = [x0y[0], x0y[-1]]                      # discard 0,1 token
            yz = [y0z[0], y0z[-1]]
            xz = [x1z[0], x1z[-1]]
            train_sentences_x0y.append(dict(x0y=xy))
            train_sentences_y0z.append(dict(y0z=yz))
            train_sentences_x1z.append(dict(x1z=xz))
            train_tokens_x.append(x0y[0])
            train_tokens_y.append(y0z[0])
            train_tokens_z.append(x1z[-1])

        # Split val sentences into 3 groups
        val_sentences_x0y = []
        val_sentences_y0z = []
        val_sentences_x1z = []
        for triple in val_sentences_raw:
            x0y = triple['x0y']
            y0z = triple['y0z']
            x1z = triple['x1z']
            xy = [x0y[0], x0y[-1]]                       # discard 0,1 token
            yz = [y0z[0], y0z[-1]]
            xz = [x1z[0], x1z[-1]]
            val_sentences_x0y.append(dict(x0y=xy))
            val_sentences_y0z.append(dict(y0z=yz))
            val_sentences_x1z.append(dict(x1z=xz))
            val_tokens_x.append(x0y[0])
            val_tokens_y.append(y0z[0])
            val_tokens_z.append(x1z[-1])

        train_data_x0y = ChainDataset(train_sentences_x0y, device)
        train_data_y0z = ChainDataset(train_sentences_y0z, device)
        train_data_x1z = ChainDataset(train_sentences_x1z, device)
        val_data_x0y = ChainDataset(val_sentences_x0y, device)
        val_data_y0z = ChainDataset(val_sentences_y0z, device)
        val_data_x1z = ChainDataset(val_sentences_x1z, device)

        data_dict['train_data_x0y'] = train_data_x0y
        data_dict['train_data_y0z'] = train_data_y0z
        data_dict['train_data_x1z'] = train_data_x1z
        data_dict['val_data_x0y'] = val_data_x0y
        data_dict['val_data_y0z'] = val_data_y0z
        data_dict['val_data_x1z'] = val_data_x1z
        data_dict['train_tokens_x'] = train_tokens_x
        data_dict['train_tokens_y'] = train_tokens_y
        data_dict['train_tokens_z'] = train_tokens_z
        data_dict['val_tokens_x'] = val_tokens_x
        data_dict['val_tokens_y'] = val_tokens_y
        data_dict['val_tokens_z'] = val_tokens_z

        # input()

    return data_dict

