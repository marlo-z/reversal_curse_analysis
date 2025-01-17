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
    # vocab_size,           
    # word_size,
    # alphabet_size,
    # num_words,
    # sample_factor,
    # train_both_directions,
    # device, 
    # split_ratio=0.7, 
    # SEED=0,
    # data_name=None,
):
    '''
    x --> y represented as: x 0 y
    y --> z represented as: y 0 z
    x -->> z represented as x 1 z
    '''
    alphabet_size = vocab_size // 4
    sample_factor = 2
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
    val_sentences = all_sentences[val_idx]


    # DEBUG: for word_size=3 reduce # of training data to make sure training loss = 0
    max_num_train = 100
    max_num_val = 50
    if word_size == 3:
        print("Orignal num train pairs:", len(train_sentences))     # 280
        train_sentences = train_sentences[:max_num_train]
        print("Reduced num train pairs:", len(train_sentences))

        print("Orignal num val pairs:", len(val_sentences))         # 120
        val_sentences = val_sentences[:max_num_val]
        print("Reduced num val pairs:", len(val_sentences))
    
    # train sentences = 3 * 100 + 2 * 50 = 400
    # val sentences   = 1 * 50           = 50


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
        x_words=x_words,
        y_words=y_words,
        z_words=z_words,
    )

    return data_dict
