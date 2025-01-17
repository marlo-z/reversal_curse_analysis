import numpy as np
import itertools
import torch
from torch.utils.data import Dataset, DataLoader

class ReverseDataset(Dataset):
    def __init__(self, sentences, device):
        if isinstance(sentences[0], list):
            self.sentences = list(sentences)
        elif isinstance(sentences[0], dict):
            self.sentences = []
            for entry in sentences:
                if 'forward' in entry:
                    self.sentences.append(entry['forward'])
                if 'backward' in entry:
                    self.sentences.append(entry['backward'])
        else:
            raise ValueError()
        self.device = device

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return torch.tensor(self.sentences[idx]).to(self.device)


def generate_reverse_dataset(
        vocab_size,            # vocab = {1, ..., n}, where n = vocab_size
        word_size,             # (xi,..., xk) form 1 x_word, where k = word_size, xi from x_alphabet
        device, 
        split_ratio=0.7, 
        SEED=0,
    ):
    ''' (Reverse Datset)

    vocab_size = every token is in {0, 1, ..., vocab_size}
    alphabet_size = vocab_size // 4             (only sample from a restricted subset of vocab, otherwise might be too hard to learn)
    word_size = how many tokens form a word
    sample_factor = 2

    x_alphabet = sample alphabet_size tokens from vocab
    y_alphabet = sample alphabet_size tokens from vocab (disjoint from x_tokens)

    x_words = cartesian product (x_alphabet)^word_size, all possible x words
    y_words = cartesian product (y_alphabet)^word_size, all possible y words

    shuffle x_words and y_words

    sample_size = max(len(x_words), sample_factor * alphabet_size)      avoid having too many words
    x_words = sub-sample x_words to sample_size
    y_words = sub-sample y_words to sample_size

    for each i, create sentence pair:
        forward:  x_word[i] --> y_word[i]
        backward: y_word[i] <-- x_word[i]

    train_sentence_raw = 70% of all sentences (depending on split_ratio)
    val_sentences_raw = 30% of all sentences

    for each sentence pair in val_sentences:
        move one of the direction to train_sentences

    train_sentences = train_sentences_raw + 50% of val_sentences_raw
    val_sentences = val_sentences_raw - 50% of val_sentences_raw
    '''
    alphabet_size = vocab_size // 4
    assert vocab_size >= 2*alphabet_size
    sample_factor=2
    np.random.seed(seed=SEED)

    NUM_RESERVED_TOKENS = 2
    right_arrow_idx = 0
    left_arrow_idx = 1

    # sample x_alphabet, y_alphabet from {0, 1, ... n}
    sampled_ids = np.random.choice(vocab_size, (2, alphabet_size), replace=False)
    sampled_ids += NUM_RESERVED_TOKENS                # shift by NUM_RESERVED_TOKENS to skip arrow indices
    x_alphabet = sampled_ids[0]
    y_alphabet = sampled_ids[1]

    # all possible x_words, y_words = cartesian product (x_alphabet)^k, (y_alphabet)^k
    x_words = np.array(list(itertools.product(x_alphabet, repeat=word_size)))
    y_words = np.array(list(itertools.product(y_alphabet, repeat=word_size)))

    # shuffle pairing between x_words and y_words (otherwise model will learn this mapping)
    np.random.shuffle(x_words)
    np.random.shuffle(y_words)

    # NOTE: num_words is no longer used (max num words not constant anymore) 
    max_num_words = round(sample_factor * alphabet_size)
    size = min(len(x_words), max_num_words)

    # sample some words from x_words, y_words
    x_words_idx = np.random.choice(len(x_words), size=size, replace=False)
    y_words_idx = np.random.choice(len(y_words), size=size, replace=False)
    x_words = x_words[x_words_idx]
    y_words = y_words[y_words_idx]

    # pair-up x_words and y_words to form sentences (x1,...,xk) --> (y1,...,yk)
    all_sentences = []
    for x, y in zip(x_words, y_words):
        forward = list(x) + [right_arrow_idx] + list(y)       # x --> y
        backward = list(y) + [left_arrow_idx] + list(x)      # y <-- x
        all_sentences.append(dict(forward=forward, backward=backward))

    # train val split
    if True:                                   # default: train_both_directions = True
        all_sentences = np.array(all_sentences)
        shuffled_idx = np.arange(len(all_sentences))
        np.random.shuffle(shuffled_idx)
        cutoff = round(len(all_sentences) * split_ratio)
        train_idx = shuffled_idx[:cutoff]
        val_idx = shuffled_idx[cutoff:]
        train_sentences = all_sentences[train_idx].tolist()     # will be appending later
        val_sentences = all_sentences[val_idx]

        print("all pairs:", len(all_sentences))

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

        # Val set: each pair of (x->y), (y<-x), move one to train, keep the other in val
        for val_sentence in val_sentences:
            # pop forward from val --> add to train
            if np.random.randint(2): # random 0 or 1
                train_sentences.append(dict(forward=val_sentence.pop('forward')))
            # pop backward from val --> add to train
            else:
                train_sentences.append(dict(backward=val_sentence.pop('backward')))
        train_sentences = np.array(train_sentences)


    num_train = sum([len(entry) for entry in train_sentences])
    num_val = sum([len(entry) for entry in val_sentences])
    # print("After moving half of val to train")
    print("--------------- Data: ----------------")
    print("train:", num_train, "val:", num_val, "sampling factor:", sample_factor)

    train_data = ReverseDataset(train_sentences, device)
    val_data = ReverseDataset(val_sentences, device)

    # for plotting token embedding cosine similarity heatmap
    x_tokens = [word[0] for word in x_words] if word_size == 1 else None
    y_tokens = [word[0] for word in y_words] if word_size == 1 else None

    data_modules = dict(
        train_data=train_data, 
        val_data=val_data, 
        num_train=num_train, 
        num_val=num_val,
        x_tokens=x_tokens,
        y_tokens=y_tokens,
        special_tokens = [0, 1]
    )

    return data_modules


# tesing
if __name__ == '__main__':

    vocab_size = 800                # default setting
    alphabet_size = vocab_size // 4
    word_size = 1                   # default setting
    sample_factor = 2               # fixed
    train_both_directions = True    # default
    device = 'cuda'
    seed = 1234

    data_modules = generate_reverse_dataset(
        vocab_size, 
        word_size, 
        alphabet_size, 
        sample_factor,
        train_both_directions, 
        device, 
        SEED=seed
    )

    train_data = data_modules['train_data']
    val_data = data_modules['val_data']

    train_loader = DataLoader(train_data, batch_size=4)
    val_loader = DataLoader(val_data, batch_size=4)

    for batch in train_loader:
        print(batch)
        breakpoint()