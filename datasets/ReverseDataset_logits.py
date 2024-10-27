import copy
import numpy as np
import itertools
import torch
from torch.utils.data import Dataset

class ReverseDataset(Dataset):
    def __init__(self, sentences, device):
        self.sentences = []
        for entry in sentences:
            if 'forward' in entry:
                self.sentences.append(entry['forward'])
            if 'backward' in entry:
                self.sentences.append(entry['backward'])

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
    alphabet_size = vocab_size // 4
    assert vocab_size >= 2*alphabet_size
    sample_factor=2
    np.random.seed(seed=SEED)

    right_arrow_idx = 0
    left_arrow_idx = 1

    # sample x_alphabet, y_alphabet from {0, 1, ... n}
    sampled_ids = np.random.choice(vocab_size, (2, alphabet_size), replace=False)
    sampled_ids += 2                # shift by 2 to skip arrow indices
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
        forward = list(x) + [0] + list(y)       # x --> y
        backward = list(y) + [1] + list(x)      # y <-- x
        all_sentences.append(dict(forward=forward, backward=backward))

    # train val split
    if True:                                   # default: train_both_directions = True
        all_sentences = np.array(all_sentences)
        shuffled_idx = np.arange(len(all_sentences))
        np.random.shuffle(shuffled_idx)
        cutoff = round(len(all_sentences) * split_ratio)
        train_idx = shuffled_idx[:cutoff]
        val_idx = shuffled_idx[cutoff:]
        train_sentences = all_sentences[train_idx].tolist()
        val_sentences = all_sentences[val_idx].tolist()

        # save raw train / val sentences, before moving some val to train
        train_sentences_raw = copy.deepcopy(train_sentences)
        val_sentences_raw = copy.deepcopy(val_sentences)

        # which direction is seen during training
        val_sentences_seen = []
        val_sentences_unseen = []

        # Val set: each pair of (x->y), (y<-x), move one to train, keep the other in val
        for val_sentence in val_sentences:
            # pop forward from val --> add to train
            if np.random.randint(2): # random 0 or 1
                forward = val_sentence.pop('forward')
                train_sentences.append(dict(forward=forward))
                
                val_sentences_seen.append(dict(forward=forward))
                val_sentences_unseen.append(val_sentence)               # only backward left

            # pop backward from val --> add to train
            else:
                backward = val_sentence.pop('backward')
                train_sentences.append(dict(backward=backward))

                val_sentences_seen.append(dict(backward=backward))
                val_sentences_unseen.append(val_sentence)               # only forward left 

        train_sentences = np.array(train_sentences)


    num_train = sum([len(entry) for entry in train_sentences])
    num_val = sum([len(entry) for entry in val_sentences])
    # print("After moving half of val to train")
    print("--------------- Data: ----------------")
    print("train:", num_train, "val:", num_val, "sampling factor:", sample_factor)

    train_data = ReverseDataset(train_sentences, device)
    val_data = ReverseDataset(val_sentences, device)

    data_dict = dict(
        train_data=train_data,
        val_data=val_data,
        num_train=num_train,
        num_val=num_val,
    )

    if word_size == 1:

        # Split train sentences into forward vs backward
        train_tokens_x = []
        train_tokens_y = []
        train_sentences_forward = []
        train_sentences_backward = []
        for pair in train_sentences_raw:
            forward = pair['forward']
            x = forward[0]
            y = forward[-1]
            train_tokens_x.append(x)
            train_tokens_y.append(y)
            train_sentences_forward.append(dict(forward=[x, y]))            # remove 0,1 tokens
            train_sentences_backward.append(dict(backward=[y, x]))

        # Split val sentences into seen vs unseen (already done during train-val split above)
        val_seen_output_tokens = []
        val_sentences_seen_new = []
        for sent in val_sentences_seen:
            if 'forward' in sent:
                forward = sent['forward']
                x = forward[0]
                y = forward[-1]
                val_seen_output_tokens.append(y)
                val_sentences_seen_new.append(dict(forward=[x, y]))        # remove 0,1 tokens
            if 'backward' in sent:
                backward = sent['backward']
                y = backward[0]
                x = backward[-1]
                val_seen_output_tokens.append(x)
                val_sentences_seen_new.append(dict(backward=[y, x]))

        val_unseen_output_tokens = []
        val_sentences_unseen_new = []
        for sent in val_sentences_unseen:
            if 'forward' in sent:
                forward = sent['forward']
                x = forward[0]
                y = forward[-1]
                val_unseen_output_tokens.append(y)
                val_sentences_unseen_new.append(dict(forward=[x, y]))       # remove 0,1 tokens
            if 'backward' in sent:
                backward = sent['backward']
                y = backward[0]
                x = backward[-1]
                val_unseen_output_tokens.append(x)
                val_sentences_unseen_new.append(dict(backward=[y, x]))

        train_data_forward = ReverseDataset(train_sentences_forward, device)
        train_data_backward = ReverseDataset(train_sentences_backward, device)
        val_data_seen = ReverseDataset(val_sentences_seen_new, device)
        val_data_unseen = ReverseDataset(val_sentences_unseen_new, device)

        # Current order: train_tokens_x[i] is matched with train_tokens_y[i] for x,y in the same (x -> y)
        '''
        train_tokens_x = []
        train_tokens_y = []
        val_tokens_x = []
        val_tokens_y = []
        for data in train_sentences_raw:
            forward = data['forward']
            train_tokens_x.append(forward[0])
            train_tokens_y.append(forward[-1])
        for data in val_sentences_raw:
            forward = data['forward']
            val_tokens_x.append(forward[0])
            val_tokens_y.append(forward[-1])
        '''

        # Re-order val: group seen tokens and unseen tokens (attempt 1)
        '''
        val_tokens_x_seen = [x for x in val_tokens_x if x in val_tokens_seen_x_set]
        val_tokens_x_unseen = [x for x in val_tokens_x if x not in val_tokens_seen_x_set]
        val_tokens_x_grouped = val_tokens_x_seen + val_tokens_x_unseen              # seen first, then unseen

        val_tokens_y_seen = [y for y in val_tokens_y if y in val_tokens_seen_y_set]
        val_tokens_y_unseen = [y for y in val_tokens_y if y not in val_tokens_seen_y_set]
        val_tokens_y_grouped = val_tokens_y_seen + val_tokens_y_unseen              # seen first, then unseen
        '''

        data_dict['train_data_forward'] = train_data_forward
        data_dict['train_data_backward'] = train_data_backward
        data_dict['val_data_seen'] = val_data_seen
        data_dict['val_data_unseen'] = val_data_unseen

        data_dict['train_tokens_x'] = train_tokens_x
        data_dict['train_tokens_y'] = train_tokens_y
        data_dict['val_seen_output_tokens'] = val_seen_output_tokens
        data_dict['val_unseen_output_tokens'] = val_unseen_output_tokens


    return data_dict

