import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW, SGD
from torch.nn.functional import softmax
from moodel.GPT2CustomModel import GPT2CustomLMHeadModel
from transformers import GPT2Config
from tqdm import tqdm
import argparse
import multiprocessing

# from dataset_utils import generate_dataset
from datasets.ReverseDataset_ICL import generate_reverse_dataset_ICL


def train(args):
    assert args.word_size == 1
    verbose=False
    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.pos_encode_type == 'rotary':
        overwrite_attention_block_with_rotary_pe(device)

    data_modules = generate_reverse_dataset_ICL(
        args.vocab_size,
        args.num_x_tokens,
        args.num_y_tokens,
        device, 
        SEED=args.seed
    )

    train_data = data_modules['train_data']
    val_data = data_modules['val_data']
    num_train = data_modules['num_train']
    num_val = data_modules['num_val']

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True)



    # Instantiating a GPT2 model with fresh intialized weights 
    # expand vocab size by 2, to accomadate for left, right arrow tokens
    config = GPT2Config(
        n_embed = args.embed_dim,            # n_heads = 12 (default)
        n_layer = args.n_layers,
        vocab_size=args.vocab_size + 3,      # tokens ids are shifted up by 3, since 0,1,2 are special tokens
        pos_encode_type=args.pos_encode_type,
    )
    model = GPT2CustomLMHeadModel._from_config(config)

    # set word embedding matrix to be frozen if neccessary
    if args.freeze_wte_wpe:
        model.transformer.wte.weight.requires_grad = False
        model.transformer.wpe.weight.requires_grad = False

    # sanity check
    original_wte_weights = model.transformer.wte.weight.detach().cpu().clone()
    original_wpe_weights = model.transformer.wpe.weight.detach().cpu().clone()

    model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, betas=args.betas)
    # optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)

    train_word_probs = []    # per epoch
    train_token_probs = []
    val_word_probs = []      # per epoch
    val_token_probs = []

    # training loop
    for epoch in tqdm(range(args.num_epochs)):
        model.train()
        train_word_probs_per_batch = []         # per batch
        train_token_probs_per_batch = []
        for sentence_batch in train_loader:

            ### Original
            optimizer.zero_grad()

            outputs, word_probs, token_prob, _, _ = model(
                input_ids=sentence_batch, 
                labels=sentence_batch, 
                word_size=args.word_size,
                loss_last_word=not args.loss_whole_sequence,
                pos_encode_type=args.pos_encode_type,
            )
            # probs for all samples in current batch
            train_word_probs_per_batch.append(np.mean(word_probs))          # take average within batch
            train_token_probs_per_batch.append(np.mean(token_prob))

            loss = outputs.loss
            loss.backward()
            optimizer.step()
            if verbose:
                print(f"epoch {epoch} train loss:", loss.item(), "train prob", np.exp(np.mean(word_probs)))
            ###


        # after each epoch, take averge across all batches
        train_word_probs.append(np.mean(train_word_probs_per_batch))
        train_token_probs.append(np.mean(train_token_probs_per_batch))

        # after each epoch (gone through all train batchs), do an eval
        model.eval()
        val_word_probs_per_batch = []
        val_token_probs_per_batch = []
        for sentence_batch in val_loader:
            with torch.no_grad():
                outputs, word_probs, token_probs, _, _ = model(
                    input_ids=sentence_batch, 
                    labels=sentence_batch, 
                    word_size=args.word_size,
                    pos_encode_type=args.pos_encode_type,
                )
            val_word_probs_per_batch.append(np.mean(word_probs))        # take average within batch
            val_token_probs_per_batch.append(np.mean(token_probs))
            if verbose:
                print("val loss:", outputs.loss.item(), 
                      "val word prob:", np.exp(np.mean(word_probs)),
                      "val token prob", np.exp(np.mean(token_probs))
                     )
        
        val_word_probs.append(np.mean(val_word_probs_per_batch))
        val_token_probs.append(np.mean(val_token_probs_per_batch))

    # Probabilities during training
    results = np.stack(
        [
            train_word_probs,
            train_token_probs,
            val_word_probs,
            val_token_probs    
        ],
        axis=0
    )


    out_dir = args.output_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_path = os.path.join(out_dir, 'probs.npy')
    np.save(out_path, results)

    plot_probs(os.path.join(args.output_dir, 'probs.npy'))

    return 


def plot_probs(data_path):

    # Load experiment results
    probs = np.load(data_path)     # 4 x num_epochs
    num_epochs = probs.shape[-1]

    train_word_probs = -1*probs[0]
    train_token_probs = -1*probs[1]
    val_word_probs = -1*probs[2]
    val_token_probs = -1*probs[3] 

    # Create Plot
    fig = plt.figure()      # figsize=(10, 6)
    ax = fig.add_subplot(111)

    # Plot train val probs
    ax.plot(np.arange(num_epochs), train_token_probs, linestyle='-', color='b', label='Train')
    ax.plot(np.arange(num_epochs), val_token_probs, linestyle='-', color='r', label='Val')


    # Set labels
    ax.set_title('Negative Log Probability over Epochs')
    ax.set_xlabel('Number of Epochs')
    ax.set_ylabel('Negative Log Probability')
    # ax.tick_params(labelsize=ticks_fontsize)
    ax.grid(True)
    ax.set_ylim(bottom=-5)
    ax.legend(loc='best')

    out_path = data_path.replace('.npy', '.pdf')
    plt.savefig(out_path)

### Main ###

if __name__ == '__main__':

    # argument parsing
    parser = argparse.ArgumentParser()

    # Model Args
    parser.add_argument('--pos_encode_type', default='absolute', choices=['absolute', 'rotary', 'null'], type=str)
    parser.add_argument('--n_layers', default=24, type=int)
    parser.add_argument('--embed_dim', default=768, type=int)

    # Data Args
    parser.add_argument('--vocab_size', default=400, type=int)
    parser.add_argument('--word_size', default=1, type=int)
    parser.add_argument('--num_x_tokens', default=100, type=int)        # for reversal ICL dataset
    parser.add_argument('--num_y_tokens', default=10, type=int)         # for reversal ICL dataset
    parser.add_argument('--ICL', action='store_true')                   # flag to reversal ICL dataset
    parser.add_argument('--seed', default=1234, type=int)

    # Training Args
    parser.add_argument('--num_epochs', default=3000, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--decay', default=0.9)
    parser.add_argument('--betas', default=(0.9, 0.999))
    parser.add_argument('--loss_whole_sequence', action='store_true')   # whether to apply loss only to last word tokens (default), or whole sequence
    parser.add_argument('--freeze_wte_wpe', action='store_true')        # whether to train word token and positonal embeddings (default), or freeze them
    parser.add_argument('--output_dir', default='exp_reverse_ICL') 

    args = parser.parse_args()
    train(args)



