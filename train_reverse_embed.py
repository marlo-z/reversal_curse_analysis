import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW, SGD
from torch.nn.functional import softmax
from model.GPT2CustomModel import GPT2CustomLMHeadModel
from transformers import GPT2Config
from tqdm import tqdm
import argparse
import multiprocessing

# from dataset_utils import generate_dataset
from datasets.ReverseDataset import generate_reverse_dataset


def train(args):
    assert args.word_size == 1
    verbose=False
    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.pos_encode_type == 'rotary':
        overwrite_attention_block_with_rotary_pe(device)

    data_modules = generate_reverse_dataset(
        args.vocab_size,
        args.word_size,
        device,
        SEED=args.seed
    )

    train_data = data_modules['train_data']
    val_data = data_modules['val_data']
    num_train = data_modules['num_train']
    num_val = data_modules['num_val']
    x_tokens = data_modules['x_tokens']
    y_tokens = data_modules['y_tokens']
    special_tokens = data_modules['special_tokens']

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True)


    # Instantiating a GPT2 model with fresh intialized weights 
    # expand vocab size by 2, to accomadate for left, right arrow tokens
    config = GPT2Config(
        n_embed = args.embed_dim,            # n_heads = 12 (default)
        n_layer = args.n_layers,
        vocab_size=args.vocab_size + 2,
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

            # print("last word probs:", last_word_probs)
            # print(type(last_word_probs))
            # input()

            loss = outputs.loss
            loss.backward()
            optimizer.step()
            if verbose:
                print(f"epoch {epoch} train loss:", loss.item(), "train prob", np.exp(np.mean(word_probs)))

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

        # print(epoch)
        # print(val_token_probs_per_batch)
        # print(val_token_probs)

        # For plotting the evolution of heatmap as training progresses (embeddings should approach near-orthogonal)


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
    # Save probabilities
    out_dir = args.output_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_path = os.path.join(out_dir, f'probs.npy')
    np.save(out_path, results)

    plot_heatmap(
        model, 
        x_tokens, 
        y_tokens, 
        special_tokens, 
        args.output_dir, 
        device
    )

    return

def plot_heatmap(
    model,
    x_tokens,
    y_tokens,
    special_tokens,
    results_dir,
    device
):
    # Cosine Similarity of embeddings
    with torch.no_grad():
        embed_matrix = model.transformer.wte

        all_tokens = torch.LongTensor(special_tokens + x_tokens + y_tokens)         # 200+200+2         (L)
        all_embeds = embed_matrix(all_tokens.to(device))                            # 200+200+2 x 768   (L x D)
        n = all_tokens.size(0)

        all_embeds_normalized = torch.nn.functional.normalize(all_embeds, dim=1)                       # L x D, normalized along dim=1
        cosine_similarity = all_embeds_normalized @ all_embeds_normalized.T       # L x D matmul D x L --> L x L  

    cosine_similarity_np = cosine_similarity.detach().cpu().numpy()

    # Save cosine similarity
    out_path = os.path.join(results_dir, f'cos_sim.npy')
    np.save(out_path, cosine_similarity_np)

    # Plot heatmap
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Cosine Similarity of Embeddings')

    im = ax.imshow(cosine_similarity_np, vmin=-1, vmax=1)
    cbar = fig.colorbar(im, ax=ax)

    out_path = os.path.join(results_dir, f'heatmap.pdf')
    plt.savefig(out_path)
    plt.close()

    return

### Main ###

if __name__ == '__main__':

    # argument parsing
    parser = argparse.ArgumentParser()

    # Model Args
    parser.add_argument('--pos_encode_type', default='absolute', choices=['absolute', 'rotary', 'null'], type=str)
    parser.add_argument('--n_layers', default=24, type=int)
    parser.add_argument('--embed_dim', default=768, type=int)

    # Data Args
    parser.add_argument('--vocab_size', default=800, type=int)
    parser.add_argument('--word_size', default=1, type=int)
    # parser.add_argument('--num_x_tokens', default=100, type=int)        # for reversal ICL dataset
    # parser.add_argument('--num_y_tokens', default=10, type=int)         # for reversal ICL dataset
    # parser.add_argument('--ICL', action='store_true')                   # flag to reversal ICL dataset
    parser.add_argument('--seed', default=1234, type=int)

    # Training Args
    parser.add_argument('--num_epochs', default=3000, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--decay', default=0.9)
    parser.add_argument('--betas', default=(0.9, 0.999))
    parser.add_argument('--loss_whole_sequence', action='store_true')   # whether to apply loss only to last word tokens (default), or whole sequence
    parser.add_argument('--freeze_wte_wpe', action='store_true')        # whether to train word token and positonal embeddings (default), or freeze them
    parser.add_argument('--output_dir', default='exp_reverse_embed') 

    args = parser.parse_args()
    train(args)
