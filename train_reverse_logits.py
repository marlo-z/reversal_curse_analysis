import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW, SGD
from torch.nn.functional import softmax
from model.GPT2CustomModel_logits import GPT2CustomLMHeadModel
from transformers import GPT2Config
from tqdm import tqdm
import argparse

# from dataset_utils import generate_dataset
from datasets.ReverseDataset_logits import generate_reverse_dataset

def train(arg):
    verbose=False
    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.pos_encode_type == 'rotary':
        overwrite_attention_block_with_rotary_pe(device)

    data_dict = generate_reverse_dataset(
        args.vocab_size,
        args.word_size,
        device,
        SEED=args.seed
    )

    train_data = data_dict['train_data']
    val_data = data_dict['val_data'] 
    num_train = data_dict['num_train']
    num_val = data_dict['num_val']

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True)

    # debug
    # print("train:", [x.tolist() for x in train_loader])
    # print("val:", [x.tolist() for x in val_loader])
    # input()

    # Instantiating a GPT2 model with fresh intialized weights 
    # expand vocab size by 2, to accomadate for left, right arrow tokens
    config = GPT2Config(
        n_layer = args.n_layers,
        vocab_size=args.vocab_size + 2,
        pos_encode_type=args.pos_encode_type,
    )
    model = GPT2CustomLMHeadModel._from_config(config)
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

            outputs, additional_out_dict = model(
                input_ids=sentence_batch, 
                labels=sentence_batch, 
                word_size=args.word_size,
                loss_last_word=not args.loss_whole_sequence,
                pos_encode_type=args.pos_encode_type,
            )

            word_probs = additional_out_dict['last_word_probs']
            token_prob = additional_out_dict['last_word_first_token_probs']
            
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
                outputs, additional_out_dict = model(
                    input_ids=sentence_batch, 
                    labels=sentence_batch, 
                    word_size=args.word_size,
                    pos_encode_type=args.pos_encode_type,
                )
                word_probs = additional_out_dict['last_word_probs']
                token_probs = additional_out_dict['last_word_first_token_probs']
            val_word_probs_per_batch.append(np.mean(word_probs))        # take average within batch
            val_token_probs_per_batch.append(np.mean(token_probs))
            if verbose:
                print("val loss:", outputs.loss.item(), 
                      "val word prob:", np.exp(np.mean(word_probs)),
                      "val token prob", np.exp(np.mean(token_probs))
                     )
        
        val_word_probs.append(np.mean(val_word_probs_per_batch))
        val_token_probs.append(np.mean(val_token_probs_per_batch))

    # after training through all epochs, visualize logits
    train_data_forward = data_dict['train_data_forward']                # train: split into forward vs backward
    train_data_backward = data_dict['train_data_backward']
    val_data_seen = data_dict['val_data_seen']                          # val:   split into seen vs unseen
    val_data_unseen = data_dict['val_data_unseen']

    train_loader_forward = DataLoader(train_data_forward, batch_size=args.batch_size, shuffle=False)
    train_loader_backward = DataLoader(train_data_backward, batch_size=args.batch_size, shuffle=False)
    val_loader_seen = DataLoader(val_data_seen, batch_size=args.batch_size, shuffle=False)
    val_loader_unseen = DataLoader(val_data_unseen, batch_size=args.batch_size, shuffle=False)

    train_tokens_x = data_dict['train_tokens_x']
    train_tokens_y = data_dict['train_tokens_y']
    val_seen_output_tokens = data_dict['val_seen_output_tokens']
    val_unseen_output_tokens = data_dict['val_unseen_output_tokens']

    # 1) Train logits forward: x -> y
    train_logits_forward = []
    for sentence_batch in train_loader_forward:
        with torch.no_grad():
            outputs, additional_out_dict = model(
                input_ids=sentence_batch, 
                labels=sentence_batch, 
                word_size=args.word_size,
                pos_encode_type=args.pos_encode_type,
                return_logits=True
            )

        logits_batch = additional_out_dict['logits']                            # M x |V| where M = len(train_tokens_y)
        
        selected_logits_batch = []
        for logits_vec in logits_batch:
            logits_vec = logits_vec.detach().cpu().numpy()
            selected_logits = [logits_vec[x] for x in train_tokens_y]           # M x M 
            selected_logits_batch.append(selected_logits)
        train_logits_forward.extend(selected_logits_batch)

    train_logits_forward = np.array(train_logits_forward)
    print('train logits forward:', train_logits_forward.shape)

    # 2) Train logits backward: y <- x
    train_logits_backward = []
    for sentence_batch in train_loader_backward:
        with torch.no_grad():
            outputs, additional_out_dict = model(
                input_ids=sentence_batch, 
                labels=sentence_batch, 
                word_size=args.word_size,
                pos_encode_type=args.pos_encode_type,
                return_logits=True
            )

        logits_batch = additional_out_dict['logits']                            # M x |V| where M = len(train_tokens_y)
        
        selected_logits_batch = []
        for logits_vec in logits_batch:
            logits_vec = logits_vec.detach().cpu().numpy()
            selected_logits = [logits_vec[x] for x in train_tokens_x]           # M x M 
            selected_logits_batch.append(selected_logits)
        train_logits_backward.extend(selected_logits_batch)

    train_logits_backward = np.array(train_logits_backward)
    print('train logits backward:', train_logits_backward.shape)

    # 3) Val logits seen: seen_input <- or -> seen_output_token
    val_logits_seen = []
    for sentence_batch in val_loader_seen:
        with torch.no_grad():
            outputs, additional_out_dict = model(
                input_ids=sentence_batch, 
                labels=sentence_batch, 
                word_size=args.word_size,
                pos_encode_type=args.pos_encode_type,
                return_logits=True
            )

        logits_batch = additional_out_dict['logits']                            # M x |V| where M = len(train_tokens_y)
            
        selected_logits_batch = []
        for logits_vec in logits_batch:
            logits_vec = logits_vec.detach().cpu().numpy()
            selected_logits = [logits_vec[x] for x in val_seen_output_tokens]   # M x M (select which logits to keep based on val_seen_output_token) 
            selected_logits_batch.append(selected_logits)
        val_logits_seen.extend(selected_logits_batch)

    val_logits_seen = np.array(val_logits_seen)
    print('val logits seen:', val_logits_seen.shape)
        

    # 4) Val logits unseen: unseen_input <- or -> unseen_output_token
    val_logits_unseen = []
    for sentence_batch in val_loader_unseen:
        with torch.no_grad():
            outputs, additional_out_dict = model(
                input_ids=sentence_batch, 
                labels=sentence_batch, 
                word_size=args.word_size,
                pos_encode_type=args.pos_encode_type,
                return_logits=True
            )

        logits_batch = additional_out_dict['logits']                            # M x |V| where M = len(train_tokens_y)
            
        selected_logits_batch = []
        for logits_vec in logits_batch:
            logits_vec = logits_vec.detach().cpu().numpy()
            selected_logits = [logits_vec[x] for x in val_unseen_output_tokens]           # M x M 
            selected_logits_batch.append(selected_logits)
        val_logits_unseen.extend(selected_logits_batch)

    # Debug: also input the sentences from val_loader_seen (append to 'val unseen direction' matrix)
    # for sentence_batch in val_loader_seen:
    #     with torch.no_grad():
    #         outputs, additional_out_dict = model(
    #             input_ids=sentence_batch, 
    #             labels=sentence_batch, 
    #             word_size=args.word_size,
    #             pos_encode_type=args.pos_encode_type,
    #             return_logits=True
    #         )

    #     logits_batch = additional_out_dict['logits']                            # M x |V| where M = len(train_tokens_y)
            
    #     selected_logits_batch = []
    #     for logits_vec in logits_batch:
    #         logits_vec = logits_vec.detach().cpu().numpy()
    #         selected_logits = [logits_vec[x] for x in val_unseen_output_tokens] # debug: input in val seen tokens, but still select the output logits corresponding to val unseen output tokens
    #         selected_logits_batch.append(selected_logits)
    #     val_logits_unseen.extend(selected_logits_batch)
    # END debug

    val_logits_unseen = np.array(val_logits_unseen)
    print('val logits unseen:', val_logits_unseen.shape)

    # Visualize logits
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    # n_train = train_logits_forward.shape[0]             # = len(train_tokens_x) = len(train_tokens_y)
    # n_val = val_logits_seen.shape[0]

    vmin=min([data.min() for data in [train_logits_forward, train_logits_backward, val_logits_seen, val_logits_unseen]])
    vmax=max([data.max() for data in [train_logits_forward, train_logits_backward, val_logits_seen, val_logits_unseen]])

    # Plot each matrix on its corresponding subplot
    im1 = axs[0, 0].imshow(train_logits_forward, cmap='magma', vmin=vmin, vmax=vmax)
    axs[0, 0].set_title('Train forward logits')
    axs[0, 0].set_xticks([])
    axs[0, 0].set_yticks([])
    # axs[0, 0].set_xticklabels(train_tokens_x)
    # axs[0, 0].set_yticklabels(train_tokens_y)
    fig.colorbar(im1, ax=axs[0, 0], orientation='vertical')

    im2 = axs[0, 1].imshow(train_logits_backward, cmap='magma', vmin=vmin, vmax=vmax)
    axs[0, 1].set_title('Train backward logits')
    axs[0, 1].set_xticks([])
    axs[0, 1].set_yticks([])
    # axs[0, 1].set_xticklabels(train_tokens_y)
    # axs[0, 1].set_yticklabels(train_tokens_x)
    fig.colorbar(im2, ax=axs[0, 1], orientation='vertical')

    im3 = axs[1, 0].imshow(val_logits_seen, cmap='magma', vmin=vmin, vmax=vmax)
    axs[1, 0].set_title('Val seen logits')
    axs[1, 0].set_xticks([])
    axs[1, 0].set_yticks([])
    # axs[1, 0].set_xticklabels(val_tokens_x)
    # axs[1, 0].set_yticklabels(val_tokens_y)
    fig.colorbar(im3, ax=axs[1, 0], orientation='vertical')

    im4 = axs[1, 1].imshow(val_logits_unseen, cmap='magma', vmin=vmin, vmax=vmax)
    axs[1, 1].set_title('Val unseen logits')
    axs[1, 1].set_xticks([])
    axs[1, 1].set_yticks([])
    # axs[1, 1].set_xticklabels(val_tokens_y)
    # axs[1, 1].set_yticklabels(val_tokens_x)
    fig.colorbar(im4, ax=axs[1, 1], orientation='vertical')

    plt.tight_layout()

    # save plot
    out_dir=args.output_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    out_file = f'logits.pdf'
    out_path = os.path.join(out_dir, out_file)
    plt.savefig(out_path)

    # save data
    out_file = f'train_forward.npy'
    out_path = os.path.join(out_dir, out_file)
    np.save(out_path, train_logits_forward)

    out_file = f'train_backward.npy'
    out_path = os.path.join(out_dir, out_file)
    np.save(out_path, train_logits_backward)

    out_file = f'val_seen.npy'
    out_path = os.path.join(out_dir, out_file)
    np.save(out_path, val_logits_seen)

    out_file = f'val_unseen.npy'
    out_path = os.path.join(out_dir, out_file)
    np.save(out_path, val_logits_unseen)

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
    parser.add_argument('--seed', default=1234, type=int)

    # Training Args
    parser.add_argument('--num_epochs', default=3000, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--decay', default=0.9)
    parser.add_argument('--betas', default=(0.9, 0.999))
    parser.add_argument('--loss_whole_sequence', action='store_true')   # whether to apply loss only to last word tokens (default), or whole sequence
    parser.add_argument('--freeze_wte_wpe', action='store_true')        # whether to train word token and positonal embeddings (default), or freeze them
    parser.add_argument('--output_dir', default='exp_reverse_logits') 

    args = parser.parse_args()
    train(args)
