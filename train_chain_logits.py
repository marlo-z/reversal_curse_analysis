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

from datasets.ChainDataset_logits import generate_chain_dataset


def train(args):
    verbose=False
    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.pos_encode_type == 'rotary':
        overwrite_attention_block_with_rotary_pe(device)

    data_dict = generate_chain_dataset(
        args.vocab_size,
        args.word_size,
        device,
        SEED=args.seed
    )

    # for data loader
    train_data = data_dict['train_data']
    val_data = data_dict['val_data']
    num_train = data_dict['num_train']
    num_val = data_dict['num_val']
    
    # for visualizing token embeddings (not used anymore)
    # x_words = data_dict['x_words']
    # y_words = data_dict['y_words']
    # z_words = data_dict['z_words']

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
    train_data_x0y = data_dict['train_data_x0y']
    train_data_y0z = data_dict['train_data_y0z']
    train_data_x1z = data_dict['train_data_x1z']
    val_data_x0y = data_dict['val_data_x0y']
    val_data_y0z = data_dict['val_data_y0z']
    val_data_x1z = data_dict['val_data_x1z']
    train_tokens_x = data_dict['train_tokens_x']
    train_tokens_y = data_dict['train_tokens_y']
    train_tokens_z = data_dict['train_tokens_z']
    val_tokens_x = data_dict['val_tokens_x']
    val_tokens_y = data_dict['val_tokens_y']
    val_tokens_z = data_dict['val_tokens_z']

    train_x0y_loader = DataLoader(train_data_x0y, batch_size=args.batch_size, shuffle=False)
    train_y0z_loader = DataLoader(train_data_y0z, batch_size=args.batch_size, shuffle=False)
    train_x1z_loader = DataLoader(train_data_x1z, batch_size=args.batch_size, shuffle=False)
    
    val_x0y_loader = DataLoader(val_data_x0y, batch_size=args.batch_size, shuffle=False)
    val_y0z_loader = DataLoader(val_data_y0z, batch_size=args.batch_size, shuffle=False)
    val_x1z_loader = DataLoader(val_data_x1z, batch_size=args.batch_size, shuffle=False)


    jobs = {
        'train_x0y': [train_x0y_loader, train_tokens_y],          # data_loader, selected_tokens
        'train_y0z': [train_y0z_loader, train_tokens_z],
        'train_x1z': [train_x1z_loader, train_tokens_z],
        'val_x0y': [val_x0y_loader, val_tokens_y],
        'val_y0z': [val_y0z_loader, val_tokens_z],
        'val_x1z': [val_x1z_loader, val_tokens_z],
    }

    def compute_logits(job_name, data_loader, selected_tokens):
        output_logits = []
        for sentence_batch in data_loader:
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
                selected_logits = [logits_vec[x] for x in selected_tokens]           # M x M 
                selected_logits_batch.append(selected_logits)
            output_logits.extend(selected_logits_batch)

        output_logits = np.array(output_logits)
        print(job_name, output_logits.shape)

        return output_logits

    job_outputs = {}
    for job_name, job_data in jobs.items():
        data_loader = job_data[0]
        selected_tokens = job_data[1]
        job_outputs[job_name] = compute_logits(
            job_name, 
            data_loader, 
            selected_tokens
        )

    # save logits data
    out_dir = args.output_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)  

    for job_name, logits_data in job_outputs.items():
        out_file = f'{job_name}.npy'
        out_path = os.path.join(out_dir, out_file)
        np.save(out_path, logits_data)

    plot(job_outputs, args.output_dir)

    return

def plot(job_outputs, out_dir):
    fig, axs = plt.subplots(2, 3, figsize=(10, 8))

    vmin = min([data.min() for key, data in job_outputs.items()])
    vmax = max([data.max() for key, data in job_outputs.items()])


    im1 = axs[0, 0].imshow(job_outputs['train_x0y'], cmap='inferno', vmin=vmin, vmax=vmax)
    axs[0, 0].set_title('Logits of B given A (Train)')
    axs[0, 0].set_xticks([])
    axs[0, 0].set_yticks([])
    fig.colorbar(im1, ax=axs[0, 0])

    im2 = axs[0, 1].imshow(job_outputs['train_y0z'], cmap='inferno', vmin=vmin, vmax=vmax)
    axs[0, 1].set_title('Logits of C given B (Train)')
    axs[0, 1].set_xticks([])
    axs[0, 1].set_yticks([])
    fig.colorbar(im2, ax=axs[0, 1])

    im3 = axs[0, 2].imshow(job_outputs['train_x1z'], cmap='inferno', vmin=vmin, vmax=vmax)
    axs[0, 2].set_title('Logits of C given A (Train)')
    axs[0, 2].set_xticks([])
    axs[0, 2].set_yticks([])
    fig.colorbar(im3, ax=axs[0, 2])

    im4 = axs[1, 0].imshow(job_outputs['val_x0y'], cmap='inferno', vmin=vmin, vmax=vmax)
    axs[1, 0].set_title('Logits of B given A (Val)')
    axs[1, 0].set_xticks([])
    axs[1, 0].set_yticks([])
    fig.colorbar(im4, ax=axs[1, 0])

    im5 = axs[1, 1].imshow(job_outputs['val_y0z'], cmap='inferno', vmin=vmin, vmax=vmax)
    axs[1, 1].set_title('Logits of C given B (Val)')
    axs[1, 1].set_xticks([])
    axs[1, 1].set_yticks([])
    fig.colorbar(im5, ax=axs[1, 1])

    im6 = axs[1, 2].imshow(job_outputs['val_x1z'], cmap='inferno', vmin=vmin, vmax=vmax)
    axs[1, 2].set_title('Logits of C given A (Val)')
    axs[1, 2].set_xticks([])
    axs[1, 2].set_yticks([])
    fig.colorbar(im6, ax=axs[1, 2])

    plt.tight_layout()

    out_file = f'logits.pdf'
    out_path = os.path.join(out_dir, out_file)
    plt.savefig(out_path)


### Main ###
from rotary_embedding_torch import RotaryEmbedding
from model.Add_Rotary_PE import overwrite_attention_block_with_rotary_pe
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
original_gpt2_attention_forward = GPT2Attention.forward                 # save orginal, might be replaced later


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
    parser.add_argument('--output_dir', default='exp_chain_logits') 

    args = parser.parse_args()
    train(args)

