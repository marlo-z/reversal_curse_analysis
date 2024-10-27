import torch
import torch.nn as nn
from typing import Optional, Tuple, Union

from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

def create_forward_with_rotary_pe(rotary_embed):
    '''
        Return the modified forward method of GPT2Attention block,
        which now has access to the rotary_embed object.
    '''
    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        # K,Q,V: [batch_size, n_tokens, embed_dim] = [B, N, 768]
        # Split into 12 heads (config.n_head)
        # K,Q,V: [batch_size, n_heads, n_tokens, embed_dim / n_heads] = [B, 12, N, 64]

        # print("K, Q, V")
        # print(query.size(), key.size(), value.size())

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)
        
        # print("split heads")
        # print(query.size(), key.size(), value.size())

        # add rotary positional embedding to queries and keys
        query = rotary_embed.rotate_queries_or_keys(query)
        key = rotary_embed.rotate_queries_or_keys(key) 
        # Source code:
        # ~/miniconda3/envs/reverse_curse/lib/python3.10/site-packages/rotary_embedding_torch/rotary_embedding_torch.py

        # print("Added Rotary PE:")
        # print(query.size(), key.size(), value.size())

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        if self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask)
        else:
            attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)
    
    return forward

def overwrite_attention_block_with_rotary_pe(device):
    '''
        Replaces GPT2Attention's forward method with modified version
        that applies rotary positional embeddings
    '''
    rotary_embed = RotaryEmbedding(32, device=device)        # input dim = (transformer_dim / n_heads) / 2
    GPT2Attention.forward = create_forward_with_rotary_pe(rotary_embed)