import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import GPT2LMHeadModel, GPT2Model
from typing import Optional, Tuple, Union
import torch.nn.functional as F
import numpy as np

from transformers.modeling_outputs import (
    CausalLMOutputWithCrossAttentions,
    BaseModelOutputWithPastAndCrossAttentions,
)

class GPT2CustomModel(GPT2Model):
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pos_encode_type: Optional[str] = 'absolute',
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)

        # GPT2Attention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and the dtype's smallest value for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)

        # Modification: do not apply default absolute pe using learned embedding vector
        # (Rotary pe will be apply in the overwritten forward method of GPT2Attention Block)
        if (pos_encode_type == 'null') or (pos_encode_type == 'rotary'):          # (null = No PE)
            hidden_states = inputs_embeds
        else:
            assert pos_encode_type == 'absolute'
            hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(
                    block.__call__,
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                    use_cache,
                    output_attentions,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )



class GPT2CustomLMHeadModel(GPT2LMHeadModel):
    ''' 
        Override GPT2LMHeadModel's forward method. 
        Only compute CrossEntropy Loss on predicted logist corresponding to 3rd label token
    '''

    def __init__(self, config):
        super().__init__(config)

        # overwrite transformer 
        self.transformer = GPT2CustomModel(config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        word_size: int = 1,
        attribute_class_word_size: int = 0,     # if is 0, then this is not used
        loss_last_word: bool = True,
        pos_encode_type: Optional[str] = 'absolute',
        compare_special_token = False, 
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            pos_encode_type=pos_encode_type,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        loss = None
        last_word_probs = None
        last_word_first_token_probs = None
        if labels is not None:
            # create loss function
            loss_fct = CrossEntropyLoss()

            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)

            # shift logits and labels so tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()          # B x L-1 x |V|
            shift_labels = labels[..., 1:].contiguous()                 # B x L-1

            # select logits for last word or whole sequence
            k = word_size
            last_word_logits = shift_logits[..., -k:, :].contiguous()   # B x k x |V|
            last_word_labels = shift_labels[..., -k:].contiguous()      # B x k
            sentence_logits = shift_logits                              # B x L-1 x |V|
            sentence_labels = shift_labels                              # B x k

            # select which tokens to apply loss, entire sequence or just last work tokens
            if loss_last_word:
                flattened_logits = last_word_logits.view(-1, last_word_logits.size(-1)) # (B*k) x |V|
                flattened_labels = last_word_labels.view(-1)                            # (B*k)
            else:
                flattened_logits = sentence_logits.view(-1, sentence_logits.size(-1))   # (B*L-1) x |V|
                flattened_labels = sentence_labels.view(-1)                             # (B*L-1)

            loss = loss_fct(flattened_logits, flattened_labels)

            # Softmax logits into probability vector (only plot probs of last word)
            batch_probs_vec = F.softmax(last_word_logits, dim=-1).detach()              # B x k x |V|, softmax along |V| dim
            
            # Debug:
            # print("shift_logits:", shift_logits.size())                 # whole sequence (B x L-1 x |V|)
            # print("shift_labesl:", shift_labels.size()) 
            # print("last_word_logits:", last_word_logits.size())         # last word      (B x k x |V|)
            # print("last_word_labels:", last_word_labels.size())
            # print("flattened_logits:", flattened_logits.size())         # either whole sequence of last word (B*k x |V| or B*L-1 x |V|)
            # print("flattened_labels:", flattened_labels.size())
            # print("batch_probs_vec:", batch_probs_vec.size())           # always last word (B x k x |V|)


            last_word_probs = []
            last_word_first_token_probs = []

            # group probs by special token == 0 or 1 
            token_0_prob = []                                   # len = batch size  
            token_1_prob = []                                   # add 0.0 to avoid nan when taking np.mean with empty list
            
            # Compute probs
            for batch_i in range(batch_probs_vec.size(0)):
                labels = last_word_labels[batch_i]                               # last_word_labels: batch x word_size, labels = [y0,...yk]
                probs_per_token = [batch_probs_vec[batch_i, token_idx, label].item() for token_idx, label in enumerate(labels)]     
                probs_first_token = probs_per_token[0]
                probs_per_word = np.prod(probs_per_token)                        # probs_per_token: word_size
                last_word_probs.append(probs_per_word)
                last_word_first_token_probs.append(probs_first_token)
                assert len(probs_per_token) == (attribute_class_word_size or word_size)

                if compare_special_token:
                    assert (word_size == 1) and (input_ids.size(1) == 3)                # assuming word-size = 1 (sequence length = 3)
                    special_token = input_ids[batch_i][1].detach().cpu().item()
                    if special_token == 0:
                        token_0_prob.append(probs_first_token)
                    else:
                        token_1_prob.append(probs_first_token)
                    

            # log probabilities
            last_word_probs = np.log([x + 1e-50 for x in last_word_probs])
            last_word_first_token_probs = np.log([x + 1e-50 for x in last_word_first_token_probs])
            if compare_special_token:
                token_0_prob = np.log(np.array(token_0_prob) + 1e-50)
                token_1_prob = np.log(np.array(token_1_prob) + 1e-50)
            
            # print(last_word_probs)

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        ), last_word_probs, last_word_first_token_probs, token_0_prob, token_1_prob


''' 
    GPT2LMHeadModel(
        (self.transformer:)
        GPT2Model(
            (self.h = List[])
            GPT2Block(
                GPT2Attention(
                    attention(k, q, v)
                )
            )
        )
    )
'''
