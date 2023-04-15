from dataclasses import dataclass
import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel, BertConfig
from typing import Optional, Tuple
from transformers.modeling_outputs import ModelOutput
import allennlp.modules.span_extractors.max_pooling_span_extractor as max_pooling_span_extractor
from allennlp.nn.util import get_mask_from_sequence_lengths, masked_log_softmax

@dataclass
class FrameSRLModelOutput(ModelOutput):
    """

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        start_logits (`torch.FloatTensor` of shape `(batch_size, FE_num, sequence_length)`):
            Span-start scores (before SoftMax).
        end_logits (`torch.FloatTensor` of shape `(batch_size, FE_num, sequence_length)`):
            Span-end scores (before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    start_logits: torch.FloatTensor = None
    end_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class BertForFrameSRL(BertPreTrainedModel):
    def __init__(self, config: BertConfig):
        super().__init__(config)
        self.config = config
        self.bert = BertModel(config, add_pooling_layer=False)
        self.start_pointer = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.end_pointer = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.FE_extractor = max_pooling_span_extractor.MaxPoolingSpanExtractor(config.hidden_size)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        self.loss_fct_nll = nn.NLLLoss(ignore_index=-1)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        FE_token_idx: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        context_length: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # eq. (4) in the paper
        sequence_output = outputs[0]

        total_loss = None

        # eq. (6) in the paper
        # step 1: extracting representations for all FEs (B, FE_num, H) avgpooling or maxpooling
        # FE_token_idx (B, FE_num, 2) -> FE_rep (B, FE_num, H) allennlp maxpoolingspanextractor
        # default: maxpooling
        if self.config.FE_pooling == 'max':
            FE_rep = self.FE_extractor(sequence_output, FE_token_idx)
        else:
            FE_rep = self.FE_extractor(sequence_output, FE_token_idx)

        # eq. (7) and (8) in the paper
        
        # step 2: using start/end_pointer to transform FE representations
        # FE_rep (B, FE_num, H) -> FE_start_query (B, FE_num, H) start_pointer
        start_query = self.start_pointer(FE_rep)
        end_query = self.end_pointer(FE_rep)

        # step 3: calculating attention scores for all tokens (B, FE_num, seq_len)
        # FE_start_query (B, FE_num, H), sequence_output (B, seq_len, H) -> start_logits (B, seq_len, FE_num) 
        start_logits = torch.bmm(sequence_output, start_query.permute(0, 2, 1))
        end_logits = torch.bmm(sequence_output, end_query.permute(0, 2, 1))

        # step 4: calculating loss if start_positions and end_positions is not None

        if start_positions is not None and end_positions is not None:
            if context_length is not None:
                max_len = int(sequence_output.shape[1])
                context_mask = get_mask_from_sequence_lengths(context_length.squeeze(), max_len).unsqueeze(-1)
                start_loss = self.loss_fct_nll(masked_log_softmax(start_logits, context_mask, dim=-2), start_positions)
                end_loss = self.loss_fct_nll(masked_log_softmax(end_logits, context_mask, dim=-2), end_positions)
                total_loss = (start_loss + end_loss) / 2
            else:
                start_loss = self.loss_fct(start_logits, start_positions)
                end_loss = self.loss_fct(end_logits, end_positions)
                total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return FrameSRLModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        



