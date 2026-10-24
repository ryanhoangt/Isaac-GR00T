import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import normalize, cross_entropy
from transformers import BertConfig, BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertEncoder


def get_config(args):
    config = BertConfig(
        hidden_size=args["hidden_size"],
        num_hidden_layers=args["num_hidden_layers"],
        num_attention_heads=args["num_attention_heads"],
        intermediate_size=args["intermediate_size"],
        llm_hidden_size=args["llm_hidden_size"],
        args=args,
    )
    return config


def retrieve_from_memory(memory, k, tasks, queries):
    ret_keys, ret_pixel_values, ret_actions = [], [], []
    ret_backbone_features, ret_backbone_attention_masks = [], []

    for task, query in zip(tasks, queries):
        scores = query @ memory[task]["keys"].T
        indices = scores.topk(k).indices.tolist()

        keys = memory[task]["keys"][indices]
        pixel_values = memory[task]["pixel_values"][indices].flatten(0, 1)
        actions = memory[task]["action"][indices]

        ret_keys.append(keys)
        ret_pixel_values.append(pixel_values)
        ret_actions.append(actions)

        features = memory[task]["backbone_features"][indices]
        attention_masks = memory[task]["backbone_attention_mask"][indices]

        ret_backbone_features.extend(list(features))
        ret_backbone_attention_masks.extend(list(attention_masks))

    ret_keys = torch.concat(ret_keys)
    ret_pixel_values = torch.concat(ret_pixel_values)
    ret_actions = torch.concat(ret_actions)

    ret_backbone_features = pad_sequence(
        ret_backbone_features, batch_first=True, padding_side="left")
    ret_backbone_attention_masks = pad_sequence(
        ret_backbone_attention_masks, batch_first=True, padding_side="left")

    return (
        (ret_keys, ret_pixel_values, ret_actions),
        (ret_backbone_features, ret_backbone_attention_masks))


def retrieve_from_memory_traj(memory, k, tasks, queries):
    ret_actions = []

    for task, query in zip(tasks, queries):
        scores = query @ memory[task]["keys"].T
        indices = scores.topk(k).indices.tolist()

        actions = [memory[task]["action"][i] for i in indices]
        ret_actions.append(actions)

    return ret_actions


class RetrieverWrapperWithHead(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.linear = nn.Linear(config.llm_hidden_size, config.hidden_size)
        self.encoder = BertEncoder(config)

    def average_pooling(self, last_hidden_state, attention_mask):
        last_hidden_state[~attention_mask.bool()] = 0
        return last_hidden_state.sum(1) / attention_mask.sum(1)[:, None]

    def forward(self, vla, inputs):
        with torch.no_grad():
            backbone_inputs, _ = vla.prepare_input(inputs)
            pixel_values = backbone_inputs["eagle_pixel_values"]
            input_ids = backbone_inputs["eagle_input_ids"]
            attention_mask = backbone_inputs["eagle_attention_mask"]

            vit_embeds = vla.backbone.eagle_model.extract_feature(pixel_values)
            llm_embeds = vla.backbone.eagle_model.get_input_embeddings()(input_ids)

            B, L, D = llm_embeds.shape
            llm_embeds = llm_embeds.reshape(B * L, D)
            selected = (input_ids.reshape(B * L) == vla.backbone.eagle_model.image_token_index)

            llm_embeds[selected] = llm_embeds[selected] * 0.0 + vit_embeds.reshape(-1, D)
            llm_embeds = llm_embeds.reshape(B, L, D)

        hidden_states = self.linear(llm_embeds)

        extended_attention_mask = self.get_extended_attention_mask(attention_mask, (B, L))
        last_hidden_state = self.encoder(hidden_states, extended_attention_mask)[0]
        vl_embeds = self.average_pooling(last_hidden_state, attention_mask)
        return normalize(vl_embeds, p=2, dim=-1)

    def compute_simclr_loss(self, vl_embeds, temperature):
        y_pred = (vl_embeds @ vl_embeds.T) / temperature
        y_pred.fill_diagonal_(torch.finfo(y_pred.dtype).min)
        y = torch.arange(len(vl_embeds), device=self.device).roll(len(vl_embeds) // 2)
        return cross_entropy(y_pred, y)
