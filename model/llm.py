from transformers import AutoTokenizer, OPTForCausalLM
from transformers import BloomForCausalLM
from peft import PeftModelForCausalLM, PeftConfig
import torch


def load_llm_backbone(
    model_name="facebook/opt-1.3b",
    model_type="",
    mixed_precision=True, 
    num_mm_tokens=None,
):
    """
    Load LLM as the backbone
    
    To work with peft, such as lora, see the example:
    
    ```py
    peft_config = LoraConfig(
        peft_type="LORA", 
        task_type="CAUSAL_LM", 
        inference_mode=False, 
        r=8, lora_alpha=32, target_modules=["q_proj", "v_proj", "k_proj"],
        lora_dropout=0.1,
    )
    model = load_llm_backbone("facebook/opt-1.3b", peft_config)
    ```
    """
    if "opt" in model_name or "opt" in model_type:
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b") # TODO: figure out whether to use "use_fast=False"
    elif "bloom" in model_name or "bloom" in model_type:
        tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-1b1")
    
    if mixed_precision:
        if "opt" in model_name or "opt" in model_type:
            model = MMOPTForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
            )
        elif "bloom" in model_name or "bloom" in model_type:
            model = MMBloomForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
            )
        else:
            raise NotImplementedError
        
        for param in model.parameters():
            param.requires_grad = False  # freeze the model - train adapters later
            if param.ndim == 1:
                # cast the small parameters (e.g. layernorm) to fp32 for stability
                param.data = param.data.to(torch.float32)

        class CastOutputToFloat(torch.nn.Sequential):
            def forward(self, x):
                return super().forward(x).to(torch.float32)

        model.lm_head = CastOutputToFloat(model.lm_head)
    else:
        if "opt" in model_name or "opt" in model_type:
            model = MMOPTForCausalLM.from_pretrained(model_name)
        elif "bloom" in model_name or "bloom" in model_type:
            model = MMBloomForCausalLM.from_pretrained(model_name)
        else:
            NotImplementedError

    model.num_mm_tokens = num_mm_tokens
    
    # Verifying the datatypes.
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes:
            dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items():
        total += v
    for k, v in dtypes.items():
        print(k, v, v / total)

    return model, tokenizer


def apply_lora(model, peft_config, num_mm_tokens):
    return MMOPTPeftModelForCausalLM(model=model, peft_config=peft_config, num_mm_tokens=num_mm_tokens)



class OPTLearnedPositionalEmbedding(torch.nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # OPT is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, attention_mask: torch.LongTensor, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        attention_mask = attention_mask.long()

        attention_mask = attention_mask[:, past_key_values_length:]
        
        # create positions depending on attention_mask
        positions = (torch.cumsum(attention_mask, dim=1).type_as(attention_mask) * attention_mask).long() - 1

        return super().forward(positions + self.offset)



class MMOPTForCausalLM(OPTForCausalLM):
    """
    Override `prepare_inputs_for_generation` in `OPTForCausalLM`.
    Add new arg `num_mm_tokens` meaning the number of external tokens to insert.
    """
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]
    
    def __init__(self, config, num_mm_tokens=None):
        super().__init__(config)
        self.num_mm_tokens = num_mm_tokens
        self.model.decoder.embed_positions = OPTLearnedPositionalEmbedding(config.max_position_embeddings, config.hidden_size)
    
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if attention_mask is not None:
            mm_attention_mask = torch.ones(attention_mask.shape[0], self.num_mm_tokens).to(attention_mask.device)
            attention_mask = torch.cat((mm_attention_mask, attention_mask), dim=1)
            
        next_past = ()
        for layer_past in past_key_values:
            next_past += (tuple(past_state[:, :, :self.num_mm_tokens] for past_state in layer_past),)
        
        if inputs_embeds is not None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": next_past,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs


class MMOPTPeftModelForCausalLM(PeftModelForCausalLM):
    """
    Override `prepare_inputs_for_generation` in `PeftModelForCausalLM`.
    Add new arg `num_mm_tokens` meaning the number of external tokens to insert.
    """
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]
    
    def __init__(self, model, peft_config: PeftConfig, adapter_name="default", num_mm_tokens=None):
        super().__init__(model, peft_config, adapter_name)
        self.num_mm_tokens = num_mm_tokens
    
    def prepare_inputs_for_generation(self, *args, task_ids: torch.Tensor = None, **kwargs):
        model_kwargs = self.base_model_prepare_inputs_for_generation(*args, **kwargs)
        return model_kwargs
    
    def base_model_prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if attention_mask is not None:
            mm_attention_mask = torch.ones(attention_mask.shape[0], self.num_mm_tokens).to(attention_mask.device)
            attention_mask = torch.cat((mm_attention_mask, attention_mask), dim=1)
            
        next_past = ()
        for layer_past in past_key_values:
            next_past += (tuple(past_state[:, :, :self.num_mm_tokens] for past_state in layer_past),)
        
        if inputs_embeds is not None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": next_past,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs


class MMBloomForCausalLM(BloomForCausalLM):
    """
    Override `prepare_inputs_for_generation` in `BloomForCausalLM`.
    Add new arg `num_mm_tokens` meaning the number of external tokens to insert.
    """
    _tied_weights_keys = ["lm_head.weight"]
    
    def __init__(self, config, num_mm_tokens=None):
        super().__init__(config)
        self.num_mm_tokens = num_mm_tokens
    
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if attention_mask is not None:
            mm_attention_mask = torch.ones(attention_mask.shape[0], self.num_mm_tokens).to(attention_mask.device)
            attention_mask = torch.cat((mm_attention_mask, attention_mask), dim=1)
            
        next_past = ()
        for layer_past in past_key_values:
            next_past += ((layer_past[0][:, :, :self.num_mm_tokens], layer_past[1][:, :self.num_mm_tokens, :]),)
        
        if inputs_embeds is not None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": next_past,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs


if __name__ == "__main__":
    model = OPTForCausalLM.from_pretrained("facebook/opt-125m")
    for k, (n, p) in enumerate(model.named_parameters()):
        print(k, n)