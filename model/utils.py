import torch

def freeze_model(model, except_name=None):
    for name, param in model.named_parameters():
        if (except_name is not None) and (except_name in name):
            param.requires_grad = True
        else:
            param.requires_grad = False

def freeze_4bit_llm(model):
    for param in model.parameters():
        param.requires_grad = False  # freeze the model - train adapters later
        if param.ndim == 1:
            # cast the small parameters (e.g. layernorm) to fp32 for stability
            param.data = param.data.to(torch.float32)

    # model.gradient_checkpointing_enable()  # reduce number of stored activations
    # model.model.decoder.project_in = lambda x: x.requires_grad_(True)

    class CastOutputToFloat(torch.nn.Sequential):
        def forward(self, x):
            return super().forward(x).to(torch.float32)

    model.lm_head = CastOutputToFloat(model.lm_head)