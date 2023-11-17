import torch
from transformers import AutoImageProcessor, ViTModel
from transformers import logging
from rangevit.rangevit_extract import RangeVitEncoderAltWrapper

logging.set_verbosity_error()

def load_rgb_encoder(model_name="facebook/vit-mae-large", mixed_precision=True):
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    if mixed_precision:
        model = ViTModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
        )
        for param in model.parameters():
            param.requires_grad = False  # freeze the model - train adapters later
            if param.ndim == 1:
                # cast the small parameters (e.g. layernorm) to fp32 for stability
                param.data = param.data.to(torch.float32)
    else:
        model = ViTModel.from_pretrained(model_name)
    model.config.output_hidden_states = True
    return model, image_processor


def load_lidar_encoder(model_name="./model_nuscenes_cs_init.pth", mixed_precision=True):
    model = RangeVitEncoderAltWrapper(
        gpu_device_id=0,
        checkpoint_file=model_name,
        orig_model_save_path='./save_path_useless/',
    )
    if mixed_precision:
        # model = model.to(torch.bfloat16)
        for n, param in model.named_parameters():
            param.requires_grad = False  # freeze the model - train adapters later
            if param.ndim == 1 or ("bn" in n):
                # cast the small parameters (e.g. layernorm) to fp32 for stability
                param.data = param.data.to(torch.float32)
            else:
                param.data = param.data.to(torch.bfloat16)

    return model, None

if __name__ == "__main__":
    # model = ViTModel.from_pretrained("facebook/vit-mae-base")
    model = load_lidar_encoder()