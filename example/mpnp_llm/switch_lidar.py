import sys
sys.path.append(sys.path[0] + '/../..')  # Adds the parent directory to sys.path
sys.path.append(sys.path[0] + '/../../model/')

from transformers import logging
from data import nuqa
from model import encoder, llm, mpnp_opt, mpnp_bloom
from utils import make_folders
from custom_trainer import SwitchLiDARTrainer
from peft import LoraConfig

logging.set_verbosity_error()

# offline train with rgb modality, llm's kv also trained

make_folders("logs", "saved_models")

llm_name = "facebook/opt-2.7b" # ["facebook/opt-1.3b", "bigscience/bloomz-1b1"]
llm_type = "opt"
rgb_encoder_name = "WinKawaks/vit-small-patch16-224"
extracted_ratio = 1.0
max_input_length = 72
max_output_length = 8
batch_size = 16
num_beams = 4
lr = 2e-5
num_epochs = 8
enable_lora = True
freeze_encoder = True
freeze_llm = True ^ enable_lora
save_model = False
status = "online"
run_validation = False
backward_depth = 13

# prepare encoder
rgb_encoder, rgb_processor = encoder.load_rgb_encoder(
    model_name=rgb_encoder_name,
    mixed_precision=True,
)
lidar_encoder, _ = encoder.load_lidar_encoder(
    model_name="../../model/model_nuscenes_cs_init.pth",
    mixed_precision=True,
)
# prepare llm
llm_backbone, text_tokenizer = llm.load_llm_backbone(
    model_name="saved_models/offline_llm.pth",
    model_type=llm_type,
    mixed_precision=True,
    num_mm_tokens=4,
)
# attach encoder to llm
if llm_backbone.config.model_type == "opt":
    lidar_llm = mpnp_opt.MLM_LIDAR(
        lidar_encoder=lidar_encoder,
        llm_backbone=llm_backbone,
        max_input_length=max_input_length,
        backward_depth=backward_depth,
        freeze_encoder=freeze_encoder,
        freeze_llm=freeze_llm,
    )
    if enable_lora:
        target_modules = []
        N = llm_backbone.config.num_hidden_layers
        for j in range(N - backward_depth, N):
            target_modules.append(f"{j}.self_attn.k_proj")
            target_modules.append(f"{j}.self_attn.v_proj")
        # LoRA is optional    
        peft_config = LoraConfig(
            peft_type="LORA",
            task_type="CAUSAL_LM",
            inference_mode=False,
            r=8, lora_alpha=32, target_modules=target_modules,
            lora_dropout=0.1,
        )
        llm_backbone = llm.apply_lora(
            model=llm_backbone,
            num_mm_tokens=4,
            peft_config=peft_config,
        )
elif llm_backbone.config.model_type == "bloom":
    lidar_llm = mpnp_bloom.MLM_LIDAR(
        lidar_encoder=lidar_encoder,
        llm_backbone=llm_backbone,
        max_input_length=max_input_length,
        backward_depth=backward_depth,
        freeze_encoder=freeze_encoder,
        freeze_llm=freeze_llm,
    )
    if enable_lora:
        target_modules = []
        N = llm_backbone.config.n_layer
        for j in range(N - backward_depth, N):
            target_modules.append(f"{j}.self_attention.query_key_value")
        # LoRA is optional    
        peft_config = LoraConfig(
            peft_type="LORA",
            task_type="CAUSAL_LM",
            inference_mode=False,
            r=8, lora_alpha=32, target_modules=target_modules,
            lora_dropout=0.1,
        )
        llm_backbone = llm.apply_lora(
            model=llm_backbone,
            num_mm_tokens=4,
            peft_config=peft_config,
        )
# prepare dataset
# load night_train
train_loader = nuqa.load_nuqa(
    data_path="nuqamini/night_80dimgaussian7/train", # TODO
    llm_name=llm_name,
    rgb_processor=rgb_processor,
    text_tokenizer=text_tokenizer,
    attention_mask_expand=4,
    max_input_length=max_input_length,
    batch_size=batch_size,
    extracted_ratio=extracted_ratio,
    shuffle=True,
    keep_in_memory=True,
    num_partitions_for_streaming=1,
)
# load night_val
val_loader = nuqa.load_nuqa(
    data_path="nuqamini/night_80dimgaussian7/validation", # TODO
    llm_name=llm_name,
    rgb_processor=rgb_processor,
    text_tokenizer=text_tokenizer,
    attention_mask_expand=4,
    max_input_length=max_input_length,
    batch_size=batch_size,
    extracted_ratio=extracted_ratio,
    shuffle=True,
    keep_in_memory=True,
    num_partitions_for_streaming=1,
)
# construct trainer
my_trainer = SwitchLiDARTrainer(
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=None,
    model=lidar_llm,
    tokenizer=text_tokenizer,
    max_input_length=max_input_length,
    max_output_length=max_output_length,
    num_beams=num_beams,
    load_model_path="",
    save_model_path="",
    enable_lora=enable_lora,
    status=status,
)
# train on night_train, evaluate on night_val
my_trainer.train(
    learning_rate=lr,
    num_epochs=num_epochs,
    run_validation=run_validation,
    save_model=save_model,
)