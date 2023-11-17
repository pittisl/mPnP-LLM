import sys
sys.path.append(sys.path[0] + '/../..')  # Adds the parent directory to sys.path
sys.path.append(sys.path[0] + '/../../model/')

from transformers import logging
from data import nuqa
from model import encoder, llm, mpnp_opt, mpnp_bloom
from utils import make_folders
from custom_trainer import OfflineTrainer

logging.set_verbosity_error()

# offline train with rgb modality, llm's kv also trained

make_folders("logs", "saved_models")

llm_name = "facebook/opt-2.7b" # ["facebook/opt-1.3b", "bigscience/bloomz-1b1"]
rgb_encoder_name = "WinKawaks/vit-small-patch16-224"
extracted_ratio = 1.0
max_input_length = 72
max_output_length = 8
batch_size = 16
num_beams = 4
lr = 2e-5
num_epochs = 8
freeze_encoder = True
freeze_llm = False
save_model = True
status = "offline"
run_validation = False
backward_depth = 13

# prepare encoder
rgb_encoder, rgb_processor = encoder.load_rgb_encoder(
    model_name=rgb_encoder_name,
    mixed_precision=True,
)
# prepare llm
llm_backbone, text_tokenizer = llm.load_llm_backbone(
    model_name=llm_name,
    mixed_precision=True,
    num_mm_tokens=4*6,
)
# attach encoder to llm
if llm_backbone.config.model_type == "opt":
    rgb_llm = mpnp_opt.MLM_RGB(
        rgb_encoder=rgb_encoder,
        llm_backbone=llm_backbone,
        max_input_length=max_input_length,
        backward_depth=backward_depth,
        freeze_encoder=freeze_encoder,
        freeze_llm=freeze_llm,
    )
elif llm_backbone.config.model_type == "bloom":
    rgb_llm = mpnp_bloom.MLM_RGB(
        rgb_encoder=rgb_encoder,
        llm_backbone=llm_backbone,
        max_input_length=max_input_length,
        backward_depth=backward_depth,
        freeze_encoder=freeze_encoder,
        freeze_llm=freeze_llm,
    )
# prepare dataset
# load day_train
train_loader = nuqa.load_nuqa(
    data_path="nuqamini/day/train", # TODO
    llm_name=llm_name,
    rgb_processor=rgb_processor,
    text_tokenizer=text_tokenizer,
    attention_mask_expand=4*6,
    max_input_length=max_input_length,
    batch_size=batch_size,
    extracted_ratio=extracted_ratio,
    shuffle=True,
    keep_in_memory=True,
    num_partitions_for_streaming=1,
)
# load day_val
val_loader = nuqa.load_nuqa(
    data_path="nuqamini/day/validation", # TODO
    llm_name=llm_name,
    rgb_processor=rgb_processor,
    text_tokenizer=text_tokenizer,
    attention_mask_expand=4*6,
    max_input_length=max_input_length,
    batch_size=batch_size,
    extracted_ratio=extracted_ratio,
    shuffle=True,
    keep_in_memory=True,
    num_partitions_for_streaming=1,
)
# load night_val
test_loader = nuqa.load_nuqa(
    data_path="nuqamini/night_80dimgaussian7/validation", # TODO
    llm_name=llm_name,
    rgb_processor=rgb_processor,
    text_tokenizer=text_tokenizer,
    attention_mask_expand=4*6,
    max_input_length=max_input_length,
    batch_size=batch_size,
    extracted_ratio=extracted_ratio,
    shuffle=True,
    keep_in_memory=True,
    num_partitions_for_streaming=1,
)
# construct trainer
my_trainer = OfflineTrainer(
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader[0],
    model=rgb_llm,
    tokenizer=text_tokenizer,
    max_input_length=max_input_length,
    max_output_length=max_output_length,
    num_beams=num_beams,
    load_model_path="",
    save_model_path="",
    status=status,
)
# train on day_train, evaluate on day_val
my_trainer.train(
    learning_rate=lr,
    num_epochs=num_epochs,
    run_validation=run_validation,
    save_model=save_model,
)
# evaluate on night_val
my_trainer.evaluate()
