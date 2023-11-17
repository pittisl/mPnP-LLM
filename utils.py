import os
import json


def my_bool(s):
    return s != 'False'

def make_folders(*paths):
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)

def save_config(args, path):
    # Convert the argparse namespace object to a dictionary
    config_dict = vars(args)
    # Write the configurations to the JSON file
    with open(path, "w") as f:
        json.dump(config_dict, f, indent=4)

def compute_f1_score(predicted_answers, target_answers):
    f1_scores = []
    
    for predicted, target in zip(predicted_answers, target_answers):
        predicted_tokens = predicted.lower().split()
        target_tokens = target.lower().split()

        common_tokens = set(predicted_tokens) & set(target_tokens)
        precision = len(common_tokens) / len(predicted_tokens) if len(predicted_tokens) > 0 else 0
        recall = len(common_tokens) / len(target_tokens) if len(target_tokens) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        f1_scores.append(f1_score)

    return sum(f1_scores) / len(f1_scores)

def compute_backward_depth_opt(image_encoder, llm_backbone, max_input_length, rho):
    # Not 100% precise, ignore some trivial operations 
    L = image_encoder.config.num_hidden_layers
    p = image_encoder.config.patch_size
    n = (224 / p)**2
    d = image_encoder.config.hidden_size
    vit_flops = L * (12 * n * d**2 + 2 * n**2 * d) + 3 * p**2 * d
    
    L = llm_backbone.config.num_hidden_layers
    n = max_input_length + image_encoder.config.num_hidden_layers
    d = llm_backbone.config.hidden_size
    v = llm_backbone.config.vocab_size
    llm_flops = L * (12 * n * d**2 + 2 * n**2 * d)
    embed_flops = n * v * d
    
    k = round(((3 * rho - 1) * (llm_flops + vit_flops) - embed_flops) / (llm_flops - embed_flops) * L)
    
    if k < 1:
        print("k is less than 1, clipped to 1.")
        k = 1
    if k > L:
        print(f"k is greater than {L}, clipped to {L}.")
        k = L
    
    true_rho = (1 + (k / L * (llm_flops - embed_flops) + embed_flops) / (llm_flops + vit_flops)) / 3
    
    print(f"User specified rho={100 * rho} %, expected achievable rho={100 * true_rho} %")
    return k

def compute_backward_depth_bloom(image_encoder, llm_backbone, max_input_length, rho):
    # Not 100% precise, ignore some trivial operations 
    L = image_encoder.config.num_hidden_layers
    p = image_encoder.config.patch_size
    n = (224 / p)**2
    d = image_encoder.config.hidden_size
    vit_flops = L * (12 * n * d**2 + 2 * n**2 * d) + 3 * p**2 * d
    
    L = llm_backbone.config.n_layer
    n = max_input_length + image_encoder.config.num_hidden_layers
    d = llm_backbone.config.hidden_size
    v = llm_backbone.config.vocab_size
    llm_flops = L * (12 * n * d**2 + 2 * n**2 * d)
    embed_flops = n * v * d
    
    k = round(((3 * rho - 1) * (llm_flops + vit_flops) - embed_flops) / (llm_flops - embed_flops) * L)
    
    if k < 1:
        print("k is less than 1, clipped to 1.")
        k = 1
    if k > L:
        print(f"k is greater than {L}, clipped to {L}.")
        k = L
    
    true_rho = (1 + (k / L * (llm_flops - embed_flops) + embed_flops) / (llm_flops + vit_flops)) / 3
    
    print(f"User specified rho={100 * rho} %, expected achievable rho={100 * true_rho} %")
    return k