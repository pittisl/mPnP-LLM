import torch
from train.base_trainer import BaseTrainerForVQAModel


class OfflineTrainer(BaseTrainerForVQAModel):
    def __init__(
        self, 
        train_loader, 
        val_loader,
        test_loader, 
        model,
        tokenizer,
        max_input_length,
        max_output_length,
        num_beams,
        load_model_path,
        save_model_path,
        status='offline',
    ):
        super().__init__(
            train_loader, 
            val_loader,
            test_loader, 
            model,
            tokenizer,
            max_input_length,
            max_output_length,
            num_beams,
            load_model_path,
            save_model_path,
            status,
        )
        self.offline_param_group = [
            {'params': self.model.rgb_key_aligner.parameters()},
            {'params': self.model.rgb_value_aligner.parameters()},
            {"params": [p for n, p in self.model.llm_backbone.named_parameters() if ("query_key_value" in n) or ("k_proj" in n) or ("v_proj" in n)]},
            {'params': self.model.linker_weights, "lr": 1e-1},
        ]
        
    def _save_model(self):
        submodules_to_save = {
            'rgb_key_aligner': self.model.rgb_key_aligner,
            'rgb_value_aligner': self.model.rgb_value_aligner,
        }
        state_dict_to_save = {key: value.state_dict() for key, value in submodules_to_save.items()}
        state_dict_to_save['linker_weights'] = self.model.linker_weights
        # We want to save the ffn and linker weights
        torch.save(state_dict_to_save, "saved_models/offline_ffn_links.pth")
        # We want to save the llm backbone where k, v projectors are trained
        self.model.llm_backbone.save_pretrained("saved_models/offline_llm.pth")


class SwitchLiDARTrainer(BaseTrainerForVQAModel):
    def __init__(
        self, 
        train_loader, 
        val_loader,
        test_loader, 
        model,
        tokenizer,
        max_input_length,
        max_output_length,
        num_beams,
        load_model_path,
        save_model_path,
        enable_lora=False,
        status='online',
    ) -> None:
        super().__init__(
            train_loader, 
            val_loader,
            test_loader, 
            model,
            tokenizer,
            max_input_length,
            max_output_length,
            num_beams,
            load_model_path,
            save_model_path,
            status
        )
        self.online_param_group = [
            {'params': self.model.lidar_key_aligner.parameters()},
            {'params': self.model.lidar_value_aligner.parameters()},
            {'params': self.model.linker_weights, "lr": 1e-1},
        ]
        if enable_lora:
            self.online_param_group += [{'params': self.model.llm_backbone.parameters()}]
    def _load_model(self):
        # load ffn and linker weights
        # the llm weights should be loaded externally
        self.model.load_state_dict(torch.load("saved_models/offline_ffn_links.pth"), strict=False)
        

class AddLiDARTrainer(BaseTrainerForVQAModel):
    def __init__(
        self, 
        train_loader, 
        val_loader,
        test_loader, 
        model,
        tokenizer,
        max_input_length,
        max_output_length,
        num_beams,
        load_model_path,
        save_model_path,
        enable_lora=False,
        status='online',
    ) -> None:
        super().__init__(
            train_loader, 
            val_loader,
            test_loader, 
            model,
            tokenizer,
            max_input_length,
            max_output_length,
            num_beams,
            load_model_path,
            save_model_path,
            status
        )
        self.online_param_group = [
            {'params': self.model.lidar_key_aligner.parameters()},
            {'params': self.model.lidar_value_aligner.parameters()},
            {'params': self.model.linker_weights, "lr": 1e-1},
        ]
        if enable_lora:
            self.online_param_group += [{'params': self.model.llm_backbone.parameters()}]
    def _load_model(self):
        # load ffn and linker weights
        # the llm weights should be loaded externally
        self.model.load_state_dict(torch.load("saved_models/offline_ffn_links.pth"), strict=False)