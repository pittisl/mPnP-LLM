import torch
import evaluate
import time
from tqdm import tqdm
# from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup #, Adafactor
# import bitsandbytes as bnb
from utils import compute_f1_score


class BaseTrainerForVQAModel:
    """
    To customize, override `self.online_param_group`, `self.offline_param_group`, 
    `self._load_model()`, and `self._save_model()`.
    
    """
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
        status,
    ) -> None:
        
        self.train_loader = train_loader
        self.val_loader = val_loader[0]
        self.test_loader = test_loader
        self.model = model
        self.tokenizer = tokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.logger = SummaryWriter(flush_secs=10)
        self.exact_match = evaluate.load('exact_match')
        self.interval = 200
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.num_beams = num_beams
        self.status = status
        self.load_model_path = load_model_path
        self.save_model_path = save_model_path
        
        # The trainable parameter set can be overridden
        if self.status == "offline":
            if hasattr(self.model, 'rgb_key_aligner'):
                self.offline_param_group = [
                    {'params': self.model.rgb_key_aligner.parameters()},
                    {'params': self.model.rgb_value_aligner.parameters()},
                    {"params": [p for n, p in self.model.llm_backbone.named_parameters() if ("k" in n) or ("v" in n)]},
                ]
            if hasattr(self.model, 'linker_weights'):
                self.offline_param_group.append({'params': self.model.linker_weights, "lr": 1e-1}) 
        else:
            if hasattr(self.model, 'lidar_key_aligner'):    
                self.online_param_group = [
                    {'params': self.model.lidar_key_aligner.parameters()},
                    {'params': self.model.lidar_value_aligner.parameters()},
                ]
            if hasattr(self.model, 'linker_weights'):
                self.online_param_group.append({'params': self.model.linker_weights, "lr": 1e-1}) 

    def train(
        self,
        learning_rate,
        num_epochs,
        run_validation=True,
        save_model=True,
    ):  
        
        if self.status == 'offline':
            param_group = self.offline_param_group
        elif self.status == 'online':
            self._load_model()
            param_group = self.online_param_group
        else:
            raise ValueError(f"Invalid status {self.status}!")
        
        self.model = self.model.to(self.device)
        
        # self._runtime_evaluate(self.val_loader)
        if hasattr(self.model, 'linker_weights'):
            print(f"Links: {self.model.linker_weights}")
        # feed partitions one by one
        for partition_idx, partition_loader in enumerate(self.train_loader):
            print(f"\n----------------- PARTITION {partition_idx} -----------------\n")
            
            # self.model.parameters()
            optimizer = torch.optim.AdamW(param_group, lr=learning_rate)
            # optimizer = Adafactor(self.param_group, scale_parameter=False, relative_step=False, lr=1e-3)
            # optimizer = bnb.optim.PagedAdamW8bit(self.param_group, lr=learning_rate)
            lr_scheduler = get_linear_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=0,
                num_training_steps=(len(partition_loader) * num_epochs),
            )
        
            # self._runtime_evaluate(self.val_loader)
            total_time = 0
            for epoch in range(num_epochs):
                t_start = time.time()
                
                self.model.train()
                total_loss = 0
                for step, batch in enumerate(tqdm(partition_loader)):
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    with torch.autocast(device_type="cuda"):
                        outputs = self.model(
                            cam_f=batch["cam_f"],
                            cam_fl=batch["cam_fl"],
                            cam_fr=batch["cam_fr"],
                            cam_b=batch["cam_b"],
                            cam_bl=batch["cam_bl"],
                            cam_br=batch["cam_br"],
                            lidar_top=batch["lidar_top"],
                            text_input_ids=batch["text_input_ids"],
                            text_attention_mask=batch["text_attention_mask"],
                            text_labels=batch["text_labels"],    
                        )
                    loss = outputs.loss
                    total_loss += loss.detach().float()
                    loss.backward()
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                
                t_end = time.time()
                epoch_time = t_end - t_start
                print(f"Epoch Time: {epoch_time} (s)")
                total_time += epoch_time
                print(f"Total Time: {total_time} (s)")
                
                if run_validation:
                    self.model.eval()
                    eval_loss = 0
                    for step, batch in enumerate(tqdm(self.val_loader)):
                        batch = {k: v.to(self.device) for k, v in batch.items()}
                        with torch.no_grad():
                            with torch.autocast(device_type="cuda"):
                                outputs = self.model(
                                    cam_f=batch["cam_f"],
                                    cam_fl=batch["cam_fl"],
                                    cam_fr=batch["cam_fr"],
                                    cam_b=batch["cam_b"],
                                    cam_bl=batch["cam_bl"],
                                    cam_br=batch["cam_br"],
                                    lidar_top=batch["lidar_top"],
                                    text_input_ids=batch["text_input_ids"],
                                    text_attention_mask=batch["text_attention_mask"],
                                    text_labels=batch["text_labels"],
                                )
                        loss = outputs.loss
                        eval_loss += loss.detach().float()
                    
                    eval_epoch_loss = eval_loss / len(self.val_loader)
                    eval_ppl = torch.exp(eval_epoch_loss)
                    train_epoch_loss = total_loss / len(partition_loader)
                    train_ppl = torch.exp(train_epoch_loss)
                    
                    self.model.print_learned_linker_weights()

                    print(f"epoch={epoch} train_ppl={train_ppl.item()} train_loss={train_epoch_loss.item()} eval_ppl={eval_ppl.item()} eval_loss={eval_epoch_loss.item()}")
                    
                    self._runtime_evaluate(self.val_loader)
                
            print(f"Total Time: {total_time} (s)")
            self._runtime_evaluate(self.val_loader)
            if hasattr(self.model, 'linker_weights'):
                print(f"Links: {self.model.linker_weights}")

        if save_model:
            self._save_model()
    
    def _runtime_evaluate(self, dataset):
        self.model.eval()
        m_f1 = 0
        m_em = 0
        
        self.tokenizer.padding_side = "left"
        total_count = 0
        with torch.no_grad():
            for step, batch in enumerate(tqdm(dataset)):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                batch_size = batch['text_lp_sources'].shape[0]
                
                with torch.autocast(device_type="cuda"):
                    outputs_tokens = self.model.generate(
                        cam_f=batch["cam_f"],
                        cam_fl=batch["cam_fl"],
                        cam_fr=batch["cam_fr"],
                        cam_b=batch["cam_b"],
                        cam_bl=batch["cam_bl"],
                        cam_br=batch["cam_br"],
                        lidar_top=batch["lidar_top"],
                        text_input_ids=batch["text_lp_sources"],
                        text_attention_mask=batch["text_lp_sources_attention_mask"],
                        max_length=self.max_output_length,
                    )
                for label in batch["text_labels"]:
                    label[label < 0] = self.tokenizer.pad_token_id
                
                outputs_text = [self.tokenizer.decode(y[len(x):], skip_special_tokens=True).strip() for y, x in zip(outputs_tokens, batch['text_lp_sources'])]
                labels_text =  [self.tokenizer.decode(x[offset:], skip_special_tokens=True).strip() for x, offset in zip(batch["text_input_ids"], batch["text_input_ids_lens"])] 
                
                # print(f"pred: {outputs_text[0]} | label: {labels_text[0]}")

                batch_em = self.exact_match.compute(predictions=outputs_text, references=labels_text)
                batch_f1 = compute_f1_score(outputs_text, labels_text)
                
                m_f1 += (batch_f1 * batch_size)
                m_em += (batch_em["exact_match"] * batch_size)
                
                total_count += batch_size
        
        m_f1 /= total_count
        m_em /= total_count
        print(f"On validation/test set, f1={100*m_f1} %, exact_match={100*m_em}")

    def evaluate(self):
        # self._load_model()
        # self.model = self.model.to(self.device)
        
        self.model.eval()
        m_f1 = 0
        m_em = 0
        
        self.tokenizer.padding_side = "left"
        total_count = 0
        with torch.no_grad():
            for step, batch in enumerate(tqdm(self.test_loader)):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                batch_size = batch['text_lp_sources'].shape[0]
                
                with torch.autocast(device_type="cuda"):
                    outputs_tokens = self.model.generate(
                        cam_f=batch["cam_f"],
                        cam_fl=batch["cam_fl"],
                        cam_fr=batch["cam_fr"],
                        cam_b=batch["cam_b"],
                        cam_bl=batch["cam_bl"],
                        cam_br=batch["cam_br"],
                        lidar_top=batch["lidar_top"],
                        text_input_ids=batch["text_lp_sources"],
                        text_attention_mask=batch["text_lp_sources_attention_mask"],
                        max_length=self.max_output_length,
                    )
                for label in batch["text_labels"]:
                    label[label < 0] = self.tokenizer.pad_token_id
                
                outputs_text = [self.tokenizer.decode(y[len(x):], skip_special_tokens=True) for y, x in zip(outputs_tokens, batch['text_lp_sources'])]
                labels_text =  [self.tokenizer.decode(x[offset:], skip_special_tokens=True) for x, offset in zip(batch["text_input_ids"], batch["text_input_ids_lens"])] 

                batch_em = self.exact_match.compute(predictions=outputs_text, references=labels_text)
                batch_f1 = compute_f1_score(outputs_text, labels_text)
                
                m_f1 += (batch_f1 * batch_size)
                m_em += (batch_em["exact_match"] * batch_size)
                
                total_count += batch_size
        
        m_f1 /= total_count
        m_em /= total_count
        print(f"On validation/test set, f1={100*m_f1} %, exact_match={100*m_em}")
    
    def _save_model(self):
        submodules_to_save = {
            'rgb_key_aligner': self.model.image_token_key_aligner,
            'rgb_value_aligner': self.model.image_token_value_aligner,
        }
        state_dict_to_save = {key: value.state_dict() for key, value in submodules_to_save.items()}
        state_dict_to_save['linker_weights'] = self.model.linker_weights
        torch.save(state_dict_to_save, self.save_model_path)
    
    def _load_model(self):
        self.model.load_state_dict(torch.load(self.load_model_path), strict=False)

