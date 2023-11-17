import torch
import torch.nn as nn
from model.utils import freeze_model


class MLM_RGB(torch.nn.Module):
    """Trainable latent connection"""
    
    def __init__(
        self, rgb_encoder=None, lidar_encoder=None, llm_backbone=None, 
        max_input_length=72, backward_depth=4, linker_temperature=1.0,
        freeze_encoder=True, freeze_llm=True,
    ):
        super().__init__()
        self.rgb_encoder = rgb_encoder
        self.rgb_num_hidden_layers = self.rgb_encoder.config.num_hidden_layers
        self.rgb_embed_dim = self.rgb_encoder.config.hidden_size
        self.rgb_ffn_dim = self.rgb_encoder.config.intermediate_size
        
        self.llm_backbone = llm_backbone
        self.llm_num_hidden_layers = self.llm_backbone.config.n_layer
        self.llm_embed_dim = self.llm_backbone.config.hidden_size
        self.llm_num_heads = self.llm_backbone.config.num_attention_heads
        self.llm_head_dim = self.llm_embed_dim // self.llm_num_heads
        self.max_input_length = max_input_length
        self.num_cls_tokens = 4
        
        self.rgb_key_aligner = nn.Sequential(
            nn.Linear(self.rgb_embed_dim, self.rgb_ffn_dim, bias=True),
            nn.GELU(),
            nn.Linear(self.rgb_ffn_dim, self.llm_embed_dim, bias=True),
        )
        self.rgb_value_aligner = nn.Sequential(
            nn.Linear(self.rgb_embed_dim, self.rgb_ffn_dim, bias=True),
            nn.GELU(),
            nn.Linear(self.rgb_ffn_dim, self.llm_embed_dim, bias=True),
        )

        self.backward_depth = backward_depth
        print(f"Backward depth of LLM: {self.backward_depth} blocks, {self.llm_num_hidden_layers} blocks in total.")
        self.linker_weights = nn.Parameter(torch.empty(self.backward_depth))
        nn.init.uniform_(self.linker_weights, a=-1.0, b=1.0)
        
        self.linker_temperature = linker_temperature # TODO: probably can be scheduled from large to small
        print("initial links:", torch.sigmoid(self.linker_weights / self.linker_temperature))
        
        if freeze_encoder:
            freeze_model(rgb_encoder)
        
        if freeze_llm:
            freeze_model(llm_backbone)
    
    def _reshape_to_multihead(self, x, seq_len, batch_size):
        # [batch, num_vit_blocks, token_dim] -> [batch, num_vit_blocks, num_heads, head_dim]
        # -> [batch, num_heads, num_vit_blocks, head_dim]
        return x.view(batch_size, seq_len, self.llm_num_heads, self.llm_head_dim).transpose(1, 2).contiguous()
        
    def forward(
        self, 
        cam_f, cam_fl, cam_fr, cam_b, cam_bl, cam_br,
        text_input_ids,
        text_attention_mask,
        text_labels,
        lidar_top=None,
    ):  
        cam_views = [
            cam_f, cam_fl, cam_fr, cam_b, cam_bl, cam_br,
        ]
        rgb_tokens = []
        for cam_view in cam_views:
            modality_outs = self.rgb_encoder(cam_view)
            modality_outs_hidden_states = modality_outs.hidden_states
            modality_tokens = torch.stack([block[:, 0, :] for block in modality_outs_hidden_states[-self.num_cls_tokens:]], dim=1) # -> [batch, num_vit_blocks, d_token]
            rgb_tokens += [modality_tokens]
        rgb_tokens = torch.cat(rgb_tokens, dim=1) # [batch, num_tokens, d]
        
        batch_size = rgb_tokens.shape[0]      
        rgb_key = self._reshape_to_multihead(
            self.rgb_key_aligner(rgb_tokens), 
            rgb_tokens.shape[1], batch_size,
        ) # -> [batch, num_heads, num_vit_blocks, head_dim]
        rgb_value = self._reshape_to_multihead(
            self.rgb_value_aligner(rgb_tokens), 
            rgb_tokens.shape[1], batch_size,
        ) # -> [batch, num_heads, num_vit_blocks, head_dim]
        
        mm_key, mm_value = rgb_key, rgb_value
        
        # BLOOM's key-value dimensions are so strange!!!
        # -> [batch*num_heads, head_dim, num_vit_blocks]
        mm_key = mm_key.view(-1, mm_key.shape[2], self.llm_head_dim).transpose(1, 2).contiguous()
        # -> [batch*num_heads, num_vit_blocks, head_dim]
        mm_value = mm_value.view(-1, mm_value.shape[2], self.llm_head_dim)
        
        # Multiply by learnable weights with discrete forward and continuous backward
        continuous_weights = torch.sigmoid(self.linker_weights / self.linker_temperature)
        
        mm_key_values_trainable = [
            (mm_key * continuous_weights[i], mm_value * continuous_weights[i])
            for i in range(self.backward_depth)
        ]
        mm_key_values_untrainable = [
            (torch.zeros_like(mm_key), torch.zeros_like(mm_value))
            for i in range(self.llm_num_hidden_layers - self.backward_depth)
        ]
        mm_key_values = mm_key_values_untrainable + mm_key_values_trainable

        # Link rgb tokens to LLM
        outputs = self.llm_backbone(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask,
            labels=text_labels,
            past_key_values=mm_key_values,
        )
        return outputs
    
    @torch.no_grad()
    def generate(
        self, cam_f, cam_fl, cam_fr, cam_b, cam_bl, cam_br,
        text_input_ids,
        text_attention_mask,
        max_length,
        num_beams=4,
        lidar_top=None,
    ):
        cam_views = [
            cam_f, cam_fl, cam_fr, cam_b, cam_bl, cam_br,
        ]
        rgb_tokens = []
        for cam_view in cam_views:
            modality_outs = self.rgb_encoder(cam_view)
            modality_outs_hidden_states = modality_outs.hidden_states
            modality_tokens = torch.stack([block[:, 0, :] for block in modality_outs_hidden_states[-self.num_cls_tokens:]], dim=1) # -> [batch, num_vit_blocks, d_token]
            rgb_tokens += [modality_tokens]
        rgb_tokens = torch.cat(rgb_tokens, dim=1) # [batch, num_tokens, d]
        
        batch_size = rgb_tokens.shape[0]      
        rgb_key = self._reshape_to_multihead(
            self.rgb_key_aligner(rgb_tokens), 
            rgb_tokens.shape[1], batch_size,
        ) # -> [batch, num_heads, num_vit_blocks, head_dim]
        rgb_value = self._reshape_to_multihead(
            self.rgb_value_aligner(rgb_tokens), 
            rgb_tokens.shape[1], batch_size,
        ) # -> [batch, num_heads, num_vit_blocks, head_dim]
        
        mm_key, mm_value = rgb_key, rgb_value
        
        mm_key = torch.repeat_interleave(mm_key, num_beams, dim=0) # -> [batch*num_beams, num_heads, num_vit_blocks, head_dim]
        mm_value = torch.repeat_interleave(mm_value, num_beams, dim=0) # -> [batch*num_beams, num_heads, num_vit_blocks, head_dim]
        
        # BLOOM's key-value dimensions are so strange!!!
        # -> [batch*num_heads, head_dim, num_vit_blocks]
        mm_key = mm_key.view(-1, mm_key.shape[2], self.llm_head_dim).transpose(1, 2).contiguous()
        # -> [batch*num_heads, num_vit_blocks, head_dim]
        mm_value = mm_value.view(-1, mm_value.shape[2], self.llm_head_dim)

        continous_weights = torch.sigmoid(self.linker_weights / self.linker_temperature)
        
        mm_key_values_trainable = [
            (mm_key * continous_weights[i], mm_value * continous_weights[i])
            for i in range(self.backward_depth)
        ]
        mm_key_values_untrainable = [
            (torch.zeros_like(mm_key), torch.zeros_like(mm_value))
            for i in range(self.llm_num_hidden_layers - self.backward_depth)
        ]
        mm_key_values = mm_key_values_untrainable + mm_key_values_trainable

        # Link rgb tokens to LLM
        outputs = self.llm_backbone.generate(
            inputs=text_input_ids, 
            attention_mask=text_attention_mask,
            past_key_values=mm_key_values,
            max_new_tokens=max_length,
            num_beams=num_beams,
            repetition_penalty=2.5, 
            length_penalty=1.0, 
            early_stopping=True,
        )
        return outputs
    
    @torch.no_grad()
    def print_learned_linker_weights(self):
        continuous_weights = torch.sigmoid(self.linker_weights / self.linker_temperature)
        print(f"\ncontinuous_weights: {continuous_weights}\n")
    
    def print_flops(self):
        # Not 100% precise, ignore some trivial operations 
        L = self.rgb_encoder.config.num_hidden_layers
        p = self.rgb_encoder.config.patch_size
        n = (224 / p)**2
        d = self.rgb_encoder.config.hidden_size
        vit_flops = L * (12 * n * d**2 + 2 * n**2 * d) + 3 * p**2 * d
        
        L = self.llm_backbone.config.n_layer
        n = self.max_input_length + self.num_cls_tokens * 4
        d = self.llm_backbone.config.hidden_size
        v = self.llm_backbone.config.vocab_size
        llm_flops = L * (12 * n * d**2 + 2 * n**2 * d)
        embed_flops = n * v * d
        forward_flops = vit_flops + llm_flops + embed_flops
        backward_flops = llm_flops * self.backward_depth / L + embed_flops
        print(f"\nbackward: {backward_flops}")
        print(f"total: {forward_flops + backward_flops}\n")
        

class MLM_LIDAR(torch.nn.Module):
    """Trainable latent connection"""
    
    def __init__(
        self, rgb_encoder=None, lidar_encoder=None, llm_backbone=None, 
        max_input_length=72, backward_depth=4, linker_temperature=1.0,
        freeze_encoder=True, freeze_llm=True,
    ):
        super().__init__()
        self.lidar_encoder = lidar_encoder # vit-small
        self.lidar_num_hidden_layers = 12
        self.lidar_embed_dim = 384
        self.lidar_ffn_dim = 4*384
        
        self.llm_backbone = llm_backbone
        self.llm_num_hidden_layers = self.llm_backbone.config.n_layer
        self.llm_embed_dim = self.llm_backbone.config.hidden_size
        self.llm_num_heads = self.llm_backbone.config.num_attention_heads
        self.llm_head_dim = self.llm_embed_dim // self.llm_num_heads
        self.max_input_length = max_input_length
        self.num_cls_tokens = 4
        
        self.lidar_key_aligner = nn.Sequential(
            nn.Linear(self.lidar_embed_dim, self.lidar_ffn_dim, bias=True),
            nn.GELU(),
            nn.Linear(self.lidar_ffn_dim, self.llm_embed_dim, bias=True),
        )
        self.lidar_value_aligner = nn.Sequential(
            nn.Linear(self.lidar_embed_dim, self.lidar_ffn_dim, bias=True),
            nn.GELU(),
            nn.Linear(self.lidar_ffn_dim, self.llm_embed_dim, bias=True),
        )

        self.backward_depth = backward_depth
        print(f"Backward depth of LLM: {self.backward_depth} blocks, {self.llm_num_hidden_layers} blocks in total.")
        self.linker_weights = nn.Parameter(torch.empty(self.backward_depth))
        nn.init.uniform_(self.linker_weights, a=-1.0, b=1.0)
        
        self.linker_temperature = linker_temperature # TODO: probably can be scheduled from large to small
        print("initial links:", torch.sigmoid(self.linker_weights / self.linker_temperature))
        
        if freeze_encoder:
            freeze_model(lidar_encoder)
        
        if freeze_llm:
            freeze_model(llm_backbone)
        
        self.print_flops()
    
    def _reshape_to_multihead(self, x, seq_len, batch_size):
        # [batch, num_vit_blocks, token_dim] -> [batch, num_vit_blocks, num_heads, head_dim]
        # -> [batch, num_heads, num_vit_blocks, head_dim]
        return x.view(batch_size, seq_len, self.llm_num_heads, self.llm_head_dim).transpose(1, 2).contiguous()
        
    def forward(
        self, 
        cam_f, cam_fl, cam_fr, cam_b, cam_bl, cam_br,
        text_input_ids,
        text_attention_mask,
        text_labels,
        lidar_top=None,
    ):  
        mm_tokens = self.lidar_encoder(lidar_top) # [batch, 4, 384]
        
        batch_size = mm_tokens.shape[0]      
        lidar_key = self._reshape_to_multihead(
            self.lidar_key_aligner(mm_tokens), 
            mm_tokens.shape[1], batch_size,
        ) # -> [batch, num_heads, num_vit_blocks, head_dim]
        lidar_value = self._reshape_to_multihead(
            self.lidar_value_aligner(mm_tokens), 
            mm_tokens.shape[1], batch_size,
        ) # -> [batch, num_heads, num_vit_blocks, head_dim]
        
        mm_key, mm_value = lidar_key, lidar_value
        
        # BLOOM's key-value dimensions are so strange!!!
        # -> [batch*num_heads, head_dim, num_vit_blocks]
        mm_key = mm_key.view(-1, mm_key.shape[2], self.llm_head_dim).transpose(1, 2).contiguous()
        # -> [batch*num_heads, num_vit_blocks, head_dim]
        mm_value = mm_value.view(-1, mm_value.shape[2], self.llm_head_dim)
        
        # Multiply by learnable weights with discrete forward and continuous backward
        continuous_weights = torch.sigmoid(self.linker_weights / self.linker_temperature)
        
        mm_key_values_trainable = [
            (mm_key * continuous_weights[i], mm_value * continuous_weights[i])
            for i in range(self.backward_depth)
        ]
        mm_key_values_untrainable = [
            (torch.zeros_like(mm_key), torch.zeros_like(mm_value))
            for i in range(self.llm_num_hidden_layers - self.backward_depth)
        ]
        mm_key_values = mm_key_values_untrainable + mm_key_values_trainable

        # Link lidar tokens to LLM
        outputs = self.llm_backbone(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask,
            labels=text_labels,
            past_key_values=mm_key_values,
        )
        return outputs
    
    @torch.no_grad()
    def generate(
        self, cam_f, cam_fl, cam_fr, cam_b, cam_bl, cam_br,
        text_input_ids,
        text_attention_mask,
        max_length,
        num_beams=4,
        lidar_top=None,
    ):
        mm_tokens = self.lidar_encoder(lidar_top) # [batch, 4, 384]
        
        batch_size = mm_tokens.shape[0]      
        lidar_key = self._reshape_to_multihead(
            self.lidar_key_aligner(mm_tokens), 
            mm_tokens.shape[1], batch_size,
        ) # -> [batch, num_heads, num_vit_blocks, head_dim]
        lidar_value = self._reshape_to_multihead(
            self.lidar_value_aligner(mm_tokens), 
            mm_tokens.shape[1], batch_size,
        ) # -> [batch, num_heads, num_vit_blocks, head_dim]
        
        mm_key, mm_value = lidar_key, lidar_value
        
        mm_key = torch.repeat_interleave(mm_key, num_beams, dim=0) # -> [batch*num_beams, num_heads, num_vit_blocks, head_dim]
        mm_value = torch.repeat_interleave(mm_value, num_beams, dim=0) # -> [batch*num_beams, num_heads, num_vit_blocks, head_dim]
        
        # BLOOM's key-value dimensions are so strange!!!
        # -> [batch*num_heads, head_dim, num_vit_blocks]
        mm_key = mm_key.view(-1, mm_key.shape[2], self.llm_head_dim).transpose(1, 2).contiguous()
        # -> [batch*num_heads, num_vit_blocks, head_dim]
        mm_value = mm_value.view(-1, mm_value.shape[2], self.llm_head_dim)

        continous_weights = torch.sigmoid(self.linker_weights / self.linker_temperature)
        
        mm_key_values_trainable = [
            (mm_key * continous_weights[i], mm_value * continous_weights[i])
            for i in range(self.backward_depth)
        ]
        mm_key_values_untrainable = [
            (torch.zeros_like(mm_key), torch.zeros_like(mm_value))
            for i in range(self.llm_num_hidden_layers - self.backward_depth)
        ]
        mm_key_values = mm_key_values_untrainable + mm_key_values_trainable

        # Link lidar tokens to LLM
        outputs = self.llm_backbone.generate(
            inputs=text_input_ids, 
            attention_mask=text_attention_mask,
            past_key_values=mm_key_values,
            max_new_tokens=max_length,
            num_beams=num_beams,
            repetition_penalty=2.5, 
            length_penalty=1.0, 
            early_stopping=True,
        )
        return outputs
    
    @torch.no_grad()
    def print_learned_linker_weights(self):
        continuous_weights = torch.sigmoid(self.linker_weights / self.linker_temperature)
        print(f"\ncontinuous_weights: {continuous_weights}\n")
    
    def print_flops(self):
        # Not 100% precise, ignore some trivial operations 
        L = self.lidar_num_hidden_layers
        p = 16
        n = (2048 / p) * (32 / p)
        d = self.lidar_embed_dim
        vit_flops = L * (12 * n * d**2 + 2 * n**2 * d) + 5 * p**2 * d
        
        L = self.llm_backbone.config.n_layer
        n = self.max_input_length + 4
        d = self.llm_backbone.config.hidden_size
        v = self.llm_backbone.config.vocab_size
        llm_flops = L * (12 * n * d**2 + 2 * n**2 * d)
        embed_flops = n * v * d
        forward_flops = vit_flops + llm_flops + embed_flops
        backward_flops = llm_flops * self.backward_depth / L + embed_flops
        print(f"\nbackward: {backward_flops}")
        print(f"total: {forward_flops + backward_flops}\n")
    
    
class MLM_RGB_LIDAR(torch.nn.Module):
    """Trainable latent connection"""
    
    def __init__(
        self, rgb_encoder=None, lidar_encoder=None, llm_backbone=None, 
        max_input_length=72, backward_depth=4, linker_temperature=1.0,
        freeze_encoder=True, freeze_llm=True,
    ):
        super().__init__()
        self.rgb_encoder = rgb_encoder
        self.rgb_num_hidden_layers = self.rgb_encoder.config.num_hidden_layers
        self.rgb_embed_dim = self.rgb_encoder.config.hidden_size
        self.rgb_ffn_dim = self.rgb_encoder.config.intermediate_size
        
        self.lidar_encoder = lidar_encoder # vit-small
        self.lidar_num_hidden_layers = 12
        self.lidar_embed_dim = 384
        self.lidar_ffn_dim = 4*384
        
        self.llm_backbone = llm_backbone
        self.llm_num_hidden_layers = self.llm_backbone.config.n_layer
        self.llm_embed_dim = self.llm_backbone.config.hidden_size
        self.llm_num_heads = self.llm_backbone.config.num_attention_heads
        self.llm_head_dim = self.llm_embed_dim // self.llm_num_heads
        self.max_input_length = max_input_length
        self.num_cls_tokens = 4
        
        self.rgb_key_aligner = nn.Sequential(
            nn.Linear(self.rgb_embed_dim, self.rgb_ffn_dim, bias=True),
            nn.GELU(),
            nn.Linear(self.rgb_ffn_dim, self.llm_embed_dim, bias=True),
        )
        self.rgb_value_aligner = nn.Sequential(
            nn.Linear(self.rgb_embed_dim, self.rgb_ffn_dim, bias=True),
            nn.GELU(),
            nn.Linear(self.rgb_ffn_dim, self.llm_embed_dim, bias=True),
        )
        
        self.lidar_key_aligner = nn.Sequential(
            nn.Linear(self.lidar_embed_dim, self.lidar_ffn_dim, bias=True),
            nn.GELU(),
            nn.Linear(self.lidar_ffn_dim, self.llm_embed_dim, bias=True),
        )
        self.lidar_value_aligner = nn.Sequential(
            nn.Linear(self.lidar_embed_dim, self.lidar_ffn_dim, bias=True),
            nn.GELU(),
            nn.Linear(self.lidar_ffn_dim, self.llm_embed_dim, bias=True),
        )

        self.backward_depth = backward_depth
        print(f"Backward depth of LLM: {self.backward_depth} blocks, {self.llm_num_hidden_layers} blocks in total.")
        self.linker_weights = nn.Parameter(torch.empty(self.backward_depth))
        nn.init.uniform_(self.linker_weights, a=-1.0, b=1.0)
        
        self.linker_temperature = linker_temperature # TODO: probably can be scheduled from large to small
        print("initial links:", torch.sigmoid(self.linker_weights / self.linker_temperature))
        
        if freeze_encoder:
            freeze_model(rgb_encoder)
            freeze_model(lidar_encoder)
        
        if freeze_llm:
            freeze_model(llm_backbone)
    
    def _reshape_to_multihead(self, x, seq_len, batch_size):
        # [batch, num_vit_blocks, token_dim] -> [batch, num_vit_blocks, num_heads, head_dim]
        # -> [batch, num_heads, num_vit_blocks, head_dim]
        return x.view(batch_size, seq_len, self.llm_num_heads, self.llm_head_dim).transpose(1, 2).contiguous()
        
    def forward(
        self, 
        cam_f, cam_fl, cam_fr, cam_b, cam_bl, cam_br,
        text_input_ids,
        text_attention_mask,
        text_labels,
        lidar_top=None,
    ):  
        # rgb
        cam_views = [
            cam_f, cam_fl, cam_fr, cam_b, cam_bl, cam_br,
        ]
        rgb_tokens = []
        for cam_view in cam_views:
            modality_outs = self.rgb_encoder(cam_view)
            modality_outs_hidden_states = modality_outs.hidden_states
            modality_tokens = torch.stack([block[:, 0, :] for block in modality_outs_hidden_states[-self.num_cls_tokens:]], dim=1) # -> [batch, num_vit_blocks, d_token]
            rgb_tokens += [modality_tokens]
        rgb_tokens = torch.cat(rgb_tokens, dim=1) # [batch, num_tokens, d]
        
        batch_size = rgb_tokens.shape[0]      
        rgb_key = self._reshape_to_multihead(
            self.rgb_key_aligner(rgb_tokens), 
            rgb_tokens.shape[1], batch_size,
        ) # -> [batch, num_heads, num_vit_blocks, head_dim]
        rgb_value = self._reshape_to_multihead(
            self.rgb_value_aligner(rgb_tokens), 
            rgb_tokens.shape[1], batch_size,
        ) # -> [batch, num_heads, num_vit_blocks, head_dim]
        
        # lidar
        mm_tokens = self.lidar_encoder(lidar_top) # [batch, 4, 384]
        lidar_key = self._reshape_to_multihead(
            self.lidar_key_aligner(mm_tokens), 
            mm_tokens.shape[1], batch_size,
        ) # -> [batch, num_heads, num_vit_blocks, head_dim]
        lidar_value = self._reshape_to_multihead(
            self.lidar_value_aligner(mm_tokens), 
            mm_tokens.shape[1], batch_size,
        ) # -> [batch, num_heads, num_vit_blocks, head_dim]
        
        mm_key = torch.cat([rgb_key, lidar_key], dim=2)
        mm_value = torch.cat([rgb_value, lidar_value], dim=2)
        
        # BLOOM's key-value dimensions are so strange!!!
        # -> [batch*num_heads, head_dim, num_vit_blocks]
        mm_key = mm_key.view(-1, mm_key.shape[2], self.llm_head_dim).transpose(1, 2).contiguous()
        # -> [batch*num_heads, num_vit_blocks, head_dim]
        mm_value = mm_value.view(-1, mm_value.shape[2], self.llm_head_dim)
        
        # Multiply by learnable weights with discrete forward and continuous backward
        continuous_weights = torch.sigmoid(self.linker_weights / self.linker_temperature)
        
        mm_key_values_trainable = [
            (mm_key * continuous_weights[i], mm_value * continuous_weights[i])
            for i in range(self.backward_depth)
        ]
        mm_key_values_untrainable = [
            (torch.zeros_like(mm_key), torch.zeros_like(mm_value))
            for i in range(self.llm_num_hidden_layers - self.backward_depth)
        ]
        mm_key_values = mm_key_values_untrainable + mm_key_values_trainable

        # Link lidar tokens to LLM
        outputs = self.llm_backbone(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask,
            labels=text_labels,
            past_key_values=mm_key_values,
        )
        return outputs
    
    @torch.no_grad()
    def generate(
        self, cam_f, cam_fl, cam_fr, cam_b, cam_bl, cam_br,
        text_input_ids,
        text_attention_mask,
        max_length,
        num_beams=4,
        lidar_top=None,
    ):
        # rgb
        cam_views = [
            cam_f, cam_fl, cam_fr, cam_b, cam_bl, cam_br,
        ]
        rgb_tokens = []
        for cam_view in cam_views:
            modality_outs = self.rgb_encoder(cam_view)
            modality_outs_hidden_states = modality_outs.hidden_states
            modality_tokens = torch.stack([block[:, 0, :] for block in modality_outs_hidden_states[-self.num_cls_tokens:]], dim=1) # -> [batch, num_vit_blocks, d_token]
            rgb_tokens += [modality_tokens]
        rgb_tokens = torch.cat(rgb_tokens, dim=1) # [batch, num_tokens, d]
        
        batch_size = rgb_tokens.shape[0]      
        rgb_key = self._reshape_to_multihead(
            self.rgb_key_aligner(rgb_tokens), 
            rgb_tokens.shape[1], batch_size,
        ) # -> [batch, num_heads, num_vit_blocks, head_dim]
        rgb_value = self._reshape_to_multihead(
            self.rgb_value_aligner(rgb_tokens), 
            rgb_tokens.shape[1], batch_size,
        ) # -> [batch, num_heads, num_vit_blocks, head_dim]
        
        # lidar
        mm_tokens = self.lidar_encoder(lidar_top) # [batch, 4, 384]
        lidar_key = self._reshape_to_multihead(
            self.lidar_key_aligner(mm_tokens), 
            mm_tokens.shape[1], batch_size,
        ) # -> [batch, num_heads, num_vit_blocks, head_dim]
        lidar_value = self._reshape_to_multihead(
            self.lidar_value_aligner(mm_tokens), 
            mm_tokens.shape[1], batch_size,
        ) # -> [batch, num_heads, num_vit_blocks, head_dim]
        
        mm_key = torch.cat([rgb_key, lidar_key], dim=2)
        mm_value = torch.cat([rgb_value, lidar_value], dim=2)
        
        mm_key = torch.repeat_interleave(mm_key, num_beams, dim=0) # -> [batch*num_beams, num_heads, num_vit_blocks, head_dim]
        mm_value = torch.repeat_interleave(mm_value, num_beams, dim=0) # -> [batch*num_beams, num_heads, num_vit_blocks, head_dim]
        
        # BLOOM's key-value dimensions are so strange!!!
        # -> [batch*num_heads, head_dim, num_vit_blocks]
        mm_key = mm_key.view(-1, mm_key.shape[2], self.llm_head_dim).transpose(1, 2).contiguous()
        # -> [batch*num_heads, num_vit_blocks, head_dim]
        mm_value = mm_value.view(-1, mm_value.shape[2], self.llm_head_dim)

        continous_weights = torch.sigmoid(self.linker_weights / self.linker_temperature)
        
        mm_key_values_trainable = [
            (mm_key * continous_weights[i], mm_value * continous_weights[i])
            for i in range(self.backward_depth)
        ]
        mm_key_values_untrainable = [
            (torch.zeros_like(mm_key), torch.zeros_like(mm_value))
            for i in range(self.llm_num_hidden_layers - self.backward_depth)
        ]
        mm_key_values = mm_key_values_untrainable + mm_key_values_trainable

        # Link lidar tokens to LLM
        outputs = self.llm_backbone.generate(
            inputs=text_input_ids, 
            attention_mask=text_attention_mask,
            past_key_values=mm_key_values,
            max_new_tokens=max_length,
            num_beams=num_beams,
            repetition_penalty=2.5, 
            length_penalty=1.0, 
            early_stopping=True,
        )
        return outputs
    
    @torch.no_grad()
    def print_learned_linker_weights(self):
        continuous_weights = torch.sigmoid(self.linker_weights / self.linker_temperature)
        print(f"\ncontinuous_weights: {continuous_weights}\n")
    
    def print_flops(self):
        # Not 100% precise, ignore some trivial operations 
        L = self.lidar_num_hidden_layers
        p = 16
        n = (2048 / p) * (32 / p)
        d = self.lidar_embed_dim
        rangevit_flops = L * (12 * n * d**2 + 2 * n**2 * d) + 5 * p**2 * d
        
        L = self.rgb_encoder.config.num_hidden_layers
        p = self.rgb_encoder.config.patch_size
        n = (224 / p)**2
        d = self.rgb_encoder.config.hidden_size
        vit_flops = L * (12 * n * d**2 + 2 * n**2 * d) + 3 * p**2 * d
        
        L = self.llm_backbone.config.n_layer
        n = self.max_input_length + self.num_cls_tokens * 4 + 4
        d = self.llm_backbone.config.hidden_size
        v = self.llm_backbone.config.vocab_size
        llm_flops = L * (12 * n * d**2 + 2 * n**2 * d)
        embed_flops = n * v * d
        forward_flops = rangevit_flops + vit_flops + llm_flops + embed_flops
        backward_flops = llm_flops * self.backward_depth / L + embed_flops
        print(f"\nbackward: {backward_flops}")
        print(f"total: {forward_flops + backward_flops}\n")