from datasets import load_from_disk, Dataset
from transformers import default_data_collator
from torch.utils.data import DataLoader
import torch
import numpy as np
import copy
import os
import multiprocessing
import time
from multiprocessing import Process


def load_nuqa(
    data_path,
    llm_name,
    rgb_processor,
    text_tokenizer,
    attention_mask_expand,
    max_input_length,
    batch_size,
    extracted_ratio=1.0,
    shuffle=True,
    keep_in_memory=False,
    num_partitions_for_streaming=1,
):
    """Load a split of nuscenes_qa_mini dataset
    train: 3,302 validation: 2,474"""
    
    # for f in os.listdir("../../" + data_path):
    #     if "cache" in f:
    #         print(f"Found cached preprocessed dataset: {f}")
    #         processed_dataset = Dataset.from_file("../../" + data_path + "/" + f)

    #         # processed_dataset.shuffle(seed=123)
    #         extracted_count = int(processed_dataset.num_rows * extracted_ratio)
    #         processed_dataset = processed_dataset.select(range(extracted_count))

    #         processed_dataset.set_format(
    #             type="torch", columns=[
    #                 "cam_f", "cam_fl", "cam_fr", "cam_b", "cam_bl", "cam_br", "lidar_top",
    #                 "text_input_ids", "text_attention_mask", "text_labels", 
    #                 "text_input_ids_lens", "text_lp_sources", "text_lp_sources_attention_mask",
    #             ]
    #         )
    #         dataloader = DataLoader(
    #             processed_dataset, shuffle=shuffle, 
    #             collate_fn=default_data_collator, 
    #             batch_size=batch_size, pin_memory=keep_in_memory,
    #         )
    #         return [dataloader]


    if "night" not in data_path:
        # day scenes, check cache existence
        for f in os.listdir("../../" + data_path):
            if "cache" in f:
                print(f"Found cached preprocessed dataset: {f}")
                processed_dataset = Dataset.from_file("../../" + data_path + "/" + f)
                processed_dataset.set_format(
                    type="torch", columns=[
                        "cam_f", "cam_fl", "cam_fr", "cam_b", "cam_bl", "cam_br", "lidar_top",
                        "text_input_ids", "text_attention_mask", "text_labels", 
                        "text_input_ids_lens", "text_lp_sources", "text_lp_sources_attention_mask",
                    ]
                )
                dataloader = DataLoader(
                    processed_dataset, shuffle=shuffle, 
                    collate_fn=default_data_collator, 
                    batch_size=batch_size, pin_memory=keep_in_memory,
                )
                return [dataloader]
    
    # night scenes or day scenes but cache not available
    dataset = load_from_disk("../../" + data_path)
    if "night" in data_path:
        # night scenes
        dataset.cleanup_cache_files()
    
    print("No cached preprocessed dataset found. Start generating...")

    # dataset.cleanup_cache_files()

    extracted_count = int(dataset.num_rows * extracted_ratio)
    dataset = dataset.select(range(extracted_count))

    partitions = []
    num_samples_per_partition = extracted_count // num_partitions_for_streaming
    for i in range(num_partitions_for_streaming):
        start_idx = i * num_samples_per_partition
        partition = dataset.select(range(start_idx, start_idx + num_samples_per_partition))
        partitions.append(partition)
    
    def _tokenize_fn(strings):
        """Tokenize a list of strings, memorize source length"""
        text_tokenizer.padding_side = "right" 
        tokenized_list = [
            text_tokenizer(
                text,
                max_length=max_input_length,
                padding="max_length", 
                truncation=True,
                return_tensors="pt",
            )
            for text in strings
        ]
        input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
        attention_mask = [tokenized.attention_mask[0] for tokenized in tokenized_list]
        input_ids_lens = labels_lens = [
            mask.sum().item() for mask in attention_mask
        ]
        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            input_ids_lens=input_ids_lens,
            labels_lens=labels_lens,
        )   
    
    def _truncate_context(context, max_length=128):
        context_tokens = text_tokenizer(context)
        truncate_ratio = max_length / (len(context_tokens["input_ids"]) + 1e-5)
        if truncate_ratio < 1:
            return context[:int(len(context) * truncate_ratio)]
        else:
            return context 
    
    def rgb_to_tensor(rgb_array):
        processed_image = rgb_processor(np.array(rgb_array, dtype=np.float32), return_tensors="pt")
        return processed_image["pixel_values"][0]
    
    def lidar_to_tensor(lidar_array):
        return torch.tensor(lidar_array, dtype=torch.float32)

    def preprocess_fn(raw_examples):
        """preprocess example strings, mask source part in labels"""
        def proc_fn(q, m_input, id_str: str):
            my_output = rgb_to_tensor(m_input)
            q.put((my_output, id_str))
        def lidar_fn(q, m_input, id_str: str):
            my_output = lidar_to_tensor(m_input)
            q.put((my_output, id_str))

        cam_f = []
        cam_fl = []
        cam_fr = []
        cam_b = []
        cam_bl = []
        cam_br = []
        lidar_top = []
        sources = []
        examples = []
        m_queue = multiprocessing.Queue()
        for c_f, c_fl, c_fr, c_b, c_bl, c_br, l_top, q, a in zip(
            raw_examples["CAM_FRONT"], raw_examples["CAM_FRONT_LEFT"], raw_examples["CAM_FRONT_RIGHT"],
            raw_examples["CAM_BACK"], raw_examples["CAM_BACK_LEFT"], raw_examples["CAM_BACK_RIGHT"],
            raw_examples["LIDAR_TOP"], raw_examples["question"], raw_examples["answer"],
        ):
            cam_f.append(rgb_to_tensor(c_f))
            cam_fl.append(rgb_to_tensor(c_fl))
            cam_fr.append(rgb_to_tensor(c_fr))
            cam_b.append(rgb_to_tensor(c_b))
            cam_bl.append(rgb_to_tensor(c_bl))
            cam_br.append(rgb_to_tensor(c_br))
            lidar_top.append(lidar_to_tensor(l_top))
            ################################################
            # p1 = Process(target=proc_fn, args=(m_queue, c_f, 'cam_f'))
            # p2 = Process(target=proc_fn, args=(m_queue, c_fl, 'cam_fl'))
            # p3 = Process(target=proc_fn, args=(m_queue, c_fr, 'cam_fr'))
            # p4 = Process(target=proc_fn, args=(m_queue, c_b, 'cam_b'))
            # p5 = Process(target=proc_fn, args=(m_queue, c_bl, 'cam_bl'))
            # p6 = Process(target=proc_fn, args=(m_queue, c_br, 'cam_br'))
            # p7 = Process(target=lidar_fn, args=(m_queue, l_top, 'lidar_top'))
            # p1.start(); p2.start(); p3.start(); p4.start(); p5.start(); p6.start(); p7.start();
            # results = [(m_queue.get(block=True, timeout=120)) for p in [p1, p2, p3, p4, p5, p6, p7]]
            # p1.terminate(); p2.terminate(); p3.terminate(); p4.terminate(); p5.terminate(); p6.terminate(); p7.terminate();
            # for item in results:
            #     output, id_str = item
            #     if id_str == 'cam_f':
            #         cam_f.append(output)
            #     elif id_str == 'cam_fl':
            #         cam_fl.append(output)
            #     elif id_str == 'cam_fr':
            #         cam_fr.append(output)
            #     elif id_str == 'cam_b':
            #         cam_b.append(output)
            #     elif id_str == 'cam_bl':
            #         cam_bl.append(output)
            #     elif id_str == 'cam_br':
            #         cam_br.append(output)
            #     elif id_str == 'lidar_top':
            #         lidar_top.append(output)
            #     else:
            #         raise Exception('Invalid id_str!')
            #############################################

            sources.append(f"question:{_truncate_context(q, 64)}</s>answer:")
            examples.append(f"question:{_truncate_context(q, 64)}</s>answer:{a}</s>")
            
        # left-padded source for validation & testing
        text_tokenizer.padding_side = "left"
        
        if 'gpt2' in llm_name:
            text_tokenizer.pad_token = text_tokenizer.eos_token
            
        lp_sources = text_tokenizer(
            sources,
            max_length=max_input_length,
            padding="max_length", 
            truncation=True,
            return_tensors="pt",
        )
        
        examples_tokenized, sources_tokenized = [_tokenize_fn(strings) for strings in (examples, sources)]

        input_ids = examples_tokenized["input_ids"]
        attention_mask = examples_tokenized["attention_mask"]
        
        attention_mask_ext = []
        for mask in attention_mask:
            mask_ext = torch.cat([torch.ones(attention_mask_expand), mask], dim=0)
            attention_mask_ext.append(mask_ext)

        labels = copy.deepcopy(input_ids)
        for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
            label[:(source_len)] = -100 # TODO: mask source whether to -1 ???
            label[label == text_tokenizer.pad_token_id] = -100 # mask paddings
        return dict(
            cam_f=cam_f,
            cam_fl=cam_fl,
            cam_fr=cam_fr,
            cam_b=cam_b,
            cam_bl=cam_bl,
            cam_br=cam_br,
            lidar_top=lidar_top,
            text_input_ids=input_ids, 
            text_attention_mask=attention_mask_ext, 
            text_labels=labels, 
            text_input_ids_lens=sources_tokenized["input_ids_lens"],
            text_lp_sources=lp_sources["input_ids"],
            text_lp_sources_attention_mask=lp_sources["attention_mask"],
        )
    
    dataloaders = []

    for partition in partitions:
        processed_dataset = partition.map(preprocess_fn, batched=True, batch_size=12)
        processed_dataset.set_format(
            type="torch", columns=[
                "cam_f", "cam_fl", "cam_fr", "cam_b", "cam_bl", "cam_br", "lidar_top",
                "text_input_ids", "text_attention_mask", "text_labels", 
                "text_input_ids_lens", "text_lp_sources", "text_lp_sources_attention_mask",
            ]
        )
        dataloader = DataLoader(
            processed_dataset, shuffle=shuffle, 
            collate_fn=default_data_collator, 
            batch_size=batch_size, pin_memory=keep_in_memory,
        )
        dataloaders.append(dataloader)
    return dataloaders