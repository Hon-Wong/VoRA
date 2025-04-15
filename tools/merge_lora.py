from models.modeling_vora import VoRAForCausalLM, VoRAConfig
import os
import torch

def key_mapping(state_dict, key_mapping_dict):
    new_state_dict = dict()
    for k, v in state_dict.items():
        flag = 0
        for prev_key in key_mapping_dict.keys():
            if prev_key in k:
                new_state_dict[k.replace(prev_key, key_mapping_dict[prev_key])] = v
                flag = 1
                break
        if flag == 0:
            new_state_dict[k] = v
    return new_state_dict

def merge_lora(checkpoint, lora_key="lora_A"):
    new_state_dict = {}
    lora_processed = set()

    for key in list(checkpoint.keys()):
        if lora_key in key:
            try:
                idx = key.index(lora_key)
            except ValueError:
                continue
            root_key = key[:idx]
            suffix = key[idx + len(lora_key):]
            
            if not suffix.startswith('.'):
                continue

            weight_key = f"{root_key}weight"
            lora_A_key = f"{root_key}lora_A.weight"
            lora_B_key = f"{root_key}lora_B.weight"
            bias_key = f"{root_key}bias"  # 新增：显式处理 bias
            
            if weight_key in lora_processed:
                continue
            lora_processed.update({weight_key, lora_A_key, lora_B_key})
            
            if any(k not in checkpoint for k in [weight_key, lora_A_key, lora_B_key]):
                raise KeyError(f"Missing keys for module {root_key}")
            
            W = checkpoint[weight_key]
            A = checkpoint[lora_A_key]
            B = checkpoint[lora_B_key]
            new_state_dict[weight_key] = W + B @ A
            
            if bias_key in checkpoint:
                new_state_dict[bias_key] = checkpoint[bias_key]
                lora_processed.add(bias_key) 

    for key, value in checkpoint.items():
        if key not in lora_processed:
            new_state_dict[key] = value

    return new_state_dict     

def partial_load_from_checkpoints(
        local_checkpoint_path, 
        ckpt_rename_parameters=None, 
        map_location="cpu",
        model=None,
        valid_prefix=None,
        lazy_load=False
    ):
    
    ckpt_rename_parameters = ckpt_rename_parameters or dict()
    if os.path.isdir(local_checkpoint_path):
        from safetensors.torch import load
        import multiprocessing
        checkpoint = {}
        files = [file for file in os.listdir(local_checkpoint_path) if file.endswith(".safetensors")]
        if len(files) == 0:
            raise ValueError(f"No safetensors file found in {local_checkpoint_path}")
        file_paths = []
        for file in files:
            file_path = os.path.join(local_checkpoint_path, file)
            if not lazy_load:
                print(f"loading checkpoint from {file_path}")
                with open(file_path, "rb") as f:
                    data = f.read()
                loaded = load(data)
                checkpoint.update(loaded)
            else:
                file_paths.append(file_path)
        if lazy_load:
            return file_paths
    else:
        checkpoint = torch.load(local_checkpoint_path, map_location=map_location)

    if "state_dict" in checkpoint:
        logger.info("partial loading checkpoint")
        state_dict = checkpoint["state_dict"]
    elif "module" in checkpoint:
        # for ds zero2 checkpoint
        logger.info("partial loading deepspeed zero2 checkpoint")
        state_dict = checkpoint["module"]
        ckpt_rename_parameters.update({"module.": ""})
    else:
        state_dict = checkpoint

    if valid_prefix:
        new_state_dict = dict()
        for k, v in state_dict.items():
            for prefix in valid_prefix:
                if k.startswith(prefix):
                    new_state_dict[k] = v
        state_dict = new_state_dict
    state_dict = key_mapping(state_dict, ckpt_rename_parameters)
    return state_dict

if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    args = parser.parse_args()

    config_path = args.config
    checkpoint_path = args.checkpoint
    save_path = args.save_dir

    with open(config_path, "r") as f:
        vora_config = yaml.safe_load(f)["model"]
    vora_config["lora"]["r"] = -1
    config = VoRAConfig(**vora_config)

    model = VoRAForCausalLM._from_config(config=config)

    state_dict = partial_load_from_checkpoints(checkpoint_path)
    state_dict = merge_lora(state_dict)
    model.load_state_dict(state_dict, strict=False)
    model.save_pretrained(save_path)
