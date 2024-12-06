import torch
import torch.distributed as dist
from .modeling_llama_ring import LlamaForCausalLM
from transformers import AutoTokenizer

def load_ring_model_tokenizer(path: str):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    model = LlamaForCausalLM.from_pretrained(path, attn_implementation='flash_attention_2', torch_dtype=torch.bfloat16, device_map=torch.device(f"cuda:{rank}"))
    model.set_dist(rank, world_size)
    tokenizer = AutoTokenizer.from_pretrained(path)
    return model, tokenizer

@torch.no_grad()
def ring_prefill_and_decode(model, input_ids, position_ids, max_new_tokens = 12, stop_ids = [], debug=False):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    generated_ids = []
    # 1. prefill by ring attention
    output = model(input_ids, position_ids=position_ids, use_cache=True, output_hidden_states=debug)
    hiddens = []
    if debug and rank == world_size - 1:
        hiddens.append(output.hidden_states[-1][:, -1, :])
    if rank == world_size - 1: # generate next_id on the last gpu
        # if output.logits.shape[1] == 0:
        #     breakpoint()
        next_id = output.logits[:, -1, :].argmax(dim=-1, keepdim=True) # （1, 1）
    else: # wait and receive next_id for other gpus
        next_id = torch.zeros((1, 1), dtype=torch.long, device=torch.device(f"cuda:{rank}"))
    # dist.barrier()
    dist.broadcast(next_id, src=world_size-1)
    generated_ids.append(next_id.item())
    # print(next_id)
    position_ids = position_ids[:, -1:] + 1
    past_key_values = output.past_key_values
    for _ in range(max_new_tokens - 1):
        output = model(next_id, position_ids=position_ids, past_key_values=past_key_values, use_cache=True, output_hidden_states=debug)
        if debug and rank == world_size - 1:
            hiddens.append(output.hidden_states[-1][:, -1, :])
        position_ids = position_ids[:, -1:] + 1
        if rank == world_size - 1:
            next_id = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            past_key_values = output.past_key_values # only update the kv cache on the last gpu
        else:
            next_id = torch.zeros((1, 1), dtype=torch.long, device=torch.device(f"cuda:{rank}"))
        dist.broadcast(next_id, src=world_size-1)
        # print(next_id)
        if next_id.item() in stop_ids:
            break
        generated_ids.append(next_id.item())
    if debug:
        return generated_ids, hiddens
    else:
        return generated_ids
        
        
    
    

def _gather_hidden_states(output):
    hs = output.hidden_states
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    gathered_hs = []

    for i, hsl in enumerate(hs):
        g_hs = [torch.empty_like(hsl) for _ in range(world_size)] if rank == 0 else None
        
        dist.barrier()
        dist.gather(hsl, gather_list=g_hs, dst=0)
        dist.barrier()

        if rank == 0:
            gathered_layer_states = torch.cat(g_hs, dim=1) 
            gathered_hs.append(gathered_layer_states)
        else:
            gathered_hs.append(None)

    return gathered_hs