import torch
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens

def get_num_transfer_tokens_sch(mask_index, steps,schedule=None,schedule_kwargs=None):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    if schedule is None:
        return get_num_transfer_tokens(mask_index,steps)
    if schedule_kwargs is None:
        schedule_kwargs = {}
   
    mask_num = mask_index.sum(dim=1, keepdim=True)
    steps = int(min(steps,mask_num[0]))
    t = torch.linspace(0, 1, steps+1)
    # at least one sample per step
    if schedule =='logit_normal':
      sigmas = sigmoid_normal_cdf(t)
    elif schedule =='shift':
      sigmas = logit_normal_schedule(schedule_kwargs.get('shift',3),t)
    elif schedule == 'cosine':
        sigmas = cosine_schedule(t)
    else:
      sigmas = t
    sigmas = sigmas.to(mask_num.device)
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64)
    
    for i in range(mask_num.size(0)):
      sigmas_sample = (sigmas*mask_num[i]).to(torch.int64)
      sigmas_sample = sigmas_sample[1:]-sigmas_sample[:-1]

      sigmas_sample = torch.clamp(sigmas_sample,1,None) # should only increase
      delta = sigmas_sample.sum() - mask_num[i]
      assert delta>=0
      j = 0
      
      while delta > 0:
        j = j % len(sigmas_sample) 
        if sigmas_sample[j] == 1:
          j += 1
          continue
        
        delta -= 1
        sigmas_sample[j] -= 1
        j += 1
    #   breakpoint()
      assert sigmas_sample.sum()==mask_num[i]
      num_transfer_tokens[i] = sigmas_sample#.to(torch.int64)
    return num_transfer_tokens.flip(-1)

def linear(y):
    return y

def cosine_schedule(x):
    """
    Cosine schedule mapping [0, 1] -> [1, 0]
    """
    x = np.clip(x, 0, 1)
    return 1-0.5 * (1 + np.cos(np.pi * x))

def sigmoid_normal_cdf(y):
    # y must be in (0, 1)
    logit_y = torch.log(y / (1 - y))
    return 0.5 * (1 + torch.erf(logit_y / torch.sqrt(torch.tensor(2.0))))
def logit_normal_schedule(shift,sigmas):
    sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
    return sigmas
import os
DEBUG_PRINT_OUTPUT = os.environ.get('DEBUG_PRINT_OUTPUT',False)
@ torch.no_grad()
def generate(model, prompt=None, steps=None, max_new_tokens=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336,inputs_embeds=None, position_ids=None,attention_mask=None,
              tokenizer=None,
                verbose=False,
                step_per_block=None,
                prefix_lm=False,
                schedule=None,
                schedule_kwargs=None,
                draft_tokens=None,
                step_ratio=None,
                merged=False,
                visual_token_mask=None,
             **kwargs):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''

    steps = max_new_tokens # min(steps,max_new_tokens)

    gen_length = max_new_tokens
    assert position_ids is None
    if prompt is None:
        assert inputs_embeds is not None
        bsz, seq_len = inputs_embeds.shape[:2]
        prompt = torch.full((bsz, seq_len), 0, dtype=torch.long).to(model.device)
    past_key_values = None
    if prefix_lm:
        past_key_values = model(None,input_embeddings=inputs_embeds,use_cache=True).attn_key_values
        # breakpoint()
        x = torch.full((bsz, gen_length), mask_id, dtype=torch.long).to(model.device)
        prompt = torch.full((bsz, 0), 0, dtype=torch.long).to(model.device)

    else:
        x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
        x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)
    
    if visual_token_mask is not None:
        original_len = visual_token_mask.shape[1]
        target_len = x.shape[1]

        if original_len < target_len:
            padded_mask = torch.zeros(
                visual_token_mask.shape[0], 
                target_len, 
                dtype=torch.bool, 
                device=visual_token_mask.device
            )
            padded_mask[:, :original_len] = visual_token_mask
            visual_token_mask = padded_mask
    
    if draft_tokens is not None:
        assert draft_tokens.shape[1] <= gen_length
        x[:, prompt.shape[1]:prompt.shape[1]+draft_tokens.shape[1]] = draft_tokens.clone()

    
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert ( steps % num_blocks == 0) or step_per_block is not None
    steps = steps // num_blocks
    if step_per_block:
        steps = min(step_per_block,block_length)
        assert step_ratio is None, 'Please do not pass both step_ratio and step_per_block'

    
    if step_ratio:
        steps = int(steps*step_ratio)


    
    if verbose:
        history = []
        
    decider_token_indices = None
    
    for num_block in range(num_blocks):

        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens_sch(block_mask_index, steps,schedule=schedule,schedule_kwargs=schedule_kwargs)
        if DEBUG_PRINT_OUTPUT:
            print(f"Block: {num_block + 1}/{num_blocks}, Steps per Block: {steps}, Block Length: {block_length}")
            print(f"Tokens generated per step {num_transfer_tokens[0]}")
        for i in range(steps):

            mask_index = (x == mask_id)
            block_mask_index = mask_index[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:]

            if block_mask_index.sum() == 0:
                continue
            # NFE += 2
            if cfg_scale > 0.:
                assert NotImplementedError('cfg_scale > 0. is not supported.')
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                #
                logits = model(x_,input_embeds_inference=[inputs_embeds,None]).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                inputs_embeds_curr = model.transformer.wte(x)

                
                if prefix_lm:
                    logits = model(None,input_embeddings=inputs_embeds_curr,past_key_values=past_key_values,visual_token_mask=visual_token_mask,decider_token_indices=decider_token_indices).logits
                else:
                    if inputs_embeds is not None:
                        inputs_embeds_curr[:,:inputs_embeds.shape[1]] = inputs_embeds
                    logits = model(None,input_embeddings=inputs_embeds_curr,output_attentions=merged,visual_token_mask=visual_token_mask,
                        decider_token_indices=decider_token_indices,timestep=i, max_timestep=steps).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
            if remasking == 'low_confidence':
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            elif remasking == 'entrophy':
                epsilon = 1e-10
                probs = F.softmax(logits.to(torch.float64), dim=-1)
                log_probs = torch.log(probs + epsilon)
                x0_p = torch.sum(probs * log_probs, dim=-1)
            elif remasking == 'margin':

                p = F.softmax(logits.to(torch.float64), dim=-1)
                sorted_probs, _ = torch.sort(p, dim=-1, descending=True)
                top1_probs = sorted_probs[:, :, 0] 
                top2_probs = sorted_probs[:, :, 1] 
                x0_p = top1_probs - top2_probs 
            else:
                raise NotImplementedError(remasking)
            
            short_len = x0.shape[1]
            full_len = x.shape[1]

            if short_len < full_len:
                padding_needed = full_len - short_len
                pad_tensor_x0 = torch.zeros(
                    x0.shape[0], 
                    padding_needed, 
                    dtype=x0.dtype, 
                    device=x0.device
                )
                x0 = torch.cat([pad_tensor_x0, x0], dim=1)
                pad_tensor_x0_p = torch.zeros(
                    x0_p.shape[0], 
                    padding_needed, 
                    dtype=x0_p.dtype, 
                    device=x0_p.device
                )
                x0_p = torch.cat([pad_tensor_x0_p, x0_p], dim=1)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            target_device = x.device
            x0 = torch.where(mask_index.to(target_device), x0.to(target_device), x) 
            confidence = torch.where(mask_index.to(target_device), x0_p.to(target_device), -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                try:
                    _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                except:
                    breakpoint()
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]
            decider_token_indices = transfer_index.clone()

            if verbose:
                history.append(x.clone().cpu())

    if verbose:
        return x,history
    return x


def main():
    device = 'cuda'

    model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    m = [{"role": "user", "content": prompt}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    out = generate(model, input_ids, steps=128, gen_length=128, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence')
    print(tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0])
    generate(model, input_ids, steps=128, gen_length=128, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence')
   

if __name__ == '__main__':
    main()