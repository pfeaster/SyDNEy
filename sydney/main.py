# We'll assume the following basic libraries are installed
import os,datetime,tkinter as tk,copy,shutil
from tkinter import simpledialog as sd, filedialog as fd
from itertools import combinations
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Other dependencies we'll check for and report if missing
missing_dependencies = []
missing_dependencies_audio = []
missing_dependencies_video = []

# For basic functionality
try:
    from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL
except:
    missing_dependencies.append('diffusers')
try:
    from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection
except:
    missing_dependencies.append('transformers')
try:
    import torch
    torch_device = "cuda" if torch.cuda.is_available(
    ) else "mps" if torch.backends.mps.is_available() else "cpu"
    torch.use_deterministic_algorithms(True)
except:
    missing_dependencies.append('torch')
try:
    from torchvision import transforms as tfms
except:
    missing_dependencies.append('torchvision')
try:
    from PIL import Image, ImageTk, ImageOps
except:
    missing_dependencies.append('Pillow (PIL)')

# For audio output
try:
    import numpy as np
except:
    missing_dependencies_audio.append('numpy')
try:
    import soundfile as sf
except:
    missing_dependencies_audio.append('soundfile')
try:
    import audio2numpy as a2n
except:
    missing_dependencies_audio.append('audio2numpy')
try:
    import torchaudio
except:
    missing_dependencies_audio.append('torchaudio')
try:
    from scipy import signal
except:
    missing_dependencies_audio.append('scipy')

# For video output
try:
    import cv2
except:
    missing_dependencies_video.append('OpenCV (cv2)')

def write_config(path_to_config):
    config_lines = ['#models ',
                    'stable-diffusion-v1-5/stable-diffusion-v1-5;',
                    'riffusion/riffusion-model-v1;',
                    'stabilityai/stable-diffusion-2-1;',
                    'stabilityai/stable-diffusion-2-1-base;',
                    'SG161222/Realistic_Vision_V1.4;',
                    'dreamlike-art/dreamlike-diffusion-1.0;',
                    'dreamlike-art/dreamlike-photoreal-2.0;',
                    'Lykon/DreamShaper;',
                    'Envvi/Inkpunk-Diffusion,* nvinkpunk;',
                    'nitrosocke/Nitro-Diffusion,* archer style;',
                    'nitrosocke/Nitro-Diffusion,* arcane style;',
                    'nitrosocke/Nitro-Diffusion,* modern disney style;',
                    'gsdf/Counterfeit-V2.5;',
                    'wavymulder/portraitplus,portrait+ style *;',
                    'proximasanfinetuning/luna-diffusion-v2.5;',
                    'prompthero/openjourney-v4,* mdjrny-v4 style;',
                    'SY573M404/f222-diffusers;',
                    'Lykon/NeverEnding-Dream;',
                    'creaoy/Foto-Assisted-Diffusion;',
                    'Yntec/foto-assisted-diffusion;',
                    'wavymulder/portraitplus;',
                    'hakurei/waifu-diffusion;',
                    'stabilityai/stable-diffusion-xl-base-1.0;',
                    'stabilityai/sdxl-turbo;',
                    'Lykon/dreamshaper-xl-turbo;',
                    'Lykon/absolute-reality-1.81;',
                    'Lykon/dreamshaper-8;',
                    'prompthero/openjourney-v4;',
                    'nitrosocke/Nitro-Diffusion;',
                    'Envvi/Inkpunk-Diffusion;',
                    'proximasanfinetuning/luna-diffusion;',
                    'CompVis/stable-diffusion-v1-1;',
                    'auffusion/auffusion-full-no-adapter;',
                    'SG161222/Realistic_Vision_V1.3;',
                    'Yntec/ThisIsReal;',
                    'Yntec/Lyriel;',
                    'danbrown/Lyriel-v1-5;',
                    'darkstorm2150/Protogen_v2.2_Official_Release']
    with open(path_to_config, 'w') as file:
        for line in config_lines:
            file.write(line+'\n')

# Set up names of reference files, create reference directories, set up some global variables
ref_dir = 'SyDNEy_ref'
if not os.path.exists(ref_dir):
    os.mkdir(ref_dir)
path_to_script = ref_dir+'/SyDNEy_script.txt'
path_to_config = ref_dir+'/SyDNEy_config.txt'
if not os.path.exists(path_to_config):
    write_config(path_to_config)
path_to_resume_point = ref_dir+"/resume_point.txt"
path_to_backups = ref_dir+"/script_backups"
path_to_resume_backups=ref_dir+"/script_backups/resume"
path_to_scheduler_config = ref_dir+"/scheduler_config_"
if not os.path.exists(path_to_backups):
    os.mkdir(path_to_backups)
if not os.path.exists(path_to_resume_backups):
    os.mkdir(path_to_resume_backups)
script_backup_prefix = "/script_backup_"
config_backup_prefix = "/config_backup_"
backup_resume_prefix = "/resume_"
work_dir = 'SyDNEy_work/'
prompt_defaults = [[[0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0]], [[0, 0, 0]], [[0, 0]], [[0]], [[0]], [[1]], [
    [1]], [[1, 0, 0, 0, 0]], [[1, 0, 0, 0, 0]], [[0]], [[0, 0, 0, 0]], [[0]], [[0, 0, 0]], [[1]], [[0]],[[0, 0]],'']
old_vae_number = [-1]
continue_script = 1
cumulative_shift = 0
os.chdir(os.path.dirname(__file__))

def set_scheduler(scheduler_number, first_model, model_ids, scheduler_model):
    # Save and re-use scheduler config to avoid needing to reload pipeline
    if scheduler_model == None:
        scheduler_model = first_model
    try:
        sched_config = torch.load(
            path_to_scheduler_config+str(scheduler_model)+'.pt')
    except:
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_ids[scheduler_model])
        display_status("Setting scheduler "+str(scheduler_number) +
                       " using model "+str(scheduler_model))
        torch.save(pipeline.scheduler.config,
                   path_to_scheduler_config+str(scheduler_model)+'.pt')
        sched_config = pipeline.scheduler.config
    if scheduler_number == 0:
        from diffusers import PNDMScheduler
        scheduler = PNDMScheduler.from_config(sched_config)
    elif scheduler_number == 1:
        from diffusers import DDIMScheduler
        scheduler = DDIMScheduler.from_config(sched_config)
    elif scheduler_number == 2:
        from diffusers import LMSDiscreteScheduler
        scheduler = LMSDiscreteScheduler.from_config(sched_config)
    elif scheduler_number == 3:
        from diffusers import EulerDiscreteScheduler
        scheduler = EulerDiscreteScheduler.from_config(sched_config)
    elif scheduler_number == 4:
        from diffusers import EulerAncestralDiscreteScheduler
        scheduler = EulerAncestralDiscreteScheduler.from_config(sched_config)
    elif scheduler_number == 5:
        from diffusers import DPMSolverMultistepScheduler
        scheduler = DPMSolverMultistepScheduler.from_config(sched_config)
    elif scheduler_number == 6:
        from diffusers import DDPMScheduler
        scheduler = DDPMScheduler.from_config(sched_config)
    elif scheduler_number == 7:
        from diffusers import KDPM2DiscreteScheduler
        scheduler = KDPM2DiscreteScheduler.from_config(sched_config)
    elif scheduler_number == 8:
        from diffusers import DPMSolverSinglestepScheduler
        scheduler = DPMSolverSinglestepScheduler.from_config(sched_config)
    elif scheduler_number == 9:
        from diffusers import DEISMultistepScheduler
        scheduler = DEISMultistepScheduler.from_config(sched_config)
    elif scheduler_number == 10:
        from diffusers import UniPCMultistepScheduler
        scheduler = UniPCMultistepScheduler.from_config(sched_config)
    elif scheduler_number == 11:
        from diffusers import HeunDiscreteScheduler
        scheduler = HeunDiscreteScheduler.from_config(sched_config)
    elif scheduler_number == 12:
        from diffusers import KDPM2AncestralDiscreteScheduler
        scheduler = KDPM2AncestralDiscreteScheduler.from_config(sched_config)
    elif scheduler_number == 13:
        from diffusers import KDPM2DiscreteScheduler
        scheduler = KDPM2DiscreteScheduler.from_config(sched_config)
    elif scheduler_number == 14:
        from diffusers import DPMSolverSDEScheduler
        scheduler = DPMSolverSDEScheduler.from_config(sched_config)
    return scheduler


def build_causal_attention_mask(bsz, seq_len, dtype):
    mask = torch.empty(bsz, seq_len, seq_len, dtype=dtype)
    mask.fill_(torch.tensor(torch.finfo(dtype).min))
    mask.triu_(1)
    mask = mask.unsqueeze(1)
    return mask


def vary_embedding(embedding, mult_prompt_variables, add_prompt_variables):
    # MULTIPLICATIONS
    if mult_prompt_variables != [[1, 0, 0, 0, 0]]:
        for entry in mult_prompt_variables:
            if len(entry) >= 2:
                if entry[1] == -1:
                    entry[1] = embedding.shape[2]-1
            if len(entry) >= 3:
                if entry[2] == -1:
                    entry[2] = embedding.shape[2]-1
            if len(entry) == 5:
                embedding[0, entry[3]:entry[4]+1,
                          entry[1]:entry[2]+1] *= entry[0]
            elif len(entry) == 4:
                embedding[0, entry[3], entry[1]:entry[2]+1] *= entry[0]
            elif len(entry) == 3:
                embedding[0, :, entry[1]:entry[2]+1] *= entry[0]
            elif len(entry) == 2:
                embedding[0, :, entry[1]] *= entry[0]
            elif len(entry) == 1:
                embedding *= entry[0]
    # ADDITIONS
    if add_prompt_variables != [[0, 0, 0, 0, 0]]:
        for entry in add_prompt_variables:
            if len(entry) >= 2:
                if entry[1] == -1:
                    entry[1] = embedding.shape[2]-1
            if len(entry) >= 3:
                if entry[2] == -1:
                    entry[2] = embedding.shape[2]-1
            if len(entry) == 5:
                embedding[0, entry[3]:entry[4]+1,
                          entry[1]:entry[2]+1] += entry[0]
            elif len(entry) == 4:
                embedding[0, entry[3], entry[1]:entry[2]+1] += entry[0]
            elif len(entry) == 3:
                embedding[0, :, entry[1]:entry[2]+1] += entry[0]
            elif len(entry) == 2:
                embedding[0, :, entry[1]] += entry[0]
            elif len(entry) == 1:
                embedding += entry[0]
    return embedding

def comb_modify(combinations_in):
    new_combination_set=[]
    for combination in combinations_in:
        new_combinations=[combination]
        for element in range(len(combination)):
            if '~~' in combination[element]:
                varpos=combination[element].replace('~~','')
                varneg=combination[element].replace('~~','-')
                if len(new_combinations)==1:
                    new_combinations=[]
                    varposcomb=list(combination)
                    varposcomb[element]=varpos
                    new_combinations.append(varposcomb)
                    varnegcomb=list(combination)
                    varnegcomb[element]=varneg
                    new_combinations.append(varnegcomb)
                else:
                    altered_combinations=[]
                    for oldcombo in new_combinations:
                        varposcomb=oldcombo
                        varposcomb[element]=varpos
                        altered_combinations.append(varposcomb.copy())
                        varnegcomb=oldcombo
                        varnegcomb[element]=varneg
                        altered_combinations.append(varnegcomb.copy())
                    new_combinations=altered_combinations
        for combination in new_combinations:
            new_combination_set.append(combination)
    return new_combination_set

def build_raw_embedding(token_numbers, token_emb_layer, prompt_variables):
    
    # Dummy placeholder values
    raw_prompt_embedding_order=[]
    order_count=[]
    raw_prompt_embedding_mean=[]
    raw_prompt_embedding_mean_alt=[]
    
    raw_prompt_embedding = torch.zeros((1, 77, token_emb_layer.embedding_dim))
    asterisk_counter = 0  # counts asterisks encountered in prompt
    and_counter = 0  # counts &s encountered in prompt
    adjust_padding = 0  # turned on after reaching (first) end token
    modify_current = None
    counter_offset = 0
    start_and_end = [0, 49406, 49407]
    for counter_orig in range(77):
        counter = counter_orig+counter_offset
        # Check for asterisk (vocabulary list item 9 or 265)
        if int(token_numbers[counter_orig]) == 9 or int(token_numbers[counter_orig]) == 265:
            raw_prompt_embedding[0, counter, :] = torch.zeros(
                (1, token_emb_layer.embedding_dim))
            if prompt_variables[2] != [[0, 0, 0]]:
                for entry in prompt_variables[2]:
                    if entry[2] == asterisk_counter or entry[2] == -1:
                        display_status(
                            "Row "+str(counter)+": Applying "+str(entry[0:2])+" at asterisk "+str(entry[2]))
                        raw_prompt_embedding[0, counter, entry[0]] += entry[1]
            asterisk_counter += 1  # iterate asterisk count

        # Padding row adjustment
        elif adjust_padding == 1:
            if prompt_variables[6] != [[0]]:
                raw_prompt_embedding[0, counter, :] = token_emb_layer(
                    token_numbers[counter_orig])*prompt_variables[6][0][0]
            if prompt_variables[3] != [[0, 0]]:
                for entry in prompt_variables[3]:
                    if len(entry) == 1:
                        raw_prompt_embedding[0, counter, :] += entry[0]
                    if len(entry) == 2:
                        # order inverted from earlier draft
                        raw_prompt_embedding[0, counter, entry[1]] += entry[0]
                    elif len(entry) == 3:
                        raw_prompt_embedding[0, counter,
                                             entry[1]:entry[2]+1] += entry[0]
            if prompt_variables[4] != [[0]]:
                raw_prompt_embedding[0, counter, raw_prompt_embedding_order[-order_count]
                                     ] += raw_prompt_embedding_mean_alt[raw_prompt_embedding_order[-order_count]]
                order_count += 1
            if prompt_variables[5] != [[0]]:
                raw_prompt_embedding[0, counter,
                                     :] += raw_prompt_embedding_mean*prompt_variables[5][0][0]
        else:
            raw_prompt_embedding[0, counter, :] = token_emb_layer(
                token_numbers[counter_orig])
            # Steps to take once end token is reached
            if int(token_numbers[counter_orig]) == 49407:
                endtok_current = counter
                # if prompt_variables[3]!=[[0,0]] or prompt_variables[4]!=[[0]] or prompt_variables[5]!=[[0]] or prompt_variables[6]==[[0]] or prompt_variables[10]!=[[0]]:
                if prompt_variables[3] != [[0, 0]] or prompt_variables[4] != [[0]] or prompt_variables[5] != [[0]] or prompt_variables[6] != [[1]] or prompt_variables[10] != [[0]]:
                    display_status(
                        "Row "+str(counter)+": Found first end token, will adjust rows accordingly")
                    adjust_padding = 1
                # Calculate variables for dynamic padding
                if prompt_variables[4] != [[0]] or prompt_variables[5] != [[0]]:
                    display_status("Calculating variables for dynamic padding")
                    raw_prompt_embedding_sum = torch.sum(
                        raw_prompt_embedding[0, 1:counter-1, :], 0)
                    raw_prompt_embedding_mean = raw_prompt_embedding_sum / \
                        (counter-1)  # convert to mean
                    if prompt_variables[4] != [[0]]:
                        raw_prompt_embedding_mean_alt = raw_prompt_embedding_mean * \
                            prompt_variables[4][0][0]  # multiply by designated factor
                        raw_prompt_embedding_sum_abs = abs(
                            raw_prompt_embedding_sum)
                        raw_prompt_embedding_order = raw_prompt_embedding_sum_abs.argsort()
                        order_count = 1  # had been 0

        # If modifying all prompt rows
        if prompt_variables[11] != None:
            for entry in prompt_variables[11]:
                if entry[2] == -1 and token_numbers[counter_orig] not in start_and_end:
                    display_status("Row "+str(counter) +
                                   ": Applying "+str(entry[0:2])+" at & ")
                    if entry[3] == 0:
                        raw_prompt_embedding[0, counter, entry[0]] += entry[1]
                    elif entry[3] == 1:
                        if entry[0] == -1:
                            raw_prompt_embedding[0, counter, :] *= entry[1]
                        else:
                            raw_prompt_embedding[0,
                                                 counter, entry[0]] *= entry[1]

        # If preceding original token was &
        if modify_current != None:
            for entry in modify_current:
                if entry[2] == and_counter:
                    display_status("Row "+str(counter)+": Applying " +
                                   str(entry[0:2])+" at & "+str(entry[2]))
                    if entry[3] == 0:
                        raw_prompt_embedding[0, counter, entry[0]] += entry[1]
                    elif entry[3] == 1:
                        if entry[0] == -1:
                            raw_prompt_embedding[0, counter, :] *= entry[1]
                        else:
                            raw_prompt_embedding[0,
                                                 counter, entry[0]] *= entry[1]
            and_counter += 1  # iterate asterisk count
            modify_current = None

        # Check for & (vocabulary list 261)
        if int(token_numbers[counter_orig]) == 261:
            modify_current = prompt_variables[11]
            counter_offset += -1

    # fill in any missing rows with existing value of "last" row
    for counter_add in range(counter+1, 77):
        raw_prompt_embedding[0, counter_add,
                             :] = raw_prompt_embedding[0, counter, :]

    if prompt_variables[10] != [[0]]:  # move end token to new location
        endtok_target = prompt_variables[10][0][0]
        if endtok_target < 0:
            endtok_target = endtok_current-endtok_target
        if endtok_target > 76:
            endtok_target = 76
        embedding_temp = torch.zeros_like(raw_prompt_embedding)
        embedding_temp[0, :endtok_current,
                       :] = raw_prompt_embedding[0, :endtok_current, :]
        embedding_temp[0, endtok_current:-1,
                       :] = raw_prompt_embedding[0, endtok_current+1, :]
        if endtok_target != 76:
            embedding_temp[0, endtok_target+1:,
                           :] = embedding_temp[0, endtok_target:-1, :]
            embedding_temp[0, endtok_target,
                           :] = raw_prompt_embedding[0, endtok_current, :]
        else:
            embedding_temp[0, 76, :] = raw_prompt_embedding[0,
                                                            endtok_current, :]
        raw_prompt_embedding = embedding_temp

    return raw_prompt_embedding


def generate_prompt_embedding(prompt, tokenizer, text_encoder, tokenizer_2, text_encoder_2):
    prompt_text = prompt[0]
    prompt_variables = prompt[1]  # consists of [[first],[second]]

    if not isinstance(prompt_text, list):
        prompt_text = [prompt_text, prompt_text, prompt_text]
    elif len(prompt[0]) == 2:
        prompt_text.append(prompt_text[1])

    # FIRST EMBEDDING (OR ONLY EMBEDDING)

    if prompt_text[0] != '~0':
        
        if prompt_variables[0][17]=='':
            prompts=[prompt_text[0]]
        else:
            prompts=[prompt_text[0],prompt_variables[0][17]]
               
        embedding_array=[]
        end_token_row_array=[]
        for prompt_working in prompts:    
            display_status("Generating embedding for prompt " +
                       str(prompt_working))

            prompt_working = prompt_working.replace('*', '* ')
            prompt_working = prompt_working.replace('  ', ' ')
            prompt_working = prompt_working.replace('&', '& ')
    
            # Translate natural-language prompt into sequence of numerical tokens
            text_inputs = tokenizer(
                prompt_working,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            token_numbers = text_inputs.input_ids[0, :]
    
            # Build raw prompt embedding by looking up the numerical array for each token
            token_emb_layer = text_encoder.text_model.embeddings.token_embedding
            raw_prompt_embedding = build_raw_embedding(
                token_numbers, token_emb_layer, prompt_variables[0])
    
            # Multiplications and additions to raw embedding
            raw_prompt_embedding = vary_embedding(
                raw_prompt_embedding, prompt_variables[0][8], prompt_variables[0][0])
    
            # Add positional embedding to raw prompt embedding to create combined raw embedding
            if prompt_variables[0][7] != [[0]]:
                pos_emb_layer = text_encoder.text_model.embeddings.position_embedding
                position_ids = text_encoder.text_model.embeddings.position_ids[:, :77]
                position_embedding = pos_emb_layer(position_ids)
                final_raw_embedding = raw_prompt_embedding + \
                    (position_embedding*prompt_variables[0][7][0][0])
            else:
                final_raw_embedding = raw_prompt_embedding
    
            final_raw_embedding = final_raw_embedding

            # Generate processed embedding from combined raw embedding
            bsz, seq_len = final_raw_embedding.shape[:2]
            causal_attention_mask = build_causal_attention_mask(
                bsz, seq_len, dtype=final_raw_embedding.dtype)
            encoder_outputs = text_encoder.text_model.encoder(
                inputs_embeds=final_raw_embedding,
                attention_mask=None,
                causal_attention_mask=causal_attention_mask,
                output_attentions=None,
                output_hidden_states=True
            )
            if tokenizer_2 == '':  # for non-XL models
                if prompt_variables[0][12] == [[0]]:  # variable 12 is clipskip
                    # Get output hidden state
                    output = encoder_outputs[0]
                else:
                    output = encoder_outputs.hidden_states[-(
                        prompt_variables[0][12][0][0]+1)]
                if prompt_variables[0][13] != [[0, 0, 0]]:  # cliprange
                    output = torch.zeros_like(output)
                    if prompt_variables[0][13][0][2] < 2:
                        layer_counter = 0
                        for layer in range(prompt_variables[0][13][0][0], prompt_variables[0][13][0][1]+1):
                            output += encoder_outputs.hidden_states[-layer]
                            layer_counter += 1
                        if prompt_variables[0][13][0][2] == 0:
                            output /= layer_counter
                    else:
                        layer_count = (
                            prompt_variables[0][13][0][1]-prompt_variables[0][13][0][0])+1
                        layer_array = torch.zeros(
                            output.shape[0], output.shape[1], output.shape[2], layer_count)
                        layer_counter = 0
                        for layer in range(prompt_variables[0][13][0][0], prompt_variables[0][13][0][1]+1):
                            layer_array[:, :, :,
                                        layer_counter] = encoder_outputs.hidden_states[-layer]
                            layer_counter += 1
                        if prompt_variables[0][13][0][2] == 2:
                            output = torch.median(layer_array, 3)[0]
                
                # Add any requested layers
                new_layers=int(prompt_variables[0][16][0][0])
                if len(prompt_variables[0][16][0])==1:
                    prompt_variables[0][16][0].append(0)
                while new_layers>0:
                    if prompt_variables[0][16][0][1]==1:
                        # Pass through final layer norm each "round" if indicated
                        output = text_encoder.text_model.final_layer_norm(
                            output)
                    bsz, seq_len = output.shape[:2]
                    causal_attention_mask = build_causal_attention_mask(
                        bsz, seq_len, dtype=output.dtype)
                    encoder_outputs = text_encoder.text_model.encoder(
                        inputs_embeds=output,
                        attention_mask=None,
                        causal_attention_mask=causal_attention_mask,
                        output_attentions=None,
                        output_hidden_states=True
                    )
                    num_layers=len(encoder_outputs.hidden_states)
                    if new_layers>=num_layers:
                        output = encoder_outputs[0]
                    else:
                        output = encoder_outputs.hidden_states[new_layers]
                    new_layers-=num_layers
                processed_embedding = text_encoder.text_model.final_layer_norm(
                    output)
                
            else:  # for XL models
                processed_embedding = encoder_outputs.hidden_states[-(
                    2+prompt_variables[0][12][0][0])]
                if prompt_variables[0][13] != [[0, 0, 0]]:
                    processed_embedding = torch.zeros_like(processed_embedding)
                    if prompt_variables[0][13][0][2] < 2:
                        layer_counter = 0
                        for layer in range(prompt_variables[0][13][0][0], prompt_variables[0][13][0][1]+1):
                            processed_embedding += encoder_outputs.hidden_states[-layer]
                            layer_counter += 1
                        if prompt_variables[0][13][0][2] == 0:
                            processed_embedding /= layer_counter
                    else:
                        layer_count = (
                            prompt_variables[0][13][0][1]-prompt_variables[0][13][0][0])+1
                        layer_array = torch.zeros(
                            processed_embedding.shape[0], processed_embedding.shape[1], processed_embedding.shape[2], layer_count)
                        layer_counter = 0
                        for layer in range(prompt_variables[0][13][0][0], prompt_variables[0][13][0][1]+1):
                            layer_array[:, :, :,
                                        layer_counter] = encoder_outputs.hidden_states[-layer]
                            layer_counter += 1
                        if prompt_variables[0][13][0][2] == 2:
                            processed_embedding = torch.median(layer_array, 3)[0]
                # Add any requested layers
                new_layers=int(prompt_variables[0][16][0][0])
                while new_layers>0:
                    bsz, seq_len = processed_embedding.shape[:2]
                    causal_attention_mask = build_causal_attention_mask(
                        bsz, seq_len, dtype=processed_embedding.dtype)
                    encoder_outputs = text_encoder.text_model.encoder(
                        inputs_embeds=processed_embedding,
                        attention_mask=None,
                        causal_attention_mask=causal_attention_mask,
                        output_attentions=None,
                        output_hidden_states=True
                    )
                    num_layers=len(encoder_outputs.hidden_states)
                    if new_layers>=num_layers:
                        output = encoder_outputs[0]
                    else:
                        output = encoder_outputs.hidden_states[new_layers-1]
                    new_layers-=num_layers
                
                    processed_embedding = output                 
                            
            end_token_row = int(
                (token_numbers == 49407).nonzero(as_tuple=True)[0][0])
            if prompt_variables[0][14] != [[1]]:
                for entry in prompt_variables[0][14]:
                    if len(entry) == 1:
                        processed_embedding[:, 1:end_token_row, :] *= entry[0]
                    elif len(entry) == 2:
                        processed_embedding[:, 1:end_token_row,
                                            entry[1]] *= entry[0]
                    elif len(entry) == 3:
                        processed_embedding[:, 1:end_token_row,
                                            entry[1]:entry[2]+1] *= entry[0]
            if prompt_variables[0][15] != [[0]]:
                for entry in prompt_variables[0][15]:
                    if len(entry) == 1:
                        processed_embedding[:, 1:end_token_row, :] += entry[0]
                    elif len(entry) == 2:
                        processed_embedding[:, 1:end_token_row,
                                            entry[1]] += entry[0]
                    elif len(entry) == 3:
                        processed_embedding[:, 1:end_token_row,
                                            entry[1]:entry[2]+1] += entry[0]
            embedding_array.append(processed_embedding)
            end_token_row_array.append(end_token_row)
        if len(embedding_array)==1:
            processed_embedding=embedding_array[0]
        else:
            processed_embedding=embedding_array[0]
            if end_token_row_array[0]>=end_token_row_array[1]:
                processed_embedding[:, end_token_row_array[0]+1:,:]=embedding_array[1][:, end_token_row_array[0]+1:,:]
            else:
                processed_embedding[:, end_token_row_array[0]+1:end_token_row_array[1]+1,:]*=0
                processed_embedding[:, end_token_row_array[1]+1:,:]=embedding_array[1][:, end_token_row_array[1]+1:,:]
        # Multiplications and additions to processed embedding
        processed_embedding = vary_embedding(
            processed_embedding, prompt_variables[0][9], prompt_variables[0][1])
    else:
        display_status('Primary embedding zeroed out.')
        processed_embedding = torch.zeros(
            1, 77, text_encoder.config.projection_dim)  # get the actual dimension
    if tokenizer_2 != '':
        prompt_embeds_list = [processed_embedding]

        # SECOND EMBEDDING FOR XL MODELS
        if prompt_text[1] != '~0':
            if prompt_text[1] == '~':
                prompt_text[1] = ''
            
            if prompt_variables[1][17]=='':
                prompts=[prompt_text[1]]
            else:
                prompts=[prompt_text[1],prompt_variables[1][17]]
                   
            embedding_array=[]
            end_token_row_array=[]
            for prompt_working in prompts:
                
                display_status(
                    "Generating second embedding for prompt "+str(prompt_working))
                prompt_working = prompt_working.replace('*', '* ')
                prompt_working = prompt_working.replace('  ', ' ')
                prompt_working = prompt_working.replace('&', '& ')
                text_inputs = tokenizer_2(
                    prompt_working,
                    padding="max_length",
                    max_length=tokenizer_2.model_max_length,
                    truncation=True,
                    return_tensors="pt"
                )
                text_input_ids = text_inputs.input_ids
                token_numbers = text_inputs.input_ids[0, :]
                token_emb_layer = text_encoder_2.text_model.embeddings.token_embedding
                raw_prompt_embedding = build_raw_embedding(
                    token_numbers, token_emb_layer, prompt_variables[1])
                raw_prompt_embedding = vary_embedding(
                    raw_prompt_embedding, prompt_variables[1][8], prompt_variables[1][0])
                if prompt_variables[1][7] != [[0]]:
                    pos_emb_layer = text_encoder_2.text_model.embeddings.position_embedding
                    position_ids = text_encoder_2.text_model.embeddings.position_ids[:, :77]
                    position_embedding = pos_emb_layer(position_ids)
                    final_raw_embedding = raw_prompt_embedding + \
                        (position_embedding*prompt_variables[1][7][0][0])
                else:
                    final_raw_embedding = raw_prompt_embedding
    
                bsz, seq_len = final_raw_embedding.shape[:2]
                causal_attention_mask = build_causal_attention_mask(
                    bsz, seq_len, dtype=final_raw_embedding.dtype)
                encoder_outputs = text_encoder_2.text_model.encoder(
                    inputs_embeds=final_raw_embedding,
                    attention_mask=None,
                    causal_attention_mask=causal_attention_mask,
                    output_attentions=None,
                    output_hidden_states=True
                )
                processed_embedding = encoder_outputs.hidden_states[-(
                    2+prompt_variables[1][12][0][0])]
                if prompt_variables[1][13] != [[0, 0, 0]]:
                    processed_embedding = torch.zeros_like(processed_embedding)
                    if prompt_variables[1][13][0][2] < 2:
                        layer_counter = 0
                        for layer in range(prompt_variables[1][13][0][0], prompt_variables[1][13][0][1]+1):
                            processed_embedding += encoder_outputs.hidden_states[-layer]
                            layer_counter += 1
                        if prompt_variables[1][13][0][2] == 0:
                            processed_embedding /= layer_counter
                    else:
                        layer_count = (
                            prompt_variables[1][13][0][1]-prompt_variables[1][13][0][0])+1
                        layer_array = torch.zeros(
                            processed_embedding.shape[0], processed_embedding.shape[1], processed_embedding.shape[2], layer_count)
                        layer_counter = 0
                        for layer in range(prompt_variables[1][13][0][0], prompt_variables[1][13][0][1]+1):
                            layer_array[:, :, :,
                                        layer_counter] = encoder_outputs.hidden_states[-layer]
                            layer_counter += 1
                        if prompt_variables[1][13][0][2] == 2:
                            processed_embedding = torch.median(layer_array, 3)[0]
                # Add any requested layers
                new_layers=int(prompt_variables[1][16][0][0])
                while new_layers>0:
                    bsz, seq_len = processed_embedding.shape[:2]
                    causal_attention_mask = build_causal_attention_mask(
                        bsz, seq_len, dtype=processed_embedding.dtype)
                    encoder_outputs = text_encoder_2.text_model.encoder(
                        inputs_embeds=processed_embedding,
                        attention_mask=None,
                        causal_attention_mask=causal_attention_mask,
                        output_attentions=None,
                        output_hidden_states=True
                    )
                    num_layers=len(encoder_outputs.hidden_states)
                    if new_layers>=num_layers:
                        output = encoder_outputs[0]
                    else:
                        output = encoder_outputs.hidden_states[new_layers-1]
                    new_layers-=num_layers
                
                    processed_embedding = output
                end_token_row = int(
                    (token_numbers == 49407).nonzero(as_tuple=True)[0][0])
                if prompt_variables[1][14] != [[1]]:
                    for entry in prompt_variables[1][14]:
                        if len(entry) == 1:
                            processed_embedding[:, 1:end_token_row, :] *= entry[0]
                        elif len(entry) == 2:
                            processed_embedding[:, 1:end_token_row,
                                                entry[1]] *= entry[0]
                        elif len(entry) == 3:
                            processed_embedding[:, 1:end_token_row,
                                                entry[1]:entry[2]+1] *= entry[0]
                if prompt_variables[1][15] != [[0]]:
                    for entry in prompt_variables[1][15]:
                        if len(entry) == 1:
                            processed_embedding[:, 1:end_token_row, :] += entry[0]
                        elif len(entry) == 2:
                            processed_embedding[:, 1:end_token_row,
                                                entry[1]] += entry[0]
                        elif len(entry) == 3:
                            processed_embedding[:, 1:end_token_row,
                                                entry[1]:entry[2]+1] += entry[0]
            embedding_array.append(processed_embedding)
            end_token_row_array.append(end_token_row)
            if len(embedding_array)==1:
                processed_embedding=embedding_array[0]
            else:
                processed_embedding=embedding_array[0]
                if end_token_row_array[0]>=end_token_row_array[1]:
                    processed_embedding[:, end_token_row_array[0]+1:,:]=embedding_array[1][:, end_token_row_array[0]+1:,:]
                else:
                    processed_embedding[:, end_token_row_array[0]+1:end_token_row_array[1]+1,:]*=0
                    processed_embedding[:, end_token_row_array[1]+1:,:]=embedding_array[1][:, end_token_row_array[1]+1:,:]
            end_token_row=end_token_row_array[0]
            processed_embedding = vary_embedding(
                processed_embedding, prompt_variables[1][9], prompt_variables[1][1])
        else:  # zero out the embedding
            display_status("Second embedding zeroed out.")
            processed_embedding = torch.zeros(
                1, 77, text_encoder_2.config.projection_dim)
        prompt_embeds_list.append(processed_embedding)
        processed_embedding = torch.concat(prompt_embeds_list, dim=-1)

        # THIRD EMBEDDING FOR XL MODELS
        if prompt_text[2] != '~0':
            if prompt_text[2] == '~':
                prompt_text[2] = ''
            display_status(
                "Generating third embedding for prompt "+str([prompt_text[2]]))
            text_inputs = tokenizer_2(
                prompt_text[2],
                padding="max_length",
                max_length=tokenizer_2.model_max_length,
                truncation=True,
                return_tensors="pt"
            )
            # NONVARIABLE PLACEHOLDER UNTIL I CAN FIGURE THIS OUT
            text_input_ids = text_inputs.input_ids
            prompt_embeds_2 = text_encoder_2(
                text_input_ids.to("cpu"),
                output_hidden_states=True
            )
            # This doesn't vary by clip skip
            pooled_embedding = prompt_embeds_2[0]
        else:  # zero out the embedding
            display_status("Third embedding zeroed out.")
            pooled_embedding = torch.zeros(
                1, text_encoder_2.config.projection_dim)
    else:
        pooled_embedding = ''
    return [processed_embedding, pooled_embedding]

def latent_shift(latent, v_shift, h_shift):
    latent_temp = torch.zeros_like(latent)
    latent_temp[:, :, :latent.shape[2]-v_shift, :] = latent[:, :, v_shift:, :]
    latent_temp[:, :, latent.shape[2]-v_shift:, :] = latent[:, :, :v_shift, :]
    latent = latent_temp
    latent_temp = torch.zeros_like(latent)
    latent_temp[:, :, :, :latent.shape[3]-h_shift] = latent[:, :, :, h_shift:]
    latent_temp[:, :, :, latent.shape[3]-h_shift:] = latent[:, :, :, :h_shift]
    return latent_temp


def get_white_latent(latent):
    if latent.ndim == 4:
        white_latent = torch.zeros_like(latent)
        white_latent[:, 0, :, :] += 0.238
        white_latent[:, 1, :, :] += 0.156
        white_latent[:, 3, :, :] += -0.126
    elif latent.ndim == 3:
        white_latent = torch.zeros_like(latent)
        white_latent[0, :, :] += 0.238
        white_latent[1, :, :] += 0.156
        white_latent[3, :, :] += -0.126
    return white_latent


def vary_latent(latent, noise_variables,mix):
    global cumulative_shift
    if noise_variables != [[[1, 1, 1, 1]], [[0, 0, 0, 0]], [[0]], [[0, 0]], [[0, 0, 0, 0, 0]]]:
        if noise_variables[0] != [[1, 1, 1, 1]]:
            if len(noise_variables[0][0]) == 1:
                latent *= noise_variables[0][0][0]
            else:
                for channel in range(4):
                    latent[:, channel, :, :] *= noise_variables[0][0][channel]
        if noise_variables[1] != [[0, 0, 0, 0]]:
            if len(noise_variables[1][0]) == 1:
                latent += noise_variables[1][0][0]
            else:
                for channel in range(4):
                    latent[:, channel, :, :] += noise_variables[1][0][channel]
        latent_to_add = latent  # to hold stable through the following processes
        for alteration in noise_variables[2]:
            if alteration == [1]:  # 2x width
                latent = latent.tile((1, 2))
            elif alteration == [2]:  # 2x height
                latent = latent.tile((2, 1))
            elif alteration == [3]:  # flip LR
                latent = latent.flip((3, 0))
            elif alteration == [4]:  # flip UD
                latent = latent.flip((2, 0))
            elif alteration == [5]:  # flip LR and UD
                latent = latent.flip((3, 0))
                latent = latent.flip((2, 0))  # can these be combined?
            elif alteration == [6]:  # 2x = RIGHT FLIPPED LR
                latent_flip = latent.flip((3, 0))
                latent = torch.cat((latent, latent_flip), dim=3)
            elif alteration == [7]:  # 2x = BOTTOM FLIPPED LR
                latent_flip = latent.flip((3, 0))
                latent = torch.cat((latent, latent_flip), dim=2)
            elif alteration == [8]:  # 2x = RIGHT FLIPPED UD
                latent_flip = latent.flip((2, 0))
                latent = torch.cat((latent, latent_flip), dim=3)
            elif alteration == [9]:  # 2x = BOTTOM FLIPPED UD
                latent_flip = latent.flip((2, 0))
                latent = torch.cat((latent, latent_flip), dim=2)
            elif alteration == [10]:  # 2x = BOTTOM FLIPPED LR AND UD
                latent_flip = latent.flip((2, 0))
                latent_flip = latent_flip.flip((3, 0))
                latent = torch.cat((latent, latent_flip), dim=3)
            elif alteration == [11]:  # 2x = RIGHT FLIPPED UD AND LR
                latent_flip = latent.flip((2, 0))
                latent_flip = latent_flip.flip((3, 0))
                latent = torch.cat((latent, latent_flip), dim=2)
            elif alteration == [12]:  # ROTATE CLOCKWISE
                latent = torch.rot90(latent, 1, (3, 2))
            elif alteration == [13]:  # ROTATE COUNTERCLOCKWISE
                latent = torch.rot90(latent, 1, (2, 3))
            elif alteration == [14]:  # INVERT RIGHT HALF
                latent_half_width = int(latent.shape[3]/2)
                latent[:, :, :, latent_half_width:] = latent[:,
                                                             :, :, latent_half_width:]*-1
            elif alteration == [15]:  # INVERT BOTTOM HALF
                latent_half_height = int(latent.shape[2]/2)
                latent[:, :, latent_half_height:,
                       :] = latent[:, :, latent_half_height:, :]*-1
            elif alteration == [16]:  # CROP TO MIDDLE HALF
                latent_third_width = int(latent.shape[3]/3)
                latent_half_width = int(latent.shape[3]/2)
                latent = latent[:, :, :,
                                latent_third_width:latent_third_width+latent_half_width+1]
            elif alteration == [17]:  # CROP TO LEFT HALF
                latent_half_width = int(latent.shape[3]/2)
                latent = latent[:, :, :, :latent_half_width]
            elif alteration == [18]:  # CROP TO TOP HALF
                latent_half_height = int(latent.shape[2]/2)
                latent = latent[:, :, :latent_half_height, :]
            # ADDED 20240207:
            elif alteration == [20]:  # add regular copy to right
                latent = torch.cat((latent, latent_to_add), dim=3)
            elif alteration == [21]:  # add flip UD copy to right
                latent = torch.cat((latent, latent_to_add.flip((2, 0))), dim=3)
            elif alteration == [22]:  # add flip LR copy to right
                latent = torch.cat((latent, latent_to_add.flip((3, 0))), dim=3)
            elif alteration == [23]:  # add flip LR+UD copy to right
                latent_flip = latent_to_add.flip((2, 0))
                latent_flip = latent_flip.flip((3, 0))
                latent = torch.cat((latent, latent_flip), dim=3)
            elif alteration == [24]:  # add inverted copy to right
                latent = torch.cat((latent, latent_to_add*-1), dim=3)
            elif alteration == [25]:  # add white latent to right
                latent = torch.cat(
                    (latent, get_white_latent(latent_to_add)), dim=3)

        if noise_variables[3] != [[0, 0]]:
            if len(noise_variables[3][0]) == 1:
                v_shift = noise_variables[3][0][0]
                h_shift = 0
            elif len(noise_variables[3][0]) == 2:
                v_shift = noise_variables[3][0][0]
                h_shift = noise_variables[3][0][1]
            if v_shift != 0:
                display_status("Vertical shift "+str(v_shift))
                # specified row becomes the top row, rest moved to bottom
                if v_shift < 0:
                    v_shift = latent.shape[2]+v_shift
                    display_status("Vertical shift equivalent to "+str(v_shift))
                if v_shift < 0 or v_shift >= latent.shape[2]:
                    display_status(
                        "Requested vertical shift exceeds latent dimension, not applied")
                else:
                    latent = latent_shift(latent, v_shift, 0)
            if h_shift != 0:
                display_status("Horizontal shift "+str(h_shift))
                if h_shift < 0:
                    h_shift = latent.shape[3]+h_shift
                    display_status("Horizontal shift equivalent to "+str(h_shift))
                if h_shift < 0 or h_shift >= latent.shape[3]:
                    display_status(
                        "Requested horizontal shift exceeds latent dimensions, not applied")
                # specified row becomes the leftmost row, rest moved to right
                else:
                    latent = latent_shift(latent, 0, h_shift)
            if mix==0:
                cumulative_shift[0] += v_shift
                cumulative_shift[1] += h_shift
                display_status("Cumulative shift "+str(cumulative_shift))
        if noise_variables[4] != [[0, 0, 0, 0, 0]]:
            for entry in noise_variables[4]:
                for npart_val in range(2):
                    if entry[npart_val]<=0 and entry[npart_val] !=int(entry[npart_val]):
                        entry[npart_val]=-int(entry[npart_val]*latent.shape[2])
                for npart_val in range(2,4):
                    if entry[npart_val]<=0 and entry[npart_val] !=int(entry[npart_val]):
                        entry[npart_val]=-int(entry[npart_val]*latent.shape[3])
                
                if entry[4] == 5:  # crop to rectangle
                    latent = latent[:, :, entry[0]:latent.shape[2] -
                                    entry[1], entry[2]:latent.shape[3]-entry[3]]
                elif entry[4] == 6:  # fill in white
                    latent[0, :, entry[0]:latent.shape[2]-entry[1], entry[2]:latent.shape[3]-entry[3]] = get_white_latent(
                        latent[0, :, entry[0]:latent.shape[2]-entry[1], entry[2]:latent.shape[3]-entry[3]])
                else:
                    for i in range(4):
                        latent_piece = latent[0, i, entry[0]:latent.shape[2] -
                                              entry[1], entry[2]:latent.shape[3]-entry[3]]
                        if entry[4] == 0:
                            latent_piece *= 0
                        elif entry[4] == 1:
                            latent_piece *= -1
                        elif entry[4] == 2:
                            latent_piece = torch.fliplr(latent_piece)
                        elif entry[4] == 3:
                            latent_piece = torch.flipud(latent_piece)
                        elif entry[4] == 4:
                            latent_piece = torch.fliplr(
                                torch.flipud(latent_piece))
                        # ADDED 20240208
                        elif entry[4] == 6:
                            latent_piece = get_white_latent(latent_piece)
                        latent[0, i, entry[0]:latent.shape[2]-entry[1],
                               entry[2]:latent.shape[3]-entry[3]] = latent_piece
    return latent

def to_latent(img: Image, vae):
    generator = torch.Generator("cpu").manual_seed(
        0)
    with torch.no_grad():
        latent = vae.encode(tfms.ToTensor()(img).unsqueeze(
            0).to(torch_device)*2-1)  # Note scaling
    return vae.config.scaling_factor * latent.latent_dist.sample(generator=generator)

def generate_noise_latent(seed_to_use, secondary_seed, height, width, noise_variables, scheduler, vae, num_inference_steps):
    if noise_variables[5] == [['', 0]] and noise_variables[6] == [[0,0,0,0,0]]:
        # Create noise latent from scratch
        generator = torch.Generator('cpu').manual_seed(seed_to_use)
        latent = torch.randn(
            (1, 4, int(height/8), int(width/8)), generator=generator)
    else:
        if noise_variables[5] != [['',0]]:
            image_source = noise_variables[5][0][0]
            display_status('image2image: '+image_source)
            try:
                # Load as PIL image, encode it, add appropriate noise level
                try:
                    image_input = Image.open(work_dir+image_source)
                except:
                    image_input = Image.open(image_source)
                image_input = image_input.convert("RGB")
                torch.manual_seed(0)
                latent = to_latent(image_input, vae)
            except:
                # load .pt file
                try:
                    package = torch.load(work_dir+image_source)
                except:
                    package = torch.load(image_source)
                latent = package[0]
            strength = noise_variables[5][0][1]
        else:
            torch.manual_seed(0)
            add_values=rgb_to_latent(int(noise_variables[6][0][0]),int(noise_variables[6][0][1]),int(noise_variables[6][0][2]),vae)
            latent = torch.zeros(1,4,int(height/8),int(width/8)).to(torch_device)
            for channel in range(4):
                latent[:, channel, :, :] += add_values[channel]
            strength = noise_variables[6][0][3]
        start_step = min(int(num_inference_steps*strength),
                         num_inference_steps)
        # The above effectively inverts the usual strength scale
        # Ordinarily num_inference_steps*strength is number of steps to run
        # Here it's the step number to start with
        generator = torch.Generator('cpu').manual_seed(seed_to_use)
        try:
            if noise_variables[5][0][2]=='1':
                noise = torch.zeros_like(latent.to('cpu'))
        except:
            if noise_variables[6][0][4]!=0:
                noise = torch.zeros_like(latent.to('cpu'))
            else:
                noise = torch.empty_like(latent.to('cpu')).normal_(generator=generator)
        noise = noise.to(torch_device)
        #latent = scheduler.add_noise(latent,noise,timesteps=torch.tensor([scheduler.timesteps[start_step-1]]))
        latent = scheduler.add_noise(latent, noise, timesteps=torch.tensor([
                                     scheduler.timesteps[start_step]]))
    if secondary_seed != None:
        torch.manual_seed(secondary_seed)
    else:
        torch.manual_seed(seed_to_use)
    mix = 0 #not relevant to this function
    latent = vary_latent(latent, noise_variables, mix)
    return latent

def generate_image_latent(latent, scheduler, num_inference_steps, guidance_scales, models, prompts, neg_prompts, step_latent_variables, i2i, rgb, contrastives, models_in_memory, models_in_memory_old, model_ids, shiftback, t_models, skipstep, mixmax_list, mixvalues):
    global prompt_defaults
    global cumulative_shift
    # Set counting variables
    prompt_counter = 0
    neg_prompt_counter = 0
    model_counter = 0
    guidance_counter = 0
    contrastives_counter = 0
    step_latent_counter = 0
    
    if i2i[0] != '':
        strength = i2i[1]
        start_step = min(int(num_inference_steps*strength),
                         num_inference_steps)
        latent = latent.to(torch_device).float()
    elif rgb !=0:
        strength = rgb
        start_step = min(int(num_inference_steps*strength),
                         num_inference_steps)
        latent = latent.to(torch_device).float()
    else:
        start_step = 0
        latent = latent.to(torch_device).float()
        latent = latent*scheduler.init_noise_sigma

    # Cycle through inference steps
    last_model = -1
    last_text_model = -1
    last_prompts = []
        
    latent_input=latent #to refer back to in case of mix
    
    # Expand mixmax_list and mixvalues if needed
    if len(mixmax_list)<len(scheduler.timesteps):
        mixmax_list_counter=0
        slots_to_fill=len(scheduler.timesteps)-len(mixmax_list)
        for j in range(slots_to_fill):
            mixmax_list.append(mixmax_list[mixmax_list_counter])
            mixvalues.append(mixvalues[mixmax_list_counter])
            mixmax_list_counter+=1
            if mixmax_list_counter==len(mixmax_list):
                mixmax_list_counter=0
    
    for tcount in range(len(scheduler.timesteps)-skipstep):
        t=scheduler.timesteps[tcount]
        
        if tcount >= start_step:

            display_status("Inference timestep: "+str(tcount+1) +
                           "/"+str(len(scheduler.timesteps)))
            display_step(str(tcount+1) + "/"+str(len(scheduler.timesteps)))
            
            noise_preds=[]
            
            for mix in range(mixmax_list[tcount]+1):
                latent = latent_input
            
                if tcount == start_step or len(step_latent_variables) > 1:
                    latent = vary_latent(
                        latent, step_latent_variables[step_latent_counter][mix],mix)
                    height = latent.shape[2]*8
                    width = latent.shape[3]*8
                # If multiple models
    
                if tcount == start_step or len(models['sequence']) > 1 or len(t_models['sequence'] > 1):
    
                    if models_in_memory == 1:
                        if models['sequence'][model_counter][mix] != last_model or models_in_memory_old == 0:
                            models_in_memory_old = 1
                            display_status("Loading stored model " +
                                           str(models['sequence'][model_counter][mix]))
                            # Get the model for this step
                            [unet] = models[models['sequence'][model_counter][mix]]
                        if t_models['sequence'][model_counter][mix] != last_text_model or models_in_memory_old == 0:
                            if len(t_models[t_models['sequence'][model_counter][mix]]) == 2:
                                # Get the model for this step
                                [tokenizer, text_encoder] = t_models[t_models['sequence']
                                                                     [model_counter][mix]]
                                tokenizer_2 = ''
                                text_encoder_2 = ''
                            elif len(t_models[t_models['sequence'][model_counter][mix]]) == 4:
                                [tokenizer, text_encoder, tokenizer_2,
                                    text_encoder_2] = t_models[t_models['sequence'][model_counter][mix]]
                        model_value = models['sequence'][model_counter][mix]
                        text_model_value = t_models['sequence'][model_counter][mix]
                    else:
                        if model_ids[models['sequence'][model_counter][mix]] != last_model:
                            display_status("Loading model " +
                                           str(models['sequence'][model_counter][mix]))
                            unet = UNet2DConditionModel.from_pretrained(
                                model_ids[models['sequence'][model_counter][mix]], subfolder="unet")
                            unet = unet.to(torch_device)
                        if model_ids[t_models['sequence'][model_counter][mix]] != last_text_model:
                            text_encoder = CLIPTextModel.from_pretrained(
                                model_ids[t_models['sequence'][model_counter][mix]], subfolder="text_encoder")
                            #text_encoder = text_encoder.to(torch_device)
                            tokenizer = CLIPTokenizer.from_pretrained(
                                model_ids[t_models['sequence'][model_counter][mix]], subfolder="tokenizer")
                            try:
                                text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
                                    model_ids[t_models['sequence'][model_counter][mix]], subfolder="text_encoder_2")
                                # text_encoder_2 = text_encoder_2.to(torch_device) #OOM
                                tokenizer_2 = CLIPTokenizer.from_pretrained(
                                    model_ids[t_models['sequence'][model_counter][mix]], subfolder="tokenizer_2")
                            except:
                                text_encoder_2 = ''
                                tokenizer_2 = ''
                        model_value = model_ids[models['sequence'][model_counter][mix]]
                        text_model_value = model_ids[t_models['sequence']
                                                     [model_counter][mix]]
                # If multiple models or prompts
                if tcount == start_step or (model_value != last_model or text_model_value != last_text_model or prompts[prompt_counter] != last_prompts):
                    last_prompts = prompts[prompt_counter][mix]
                    display_status("EMBEDDING:")
                    [embedding, pooled_embedding] = generate_prompt_embedding(
                        prompts[prompt_counter][mix], tokenizer, text_encoder, tokenizer_2, text_encoder_2)  # Get the embedding for this step
                    # CONTRASTIVE MANIPULATIONS HERE
                    if contrastives[contrastives_counter][mix][0] != [[['', prompt_defaults], 0]]:
                        display_status("Generating single-prompt contrastives:")
                        embedding_reference = embedding  # Hold embedding stable
                        pooled_embedding_reference = pooled_embedding
                        for entry in contrastives[contrastives_counter][mix][0]:
                            [alt_embedding, pooled_alt_embedding] = generate_prompt_embedding(
                                entry[0], tokenizer, text_encoder, tokenizer_2, text_encoder_2)
                            # What's distinctive about 'main' embedding
                            embedding_diff = embedding_reference-alt_embedding
                            embedding_diff *= entry[1]
                            embedding += embedding_diff
                            if pooled_embedding_reference != '':
                                pooled_embedding_diff = pooled_embedding_reference-pooled_alt_embedding
                                pooled_embedding_diff *= entry[1]
                                pooled_embedding += pooled_embedding_diff
                    if contrastives[contrastives_counter][mix][1] != [[['', prompt_defaults], ['', prompt_defaults], 0]]:
                        display_status("Generating prompt-pair contrastives:")
                        for entry in contrastives[contrastives_counter][mix][1]:
                            [alt_embedding_1, pooled_alt_embedding_1] = generate_prompt_embedding(
                                entry[0], tokenizer, text_encoder, tokenizer_2, text_encoder_2)
                            [alt_embedding_2, pooled_alt_embedding_2] = generate_prompt_embedding(
                                entry[1], tokenizer, text_encoder, tokenizer_2, text_encoder_2)
                            embedding_diff = alt_embedding_1-alt_embedding_2
                            embedding += (embedding_diff*entry[2])
                            if pooled_embedding != '':
                                pooled_embedding_diff = pooled_alt_embedding_1-pooled_alt_embedding_2
                                pooled_embedding += (pooled_embedding_diff *
                                                     entry[2])
                    if text_encoder_2 == '' or neg_prompts[neg_prompt_counter][mix][0] != '':
                        display_status("UNCOND/NEG EMBEDDING:")
                        [uncond_embedding, pooled_uncond_embedding] = generate_prompt_embedding(
                            neg_prompts[neg_prompt_counter][mix], tokenizer, text_encoder, tokenizer_2, text_encoder_2)
                        # If negative prompt is assigned a strength other than 1
                        if neg_prompts[neg_prompt_counter][mix][2] != 1 and neg_prompts[neg_prompt_counter][mix][0] != '':
                            if text_encoder_2 != '':  # ForSDXL-type models just multiply
                                uncond_embedding *= neg_prompts[mix][neg_prompt_counter][2]
                            else:  # otherwise use difference from unconditional embedding
                                [ref_uncond_embedding, ref_pooled_uncond_embedding] = generate_prompt_embedding(
                                    ['', [prompt_defaults, prompt_defaults]], tokenizer, text_encoder, tokenizer_2, text_encoder_2)
                                embedding_diff = uncond_embedding-ref_uncond_embedding
                                uncond_embedding = ref_uncond_embedding + \
                                    (embedding_diff *
                                     neg_prompts[neg_prompt_counter][mix][2])
                    else:
                        display_status("Using zeroed-out UNCOND/NEG EMBEDDING")
                        uncond_embedding = torch.zeros_like(embedding)
                        pooled_uncond_embedding = torch.zeros_like(
                            pooled_embedding)
    
                    embedding = torch.cat([uncond_embedding, embedding])
    
                # If multiple guidance scales
                if tcount == start_step or len(guidance_scales) > 1:
                    # Get the guidance scale for this step
                    guidance_scale = guidance_scales[guidance_counter][mix]
                latent_model_input = torch.cat([latent]*2)
                latent_model_input = scheduler.scale_model_input(
                    latent_model_input, timestep=t)
                with torch.no_grad():
                    if text_encoder_2 == '':
                        noise_pred = unet(latent_model_input.to(
                            torch_device), t, encoder_hidden_states=embedding.to(torch_device)).sample
                    else:
                        original_size = (width, height)
                        target_size = (width, height)
                        crops_coords_top_left = (0, 0)
                        add_time_ids = torch.tensor(
                            [list(original_size + crops_coords_top_left + target_size)])
                        negative_add_time_ids = add_time_ids
                        add_time_ids = torch.cat(
                            [negative_add_time_ids, add_time_ids], dim=0).to(torch_device)
                        add_text_embeds = torch.cat(
                            [pooled_uncond_embedding, pooled_embedding], dim=0).to(torch_device)
                        added_cond_kwargs = {
                            "text_embeds": add_text_embeds, "time_ids": add_time_ids}
                        noise_pred = unet(latent_model_input.to(torch_device), t, encoder_hidden_states=embedding.to(
                            torch_device), added_cond_kwargs=added_cond_kwargs)[0]
    
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_preds.append(noise_pred_uncond+guidance_scale * \
                    (noise_pred_text - noise_pred_uncond))
                    
                last_model = model_value
                last_text_model = text_model_value
                
            noise_pred_out=torch.zeros_like(noise_preds[0])      
            for mixcounter,noise_pred_temp in enumerate(noise_preds):
                noise_pred_out+=noise_pred_temp*mixvalues[tcount][mixcounter]
                
            #noise_pred_out=noise_preds[0]
                
            latent=scheduler.step(noise_pred_out, t, latent).prev_sample   

            #step-processed latent for resubmission
            latent_input = latent

            # if last round and option is selected,
            if tcount == num_inference_steps-1 and cumulative_shift != [0, 0] and shiftback == 1:
                display_status("Re-setting latent to original boundaries....")
                v_shiftback = latent.shape[2]-cumulative_shift[0]
                h_shiftback = latent.shape[3]-cumulative_shift[1]
                while v_shiftback < 0:
                    v_shiftback += latent.shape[2]
                while h_shiftback < 0:
                    h_shiftback += latent.shape[3]
                latent = latent_shift(latent, v_shiftback, h_shiftback)

            # Iterate and/or reset counters
            model_counter += 1
            if model_counter == len(models['sequence']):
                model_counter = 0
            prompt_counter += 1
            if prompt_counter == len(prompts):
                prompt_counter = 0
            neg_prompt_counter += 1
            if neg_prompt_counter == len(neg_prompts):
                neg_prompt_counter = 0
            guidance_counter += 1
            if guidance_counter == len(guidance_scales):
                guidance_counter = 0
            contrastives_counter += 1
            if contrastives_counter == len(contrastives):
                contrastives_counter = 0
            step_latent_counter += 1
            if step_latent_counter == len(step_latent_variables):
                step_latent_counter = 0           
    display_step('')
    return latent

def stagger_generate_image_latent(latent, concat_stepstarts, concat_stepvars, h_shift_in, v_shift_in, verticals, shiftback, mixmax_list, mixvalues):
    
    # Largely duplicated from the generate_image_latent above
    # but reworked for seamless generation of longer sound-spectrographic latents
    global prompt_defaults
    global cumulative_shift
    h_shift_count=0
    v_shift_count=0
    if verticals != 1:
        piece_count = len(concat_stepvars)
        width_per_piece = int(latent.size()[3]/piece_count)
        height_per_piece = latent.size()[2]
        pieces_per_row = piece_count/verticals
        if int(pieces_per_row) != pieces_per_row:
            pieces_per_row = int(pieces_per_row)+1
        else:
            pieces_per_row = int(pieces_per_row)
        # REORGANIZE LATENT AND STORE CORNER POINTS OF PIECES
        new_latent = torch.zeros(
            1, 4, height_per_piece*verticals, width_per_piece*pieces_per_row)
        piece_coordinates = []
        for concat_latent_row in range(verticals):
            # new_latent[:,:,concat_latent_row*height_per_piece:(concat_latent_row+1)*height_per_piece,pieces_per_row*width_per_piece*concat_latent_row:pieces_per_row*width_per_piece*(concat_latent_row+1)]=latent[:,:,:,concat_latent_row*width_per_piece*pieces_per_row:(concat_latent_row+1)*width_per_piece*pieces_per_row]
            new_latent[:, :, concat_latent_row*height_per_piece:(concat_latent_row+1)*height_per_piece, :width_per_piece*pieces_per_row] = latent[:,
                                                                                                                                                  :, :, concat_latent_row*width_per_piece*pieces_per_row:(concat_latent_row+1)*width_per_piece*pieces_per_row]

            for piece in range(pieces_per_row):
                piece_coordinates.append([concat_latent_row*height_per_piece, (concat_latent_row+1)
                                         * height_per_piece, piece*width_per_piece, (piece+1)*width_per_piece])
        latent = new_latent

    # Set counting variables
    prompt_counter = 0
    neg_prompt_counter = 0
    guidance_counter = 0
    contrastives_counter = 0
    step_latent_counter = 0


    i2i = concat_stepvars[0][7]
    num_inference_steps = concat_stepvars[0][1]

    rgb = concat_stepvars[0][15]

    scheduler = concat_stepvars[0][0]
    model_ids = concat_stepvars[0][11]
    models_in_memory = concat_stepvars[0][9]
    skipstep = concat_stepvars[0][14]
    if models_in_memory == 1:
        stored_unets = {}
        stored_text_encoders = {}
        stored_tokenizers = {}
        stored_text_encoders_2 = {}
        stored_tokenizers_2 = {}
    # Take all these values from first job in stagger sequence
    # No provision is made for varying them WITHIN a stagger sequence
    # because there's no logical way to handle this circumstance

    # Creating multiple copies of scheduler (not sure how else to do this)
    scheduler_set = []
    for stagger_count in range(len(concat_stepstarts)-1):
        scheduler_set.append(copy.deepcopy(scheduler))

    if i2i[0] != '':
        strength = i2i[1]
        start_step = min(int(num_inference_steps*strength),
                         num_inference_steps)
        latent = latent.to(torch_device).float()
    elif rgb != 0:
        strength = rgb
        start_step = min(int(num_inference_steps*strength),
                         num_inference_steps)
        latent = latent.to(torch_device).float()
    else:
        start_step = 0
        latent = latent.to(torch_device).float()
        latent = latent*scheduler.init_noise_sigma

    # Cycle through inference steps

    old_model_number = -1
    old_text_model = -1
    for tcount in range(len(scheduler.timesteps)-skipstep):
        t=scheduler.timesteps[tcount]
        if tcount >= start_step:
            display_status("Inference timestep: "+str(tcount+1) +
                           "/"+str(len(scheduler.timesteps)))
            display_step(str(tcount+1) + "/"+str(len(scheduler.timesteps)))
            for stagger_count in range(len(concat_stepstarts)-1):
                noise_preds=[]
                for mix in range(mixmax_list[tcount]+1):
                    display_status("Processing staggered chunk " +
                                   str(stagger_count))
                    # PROCESS ONE CHUNK OF CONCATENATED LATENT
                    
                    def mixconvert(input_list,mix):
                        temp_list=[]
                        for step_entry in input_list:
                            try:
                                temp_list.append(step_entry[mix])
                            except:
                                temp_list.append(step_entry)
                        return temp_list
                    
                    mix_temp=mix

                    guidance_scales = mixconvert(concat_stepvars[stagger_count][2],mix_temp)
                    prompts = mixconvert(concat_stepvars[stagger_count][4],mix_temp) 
                    neg_prompts = mixconvert(concat_stepvars[stagger_count][5],mix_temp)
                    step_latent_variables = mixconvert(concat_stepvars[stagger_count][6],mix_temp)
                    contrastives = mixconvert(concat_stepvars[stagger_count][8],mix_temp)
                    # Get the model number for this stagger segment and this step
                    model_number = concat_stepvars[stagger_count][3][tcount][mix_temp]
                    text_model = concat_stepvars[stagger_count][13][tcount][mix_temp]

                    # Check if model is already loaded as the current model
                    # Load it if it isn't
                    if model_number != old_model_number:
                        if models_in_memory == 1:
                            if model_number not in stored_unets:
                                unet = UNet2DConditionModel.from_pretrained(
                                    model_ids[model_number], subfolder="unet")
                                unet = unet.to(torch_device)
                                stored_unets[model_number] = unet
                            else:
                                unet = stored_unets[model_number]
                        else:
                            display_status("Loading UNet " +
                                           str(model_number)+" into memory")
                            unet = UNet2DConditionModel.from_pretrained(
                                model_ids[model_number], subfolder="unet")
                            unet = unet.to(torch_device)
                    if text_model != old_text_model:
                        if models_in_memory == 1:
                            if text_model not in stored_text_encoders:
                                text_encoder = CLIPTextModel.from_pretrained(
                                    model_ids[text_model], subfolder="text_encoder")
                                stored_text_encoders[text_model] = text_encoder
                            else:
                                text_encoder = stored_text_encoders[text_model]
                            if text_model not in stored_tokenizers:
                                tokenizer = CLIPTokenizer.from_pretrained(
                                    model_ids[text_model], subfolder="tokenizer")
                                stored_tokenizers[text_model] = tokenizer
                            else:
                                tokenizer = stored_tokenizers[text_model]
                            try:
                                if text_model not in stored_text_encoders_2:
                                    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
                                        model_ids[text_model], subfolder="text_encoder_2")
                                    stored_text_encoders_2[text_model] = text_encoder_2
                                else:
                                    text_encoder_2 = stored_text_encoders_2[text_model]
                                if text_model not in stored_tokenizers_2:
                                    tokenizer_2 = CLIPTokenizer.from_pretrained(
                                        model_ids[text_model], subfolder="tokenizer_2")
                                    stored_tokenizers_2[text_model] = tokenizer_2
                                else:
                                    tokenizer_2 = stored_tokenizers_2[text_model]
                            except:
                                text_encoder_2 = ''
                                tokenizer_2 = ''
                        else:
                            display_status("Loading text model " +
                                           str(text_model)+" into memory")
                            text_encoder = CLIPTextModel.from_pretrained(
                                model_ids[text_model], subfolder="text_encoder")
                            #text_encoder = text_encoder.to(torch_device)
                            tokenizer = CLIPTokenizer.from_pretrained(
                                model_ids[text_model], subfolder="tokenizer")
                            try:
                                text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
                                    model_ids[text_model], subfolder="text_encoder_2")
                                tokenizer_2 = CLIPTokenizer.from_pretrained(
                                    model_ids[text_model], subfolder="tokenizer_2")
                            except:
                                tokenizer_2 = ''
                                text_encoder_2 = ''
    
                    last_prompts = []
                    if stagger_count < len(concat_stepstarts)-1:
                        if verticals == 1:
                            temp_latent = latent[:, :, :, concat_stepstarts[stagger_count]
                                :concat_stepstarts[stagger_count+1]]
                        else:
                            coord = piece_coordinates[stagger_count]
                            temp_latent = latent[:, :, coord[0]
                                :coord[1], coord[2]:coord[3]]
    
                    if tcount == start_step or len(step_latent_variables) > 1:
                        temp_latent = vary_latent(
                            temp_latent, step_latent_variables[step_latent_counter],mix)
                        height = temp_latent.shape[2]*8
                        width = temp_latent.shape[3]*8
    
                    # If multiple models or prompts
                    if tcount == start_step or (prompts[prompt_counter] != last_prompts):
                        last_prompts = prompts[prompt_counter]
                        display_status("EMBEDDING:")
                        [embedding, pooled_embedding] = generate_prompt_embedding(
                            prompts[prompt_counter], tokenizer, text_encoder, tokenizer_2, text_encoder_2)  # Get the embedding for this step
                        # CONTRASTIVE MANIPULATIONS HERE
                        if contrastives[contrastives_counter][0] != [[['', prompt_defaults], 0]]:
                            display_status(
                                "Generating single-prompt contrastives:")
                            embedding_reference = embedding  # Hold embedding stable
                            pooled_embedding_reference = pooled_embedding
                            for entry in contrastives[contrastives_counter][0]:
                                [alt_embedding, pooled_alt_embedding] = generate_prompt_embedding(
                                    entry[0], tokenizer, text_encoder, tokenizer_2, text_encoder_2)
                                # What's distinctive about 'main' embedding
                                embedding_diff = embedding_reference-alt_embedding
                                embedding_diff *= entry[1]
                                embedding += embedding_diff
                                if pooled_embedding_reference != '':
                                    pooled_embedding_diff = pooled_embedding_reference-pooled_alt_embedding
                                    pooled_embedding_diff *= entry[1]
                                    pooled_embedding += pooled_embedding_diff
                        if contrastives[contrastives_counter][1] != [[['', prompt_defaults], ['', prompt_defaults], 0]]:
                            display_status("Generating prompt-pair contrastives:")
                            for entry in contrastives[contrastives_counter][1]:
                                [alt_embedding_1, pooled_alt_embedding_1] = generate_prompt_embedding(
                                    entry[0], tokenizer, text_encoder, tokenizer_2, text_encoder_2)
                                [alt_embedding_2, pooled_alt_embedding_2] = generate_prompt_embedding(
                                    entry[1], tokenizer, text_encoder, tokenizer_2, text_encoder_2)
                                embedding_diff = alt_embedding_1-alt_embedding_2
                                embedding += (embedding_diff*entry[2])
                                if pooled_embedding != '':
                                    pooled_embedding_diff = pooled_alt_embedding_1-pooled_alt_embedding_2
                                    pooled_embedding += (pooled_embedding_diff *
                                                         entry[2])
                        if text_encoder_2 == '' or neg_prompts[neg_prompt_counter][0] != '':
                            display_status("UNCOND/NEG EMBEDDING:")
                            [uncond_embedding, pooled_uncond_embedding] = generate_prompt_embedding(
                                neg_prompts[neg_prompt_counter], tokenizer, text_encoder, tokenizer_2, text_encoder_2)
                            # If negative prompt is assigned a strength other than 1
                            if neg_prompts[neg_prompt_counter][2] != 1 and neg_prompts[neg_prompt_counter][0] != '':
                                if text_encoder_2 != '':  # ForSDXL-type models just multiply
                                    uncond_embedding *= neg_prompts[neg_prompt_counter][2]
                                else:  # otherwise use difference from unconditional embedding
                                    [ref_uncond_embedding, ref_pooled_uncond_embedding] = generate_prompt_embedding(
                                        ['', [prompt_defaults, prompt_defaults]], tokenizer, text_encoder, tokenizer_2, text_encoder_2)
                                    embedding_diff = uncond_embedding-ref_uncond_embedding
                                    uncond_embedding = ref_uncond_embedding + \
                                        (embedding_diff *
                                         neg_prompts[neg_prompt_counter][2])
                        else:
                            display_status("Using zeroed-out UNCOND/NEG EMBEDDING")
                            uncond_embedding = torch.zeros_like(embedding)
                            pooled_uncond_embedding = torch.zeros_like(
                                pooled_embedding)
                        embedding = torch.cat([uncond_embedding, embedding])
    
                    # If multiple guidance scales
                    if tcount == start_step or len(guidance_scales) > 1:
                        # Get the guidance scale for this step
                        guidance_scale = guidance_scales[guidance_counter]
                    latent_model_input = torch.cat([temp_latent]*2)
                    latent_model_input = scheduler_set[stagger_count].scale_model_input(
                        latent_model_input, timestep=t)
                    with torch.no_grad():
                        if text_encoder_2 == '':
                            noise_pred = unet(latent_model_input.to(
                                torch_device), t, encoder_hidden_states=embedding.to(torch_device)).sample
                        else:
                            original_size = (width, height)
                            target_size = (width, height)
                            crops_coords_top_left = (0, 0)
                            add_time_ids = torch.tensor(
                                [list(original_size + crops_coords_top_left + target_size)])
                            negative_add_time_ids = add_time_ids
                            add_time_ids = torch.cat(
                                [negative_add_time_ids, add_time_ids], dim=0).to(torch_device)
                            add_text_embeds = torch.cat(
                                [pooled_uncond_embedding, pooled_embedding], dim=0).to(torch_device)
    
                            added_cond_kwargs = {
                                "text_embeds": add_text_embeds, "time_ids": add_time_ids}
                            noise_pred = unet(latent_model_input.to(torch_device), t, encoder_hidden_states=embedding.to(
                                torch_device), added_cond_kwargs=added_cond_kwargs)[0]
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_preds.append(noise_pred_uncond+guidance_scale * \
                        (noise_pred_text - noise_pred_uncond))

                noise_pred_out=torch.zeros_like(noise_preds[0])      
                for mixcounter,noise_pred_temp in enumerate(noise_preds):
                    noise_pred_out+=noise_pred_temp*mixvalues[tcount][mixcounter]
        
                temp_latent = scheduler_set[stagger_count].step(
                    noise_pred_out, t, temp_latent).prev_sample

                # Substitute result of this step to the correct position concatenated whole latent
                if verticals == 1:
                    latent[:, :, :, concat_stepstarts[stagger_count]:concat_stepstarts[stagger_count+1]] = temp_latent
                else:
                    latent[:, :, coord[0]:coord[1],
                           coord[2]:coord[3]] = temp_latent
                
            # Iterate and/or reset counters
            last_prompts = prompts[prompt_counter]
            prompt_counter += 1
            if prompt_counter == len(prompts):
                prompt_counter = 0
            neg_prompt_counter += 1
            if neg_prompt_counter == len(neg_prompts):
                neg_prompt_counter = 0
            guidance_counter += 1
            if guidance_counter == len(guidance_scales):
                guidance_counter = 0
            contrastives_counter += 1
            if contrastives_counter == len(contrastives):
                contrastives_counter = 0
            step_latent_counter += 1
            if step_latent_counter == len(step_latent_variables):
                step_latent_counter = 0
    
            # AFTER PROCESSING THIS STEP, SHIFT WHOLE LATENT BY DESIGNATED AMOUNT
            # This is where the "staggering" occurs
            v_shift=v_shift_in[v_shift_count]
            h_shift=h_shift_in[h_shift_count]
            if v_shift<0:
                v_shift=latent.shape[2]+v_shift
            if h_shift<0:
                h_shift=latent.shape[3]+h_shift
            v_shift_count+=1
            if v_shift_count==len(v_shift_in):
                v_shift_count=0
            h_shift_count+=1
            if h_shift_count==len(h_shift_in):
                h_shift_count=0
            
            latent = latent_shift(latent, v_shift, h_shift)
            cumulative_shift[0] += v_shift
            cumulative_shift[1] += h_shift

    if cumulative_shift != [0, 0] and shiftback == 1:
        display_status("Re-setting latent to original boundaries....")
        v_shiftback = latent.shape[2]-cumulative_shift[0]
        h_shiftback = latent.shape[3]-cumulative_shift[1]
        while h_shiftback < 0:
            h_shiftback += latent.shape[3]
        while v_shiftback < 0:
            v_shiftback += latent.shape[2]
        latent = latent_shift(latent, v_shiftback, h_shiftback)

    # CLEAR MEMORY
    stored_unets = {}
    stored_text_encoders = {}
    stored_tokenizers = {}
    if torch_device == 'cuda':
        torch.cuda.empty_cache()
    display_step('')
    return latent

def generate_image_from_latent(vae, latent):
    with torch.no_grad():
        image = vae.decode(
            latent / vae.config.scaling_factor)[0]
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()[0]
    image = (image * 255).round().astype("uint8")
    image = Image.fromarray(image)
    return image

def prepare_model_data(model_numbers, model_ids, model_prompts, models_in_memory):
    models = {}
    if models_in_memory == 1:  # Stores models in memory to avoid repeated loading at expense of memory usage
        display_status('Loading models into memory.')
        models['sequence'] = []
        models['prompt'] = []
        model_counter = 0
        number_pool = []
        for mcount, model_number_dictionary in enumerate(model_numbers):
            sequence_temp={}
            for model_number in model_number_dictionary:
                models['prompt'].append(model_prompts[model_number_dictionary[model_number]])
                if model_number_dictionary[model_number] not in number_pool:
                    display_status('Unet sequence '+str(mcount)+': Storing UNet number ' +
                                   str(model_number_dictionary[model_number])+' in position '+str(model_counter))
                    unet = UNet2DConditionModel.from_pretrained(
                        model_ids[model_number_dictionary[model_number]], subfolder="unet")
                    unet = unet.to(torch_device)
                    number_pool.append(model_number_dictionary[model_number])
                    models[model_counter] = [unet]
                    sequence_temp[model_number]=model_counter
                    model_counter += 1
                else:
                    display_status('UNet sequence '+str(mcount)+': UNet number ' +
                                   str(model_number_dictionary[model_number])+' already stored in memory')
                    sequence_temp[model_number]=number_pool.index(model_number_dictionary[model_number])
            models['sequence'].append(sequence_temp)
    else:
        models['sequence'] = model_numbers
        models['prompt'] = []
        for model_number in model_numbers:
            models['prompt'].append(model_prompts[model_number])
    return models

def prepare_text_model_data(text_models, model_ids, models_in_memory):
    t_models = {}
    if models_in_memory == 1:  # Stores models in memory to avoid repeated loading at expense of memory usage
        text_model_pool = []
        text_model_counter = 0
        t_models['sequence'] = []
        for tcount, text_model_dictionary in enumerate(text_models):
            sequence_temp={}
            for text_model in text_model_dictionary:
                if text_model_dictionary[text_model] not in text_model_pool:
                    display_status('Text model sequence '+str(tcount)+': Storing text model number ' +
                                   str(text_model_dictionary[text_model])+' in position '+str(text_model_counter))
                    text_encoder = CLIPTextModel.from_pretrained(
                        model_ids[text_model_dictionary[text_model]], subfolder="text_encoder")
                    #text_encoder = text_encoder.to(torch_device)
                    tokenizer = CLIPTokenizer.from_pretrained(
                        model_ids[text_model_dictionary[text_model]], subfolder="tokenizer")
                    try:
                        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
                            model_ids[text_model_dictionary[text_model]], subfolder="text_encoder_2")
                        # text_encoder_2=text_encoder_2.to(torch_device) #OOM error
                        tokenizer_2 = CLIPTokenizer.from_pretrained(
                            model_ids[text_model_dictionary[text_model]], subfolder="tokenizer_2")
                        t_models[text_model_counter] = [tokenizer,
                                                        text_encoder, tokenizer_2, text_encoder_2]
                    except:
                        t_models[text_model_counter] = [tokenizer, text_encoder]
                    text_model_pool.append(text_model_dictionary[text_model])
                    sequence_temp[text_model]=text_model_counter
                    text_model_counter += 1
                else:
                    display_status('Text model sequence '+str(tcount) +
                                   ': Text model number '+str(text_model_dictionary[text_model])+' already stored in memory')
                    sequence_temp[text_model]=text_model_pool.index(text_model_dictionary[text_model])
            t_models['sequence'].append(sequence_temp)
    else:
        t_models['sequence'] = text_models
    return t_models

def spectrophone(image, filename, audio_channels, sample_rate, autype):
    max_volume = 50
    power_for_image = 0.25
    n_mels = 512
    padded_duration_ms = 400  # [ms]
    step_size_ms = 10
    mel_scale = True
    num_griffin_lim_iters = 32
    window_duration_ms = 100
    n_fft = int(padded_duration_ms / 1000.0 * sample_rate)
    hop_length = int(step_size_ms / 1000.0 * sample_rate)
    win_length = int(window_duration_ms / 1000.0 * sample_rate)
    data_set = np.array(image).astype(np.float32)
    waveform_group = {}
    for color_channel in range(audio_channels):
        # reverses direction of vertical axis, selects only first color channel
        data = data_set[::-1, :, color_channel]
        data = 255 - data
        data = data * max_volume / 255
        data = np.power(data, 1 / power_for_image)
        Sxx_torch = torch.from_numpy(data).to(torch_device)
        if mel_scale:
            mel_inv_scaler = torchaudio.transforms.InverseMelScale(
                n_mels=n_mels,
                sample_rate=sample_rate,
                f_min=0,
                f_max=10000,
                n_stft=n_fft // 2 + 1,
                norm=None,
                mel_scale="htk",
                # max_iter=max_mel_iters #Works in some versions, raises an error in others
            ).to(torch_device)
            Sxx_torch = mel_inv_scaler(Sxx_torch)

        griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            power=1.0,
            n_iter=num_griffin_lim_iters,
        ).to(torch_device)
        waveform = (griffin_lim(Sxx_torch).cpu().numpy())
        waveform = waveform/np.max(waveform)
        if audio_channels > 1:
            waveform_group[color_channel] = waveform
    if audio_channels == 1:
        audio_filename = filename+'.'+autype
        sf.write(audio_filename, waveform.T, sample_rate)
    elif audio_channels == 2:
        stereo_waveform = np.vstack((waveform_group[0], waveform_group[1]))
        audio_filename = filename+'_stereo.'+autype
        sf.write(audio_filename, stereo_waveform.T, sample_rate)
    elif audio_channels == 3:
        for channel in range(3):
            audio_filename = filename+'_'+str(channel)+'.'+autype
            sf.write(audio_filename, waveform_group[channel].T, sample_rate)


def make_spectrogram_from_audio(waveform, filename, sample_rate, normalize, rewidth):
    power_for_image = 0.25
    n_mels = 512
    padded_duration_ms = 400
    step_size_ms = 10
    window_duration_ms = 100
    n_fft = int(padded_duration_ms / 1000.0 * sample_rate)
    hop_length = int(step_size_ms / 1000.0 * sample_rate)
    win_length = int(window_duration_ms / 1000.0 * sample_rate)
    spectrogram_func = torchaudio.transforms.Spectrogram(
        n_fft=n_fft,
        power=None,
        hop_length=hop_length,
        win_length=win_length,
    )
    waveform_tensor = torch.from_numpy(
        waveform.astype(np.float32)).reshape(1, -1)
    Sxx_complex = spectrogram_func(waveform_tensor).numpy()[0]
    Sxx_mag = np.abs(Sxx_complex)
    mel_scaler = torchaudio.transforms.MelScale(
        n_mels=n_mels,
        sample_rate=sample_rate,
        f_min=0,
        f_max=10000,
        n_stft=n_fft // 2 + 1,
        norm=None,
        mel_scale="htk",
    )
    spectrogram = mel_scaler(torch.from_numpy(Sxx_mag)).numpy()
    data = np.power(spectrogram, power_for_image)
    # If normalize = 0 then divide by maximum of each clip
    if normalize == 0:
        data = data-np.min(data)
        data = data / np.max(data)
    else:
        data = data / normalize
        data[data > 1] = 1
    data = data * 255
    data = 255 - data
    image = Image.fromarray(data.astype(np.uint8))
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    #image = image.convert("RGB")
    if rewidth == 1:
        if image.width/8 != int(image.width/8):
            new_width = (int(image.width/8)+1)*8
            image = image.resize((new_width, 512))
    return image


def prep_audio_input(path, subfolder, estimatedDuration, testMargin, reverse, normalize, rewidth, defwidth, freqblur, double, imtype):
    display_status("Preparing spectrograms for "+path)
    try:
        y_in, sr = a2n.audio_from_file(path)
    except:
        y_in, sr = a2n.audio_from_file(work_dir+path)
    try:
        if reverse == 1:
            y_in = np.flipud(y_in)
        y = y_in[:, 0]
        channels = 2
    except:
        if reverse == 1:
            y_in = np.flip(y_in)
        y = y_in
        channels = 1
    if defwidth == 0:
        frac = 10
        # Low-resolution pass
        estimate20 = (estimatedDuration*sr)/frac
        testMargin20 = (testMargin*sr)/frac
        startLoopLength = int(estimate20-testMargin20)
        stopLoopLength = int(estimate20+testMargin20)
        if(len(y.shape)) == 2:
            yrectified = np.abs(np.sum(y, axis=1)/2)
        else:
            yrectified = np.abs(y)
        yrectdown = signal.resample(yrectified, int(yrectified.shape[0]/20))
        maxval = []
        for testwidth in range(startLoopLength, stopLoopLength):
            testheight = int(np.floor(yrectdown.shape[0])/testwidth)
            testshape = yrectdown[0:(testwidth*testheight)
                                  ].reshape((testheight, testwidth))
            testsum = np.sum(testshape, axis=0)/testheight
            maxval.append(np.max(testsum)-np.min(testsum[testsum > 0]))
        loresval = int((np.argmax(maxval))+startLoopLength)*frac
        maxval = []
        startLoopLength = int(loresval-frac)
        stopLoopLength = int(loresval+frac)
        for testwidth in range(startLoopLength, stopLoopLength):
            testheight = int(np.floor(yrectified.shape[0])/testwidth)
            testshape = yrectified[0:(
                testwidth*testheight)].reshape((testheight, testwidth))
            testsum = np.sum(testshape, axis=0)/testheight
            maxval.append(np.max(testsum)-np.min(testsum))
        topval = int(np.argmax(maxval)+startLoopLength)
    else:
        topval = defwidth
    if subfolder == '':
        filename_start = 'audio_'
    else:
        filename_start = subfolder+'/audio_'
    if double == 1:
        if channels == 1:
            y_in = np.pad(y_in, topval)
        else:
            y_in = np.pad(y_in, ((topval, topval), (0, 0)))
    counter = 100000
    for rowcount in range(0, y_in.shape[0], topval):
        for channel in range(channels):
            if double == 0:
                if channels == 1:
                    weftrow = y_in[rowcount:rowcount+topval]
                else:
                    weftrow = y_in[rowcount:rowcount+topval, channel]
            else:
                if channels == 1:
                    weftrow = y_in[rowcount:rowcount+(topval*2)]
                else:
                    weftrow = y_in[rowcount:rowcount+(topval*2), channel]
            if counter == 100000:
                row_width = weftrow.shape[0]
                weftrow_temp = np.zeros_like(weftrow)
            elif weftrow.shape[0] != row_width:
                weftrow_temp[:weftrow.shape[0]] = weftrow
                weftrow = weftrow_temp
            filename = filename_start+str(counter)+'.'+imtype
            image = make_spectrogram_from_audio(
                weftrow, filename, sr, normalize, rewidth)
            if freqblur != None:
                orig_imwidth = image.width
                orig_imheight = image.height
                image = image.resize(
                    (orig_imwidth, int(orig_imheight/freqblur)))
                image = image.resize((orig_imwidth, orig_imheight))
            if channel == 0:
                image_out = image
            elif channel == 1:
                image_out = Image.merge('RGB', (image_out, image, image))
        if not os.path.isdir(subfolder):
            os.makedirs(subfolder)
        image_out.convert('RGB')
        image_out.save(filename)
        counter += 1


def make_video(path, raw_job_string, job, nopt):
    img_array = []
    recreated_job_string = ''
    path = work_dir+path
    for image in os.listdir(path):
        if not image.endswith('.pt'):
            try:
                display_status("Added "+image)
                frame = cv2.imread(path+'/'+image)
                height, width, layers = frame.shape
                size = (width, height)
                img_array.append(frame)
            except:
                display_status("Skipped "+image)
        else:
            package_contents = torch.load(path+'/'+image)
            recreated_job_string += '#new '+package_contents[1]
    if nopt==0:
        recreated_job_string += '#new '+raw_job_string
        package = [img_array, recreated_job_string, job, __file__]
        torch.save(package, path+'.pt')
    out = cv2.VideoWriter(
        path+'.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    display_status('Video written.')
    
def chop_image(image_source,chop_dir,vertical_parts,horizontal_parts,vertical_height):
    display_status('Chopping up image source: '+image_source)
    # Load as PIL image, encode it, add appropriate noise level
    try:
        im = Image.open(work_dir+image_source)
    except:
        im = Image.open(image_source)
    dim01=im.size[0] #width
    dim02=im.size[1] #height
    dim02round=vertical_height*vertical_parts
    dim02part=vertical_height
    dim01ratio=dim02round/dim02 #and work out the ratio
    dim01*=dim01ratio
    dim01round=8*horizontal_parts
    run_once=0
    while dim01round<dim01 or run_once==0:
        dim01round+=8*horizontal_parts
        dim01part=dim01round/horizontal_parts
        run_once=1
    im_large=im.resize((dim01round,dim02round),Image.Resampling.LANCZOS)
    counter=0
    for pieces_down in range(vertical_parts):
        for pieces_across in range(horizontal_parts):
            dims=(int(pieces_across*dim01part),int(pieces_down*dim02part),int((pieces_across+1)*dim01part),int((pieces_down+1)*dim02part))
            im_crop=im_large.crop(dims)
            if not os.path.exists(work_dir+chop_dir):
                os.makedirs(work_dir+chop_dir)
            im_crop.save('test.tif')
            im_crop.save(work_dir+chop_dir+'/chop_'+str(100000+counter)+'.tif')
            counter+=1

# Helper function for converting comma-delimited string inputs into numerical lists
def num_parse(instring):
    out_list = []
    separate_strings = instring.split('+')
    for string in separate_strings:
        string_list = []
        string_parts = string.split(',')
        for string_part in string_parts:
            try:
                string_list.append(int(string_part))
            except:
                string_list.append(float(string_part))
        out_list.append(string_list)
    return out_list

# WAVCAT FUNCTIONS

def altGriffinLim(specgram, sample_rate, win_dur):
    def _get_complex_dtype(real_dtype: torch.dtype):
        if real_dtype == torch.double:
            return torch.cdouble
        if real_dtype == torch.float:
            return torch.cfloat
        if real_dtype == torch.half:
            return torch.complex32
    n_fft = int(400 / 1000.0 * sample_rate)
    win_length = int(win_dur / 1000.0 * sample_rate)
    hop_length = int(10 / 1000.0 * sample_rate)
    window = torch.hann_window(win_length, periodic=True).to(torch_device)
    shape = specgram.size()
    specgram = specgram.reshape([-1] + list(shape[-2:]))
    specgram = specgram.pow(1 / 1.0)
    angles = torch.full(specgram.size(), 1, dtype=_get_complex_dtype(
        specgram.dtype), device=specgram.device)
    for _ in range(32):
        inverse = torch.istft(
            specgram * angles, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, length=None
        )
        rebuilt = torch.stft(
            input=inverse,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        # Update our phase estimates
        angles = rebuilt
        angles = angles.div(angles.abs().add(1e-16))
    return angles


def spectralize(image, sample_rate, win_dur, audio_channels):
    data_set = np.array(image).astype(np.float32)
    angles = {}
    specgram = {}
    n_fft = int(400 / 1000.0 * sample_rate)
    mel_inv_scaler = torchaudio.transforms.InverseMelScale(
        n_mels=512,
        sample_rate=sample_rate,
        f_min=0,
        f_max=10000,
        n_stft=n_fft // 2 + 1,
        norm=None,
        mel_scale="htk"
    ).to(torch_device)
    for color_channel in range(audio_channels):
        # reverses direction of vertical axis, selects only designated color channel
        data = data_set[::-1, :, color_channel]
        data = 255 - data
        data = data * 50 / 255
        data = np.power(data, 1 / 0.25)
        Sxx_torch = torch.from_numpy(data).to(torch_device)
        Sxx_torch = mel_inv_scaler(Sxx_torch)
        specgram[color_channel] = Sxx_torch.to('cpu')
        angles[color_channel] = altGriffinLim(
            Sxx_torch, sample_rate, win_dur).to('cpu')
    # To clear CUDA memory
    if torch_device == 'cuda':
        Sxx_torch = torch.zeros(1)
        Sxx_torch = Sxx_torch.to('cpu')
        torch.cuda.empty_cache()
    return specgram, angles

def toWaveform(specgram, angles, sample_rate, win_dur):
    n_fft = int(400 / 1000.0 * sample_rate)
    win_length = int(win_dur / 1000.0 * sample_rate)
    hop_length = int(10 / 1000.0 * sample_rate)
    window = torch.hann_window(win_length, periodic=True).to(torch_device)
    specgram = torch.from_numpy(specgram).to(torch_device)
    angles = torch.from_numpy(angles).to(torch_device)
    shape = specgram.size()
    waveform = torch.istft(
        specgram * angles, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, length=None
    )
    waveform = waveform.reshape(shape[:-2] + waveform.shape[-1:])
    waveform = waveform.to('cpu')
    return waveform


def wavWrite(waveform, filename, sample_rate):
    waveform_adjusted = waveform/np.max(waveform)
    sf.write(filename, waveform_adjusted.T, sample_rate)


def wavcat(filepath, vae_option, chunk_size, overlap_size, overlap_type, sample_rate, win_dur, subfolder, audio_channels, autype, raw_job_string, audio_inv, folder_delete):
    job = [filepath, vae_option, chunk_size, overlap_size, overlap_type,
           sample_rate, win_dur, subfolder, audio_channels, autype]
    if filepath.startswith(work_dir):
        folder_filepath = filepath
    else:
        folder_filepath = work_dir+filepath
    vae = AutoencoderKL.from_pretrained(vae_option, subfolder="vae")
    vae = vae.to(torch_device)
    imlist = []
    for image_location in os.listdir(folder_filepath):
        if image_location.endswith('.pt'):
            imlist.append(image_location)
    counter = 0
    tiled_image = ''
    recreated_job_string = ''
    for image_location in imlist:
        try:
            package_contents = torch.load(folder_filepath+'/'+image_location)
            recreated_job_string += '#new '+str(package_contents[1])
            image_file = package_contents[0]
            if audio_inv==1:
                image_file*=-1
            if counter == 0:
                tiled_image = image_file
                counter = 1
            else:
                tiled_image = torch.cat((tiled_image, image_file), 3)
            display_status("Concatenated "+image_location)
        except:
            display_status("Failed to concatenate "+image_location)
    recreated_job_string += '#new '+raw_job_string
    display_status("Tiled image size = "+str(tiled_image.shape))
    if tiled_image != '':
        output_string = filepath+'_wavcat_'
        datestring = str(datetime.datetime.now())
        datestring = datestring.replace(":", "-")
        datestring = datestring.replace(".", "-")
        datestring = datestring.replace(" ", "-")
        output_string = output_string+'_'+datestring
        if subfolder != None:
            if not os.path.exists(subfolder):
                os.makedirs(subfolder)
            output_string = subfolder+'/'+output_string
        package = [tiled_image, recreated_job_string, job, __file__]
        torch.save(package, output_string+'.pt')
        chunk_count = int(tiled_image.shape[3]/chunk_size)
        mag_sequence = {}
        angle_sequence = {}
        for i in range(chunk_count):
            display_status("Processing chunk "+str(i+1) +
                           " of "+str(chunk_count))
            image_part = tiled_image[:, :, :,
                                     (i)*chunk_size:((i+1)*chunk_size)+overlap_size]
            image_part = generate_image_from_latent(vae, image_part)
            mags, angles = spectralize(
                image_part, sample_rate, win_dur, audio_channels)
            if i == 0:
                display_status("Getting unpadded size for cropping")
                test_mags, test_angles = spectralize(generate_image_from_latent(
                    vae, tiled_image[:, :, :, (i)*chunk_size:((i+1)*chunk_size)]), sample_rate, win_dur, 1)
                cropping_size = mags[0].shape[1]-test_mags[0].shape[1]
                cropping_size_1 = int(cropping_size/2)
                cropping_size_2 = cropping_size-cropping_size_1
                mags_forward = {}
                angles_forward = {}
                for k in range(audio_channels):
                    mags_forward[k] = mags[k][:, -cropping_size_2:]
                    angles_forward[k] = angles[k][:, :, -cropping_size_2:]
                    mag_sequence[k] = mags[k][:, :-cropping_size_2]
                    angle_sequence[k] = angles[k][:, :, :-cropping_size_2]
            else:
                mags_next = {}
                angles_next = {}
                for k in range(audio_channels):
                    mags_next[k] = mags[k][:, cropping_size_1:-cropping_size_2]
                    angles_next[k] = angles[k][:, :,
                                               cropping_size_1:-cropping_size_2]
                    lapwidth = int(mags_forward[k].shape[1])
                    difflist = []
                    for j in range(lapwidth):
                        if overlap_type == 1:
                            # find most similar column in the overlap
                            difflist.append(
                                np.sum(np.abs(mags_next[k].numpy()[:, j]-mags_forward[k].numpy()[:, j])))
                        elif overlap_type == 2:
                            # find column with mutually lowest amplitude in the overlap
                            difflist.append(
                                np.sum(np.abs(mags_next[k].numpy()[:, j]+mags_forward[k].numpy()[:, j])))
                    # get the index of that column
                    diffmin = difflist.index(min(difflist))
                    # substitute the portion of the previous segment leading up to that column
                    mags_next[k][:, :diffmin] = mags_forward[k][:, :diffmin]
                    angles_next[k][:, :,
                                   :diffmin] = angles_forward[k][:, :, :diffmin]
                    mag_sequence[k] = np.hstack(
                        (mag_sequence[k], mags_next[k]))
                    angle_sequence[k] = np.dstack(
                        (angle_sequence[k], angles_next[k]))
                    mags_forward[k] = mags[k][:, -cropping_size_2:]
                    angles_forward[k] = angles[k][:, :, -cropping_size_2:]
        display_status("Processing remainder")
        image_part = tiled_image[:, :, :, (i+1)*chunk_size:]
        tiled_image=[] #to clear memory
        if image_part.shape[3] > 0:
            image_part = generate_image_from_latent(vae, image_part)
            mags, angles = spectralize(
                image_part, sample_rate, win_dur, audio_channels)
            for k in range(audio_channels):
                mags_next[k] = mags[k][:, cropping_size_1:]
                angles_next[k] = angles[k][:, :, cropping_size_1:]
                lapwidth = int(mags_forward[k].shape[1])
                try:
                    difflist = []
                    for j in range(lapwidth):
                        if overlap_type == 1:
                            # find most similar column in the overlap
                            difflist.append(
                                np.sum(np.abs(mags_next[k].numpy()[:, j]-mags_forward[k].numpy()[:, j])))
                        elif overlap_type == 2:
                            # find column with mutually lowest amplitude in the overlap
                            difflist.append(
                                np.sum(np.abs(mags_next[k].numpy()[:, j]+mags_forward[k].numpy()[:, j])))
                    # get the index of that column
                    diffmin = difflist.index(min(difflist))
                    mags_next[k][:, :diffmin] = mags_forward[k][:, :diffmin]
                    angles_next[k][:, :,
                                   :diffmin] = angles_forward[k][:, :, :diffmin]
                    mag_sequence[k] = np.hstack(
                        (mag_sequence[k], mags_next[k]))
                    angle_sequence[k] = np.dstack(
                        (angle_sequence[k], angles_next[k]))
                except:  # if anything goes wrong with the final step
                    mag_sequence[k] = np.hstack(
                        (mag_sequence[k], mags_forward[k]))
                    angle_sequence[k] = np.dstack(
                        (angle_sequence[k], angles_forward[k]))
        display_status("Image width = "+str(mag_sequence[0].shape[1]))
        if torch_device == 'cuda':
            # To clear CUDA memory
            vae = torch.zeros(
                1).to('cpu')
            torch.cuda.empty_cache()
        try:
            display_status("Processing channel one")
            waveform = toWaveform(
                mag_sequence[0], angle_sequence[0], sample_rate, win_dur)
            display_status("WAV width = "+str(waveform.shape[0]))
            if audio_channels == 1 or audio_channels == 3:
                wavWrite(np.array(waveform), output_string +
                         '_01.'+autype, sample_rate)
            if audio_channels == 2:
                display_status("Processing channel two")
                stereo_waveform = np.vstack((waveform, toWaveform(
                    mag_sequence[1], angle_sequence[1], sample_rate, win_dur)))
                wavWrite(stereo_waveform, output_string+'.'+autype, sample_rate)
            else:
                display_status("Processing channel two")
                wavWrite(np.array(toWaveform(
                    mag_sequence[1], angle_sequence[1], sample_rate, win_dur)), output_string+'_02.'+autype, sample_rate)
                display_status("Processing channel three")
                wavWrite(np.array(toWaveform(
                    mag_sequence[2], angle_sequence[2], sample_rate, win_dur)), output_string+'_03.'+autype, sample_rate)
            display_status("Wavcat complete.")
        except Exception as e:
            display_status("Error: "+str(e))
            audio_failed=1
            parts=1
            length_to_handle=mag_sequence[0].shape[1]
            while audio_failed==1:
                parts+=1
                display_status("Failed; trying to process in "+str(parts)+" parts")
                split_length=int(length_to_handle/parts)+1
                continue_trying=1
                for part_count in range(parts):
                    if continue_trying==1:
                        display_status("Processing part "+str(part_count+1))
                        display_status("Processing channel one")
                        try:
                            startpoint=part_count*split_length
                            endpoint=(part_count+1)*split_length
                            if endpoint>length_to_handle:
                                endpoint=length_to_handle
                            waveform = toWaveform(
                                mag_sequence[0][:,startpoint:endpoint], angle_sequence[0][0,:,startpoint:endpoint], sample_rate, win_dur)
                            display_status("WAV width = "+str(waveform.shape[0]))
                            if audio_channels == 1 or audio_channels == 3:
                                wavWrite(np.array(waveform), output_string +
                                         '_01_'+str(part_count)+'.'+autype, sample_rate)
                            if audio_channels == 2:
                                display_status("Processing channel two")
                                stereo_waveform = np.vstack((waveform, toWaveform(
                                    mag_sequence[1][:,startpoint:endpoint], angle_sequence[1][0,:,startpoint:endpoint], sample_rate, win_dur)))
                                wavWrite(stereo_waveform, output_string+'_'+str(part_count)+'.'+autype, sample_rate)
                            else:
                                display_status("Processing channel two")
                                wavWrite(np.array(toWaveform(
                                    mag_sequence[1][:,startpoint:endpoint], angle_sequence[1][0,:,startpoint:endpoint], sample_rate, win_dur)), output_string+'_02_'+str(part_count)+'.'+autype, sample_rate)
                                display_status("Processing channel three")
                                wavWrite(np.array(toWaveform(
                                    mag_sequence[2][:,startpoint:endpoint], angle_sequence[2][0,:,startpoint:endpoint], sample_rate, win_dur)), output_string+'_03_'+str(part_count)+'.'+autype, sample_rate)
                            display_status("Wavcat complete.")
                            audio_failed=0
                        except Exception as e:
                            display_status("Error: "+str(e))
                            continue_trying=0
    if folder_delete==1:
        display_status("Deleting wavcat source folder.")
        shutil.rmtree(folder_filepath)        

def evalstring(job_string_input):
    while '{' in job_string_input:
        start_exp = job_string_input.find('{')
        end_exp = job_string_input.find('}')
        exp = job_string_input[start_exp+1:end_exp]
        ldict = {}
        try:
            exec('exp_eval='+exp, globals(), ldict)
            exp_eval = ldict['exp_eval']
            job_string_input = job_string_input.replace(
                '{'+exp+'}', str(exp_eval))
        except:
            job_string_input=job_string_input[:start_exp]+'«'+job_string_input[start_exp+1:]
    job_string_input=job_string_input.replace('«',"{")
    return(job_string_input)
       

def run_script():
    engage_script(0)


def parse_script():
    engage_script(1)


def resume_script():
    engage_script(2)


def engage_script(engage_mode):
    global cumulative_shift
    global continue_script
    global old_vae_number
    global prompt_defaults
    continue_script = 1
    save_script()
    if os.path.exists(path_to_resume_point) and engage_mode == 2:
        file = open(path_to_resume_point)
        resume_point = int(file.read())
        file.close()
    else:
        resume_point = 0
    models_in_memory_old = 0
    old_model_numbers = [-1]
    old_text_models = [-1]
    old_scheduler_number = [-1]

    # Read in a SyDNEy script
    with open(path_to_script, 'r') as file:
        init_job_string_input = file.read().replace('\n', ' ')
        init_job_string_input += '   '

    # Import configuration if given
    job_string_input_whole = init_job_string_input.split('#body')
    configuration = ''
    if len(job_string_input_whole) > 1:
        configuration = job_string_input_whole[0]
        job_string_input_whole = job_string_input_whole[1]
    else:
        job_string_input_whole = job_string_input_whole[0]
        try:
            with open(path_to_config, 'r') as file:
                configuration = file.read().replace('\n', '')
        except:
            display_status(
                "No sydney_config.txt file found, using default configuration.")

    cumulative_jobcount = 0
    job_string_inputs = job_string_input_whole.split('#restart')
    for job_string_input in job_string_inputs:
        if continue_script == 1:
            try:
                if configuration != '':
                    config_parts = configuration.split('#')
                    for config_part in config_parts:
                        if config_part.startswith('models'):
                            config_part = config_part[6:]
                            model_strings = config_part.split(';')
                            model_ids = []
                            model_prompts = []
                            model_counter = 0
                            for model_string in model_strings:
                                model_string = model_string.split(',')
                                if len(model_string) == 1:
                                    model_ids.append(model_string[0].strip())
                                    model_prompts.append('')
                                elif len(model_string) == 2:
                                    model_ids.append(model_string[0].strip())
                                    model_prompts.append(
                                        model_string[1].strip())
                                model_counter += 1

                # Supply placeholder variables for settings that don't otherwise need any
                    job_string_input = job_string_input.replace(
                        '$2copy', '$2copy 0')
                    job_string_input = job_string_input.replace(
                        '$2neg_copy', '$2neg_copy 0')
                    job_string_input = job_string_input.replace(
                        '$reverse', '$reverse 1')
                    job_string_input = job_string_input.replace(
                        '$keep_width', '$keep_width 1')
                    job_string_input = job_string_input.replace(
                        '$double', '$double 1')
                    job_string_input = job_string_input.replace(
                        '$nodecode', '$nodecode 1')
                    job_string_input = job_string_input.replace(
                        '$noimgsave', '$noimgsave 1')
                    job_string_input = job_string_input.replace(
                        '$outskip', '$outskip 1')
                    job_string_input = job_string_input.replace(
                        '$namestrip', '$namestrip 1')
                    job_string_input = job_string_input.replace(
                        '$nopt', '$nopt 1')
                    job_string_input = job_string_input.replace(
                        '$deldir', '$deldir 1')
                    
                # Get a manual resume point if present
                if '#resume' in job_string_input and resume_point==0:
                    resumestart = job_string_input.find('#resume')
                    resumestop = job_string_input[resumestart+8:].find(':')
                    resume_point = (job_string_input[resumestart+8:resumestart+8+resumestop])
                    resume_point=int(resume_point)
                    job_string_input.replace(job_string_input[resumestart:resumestop+1],'')
                    resumption=1

                # Unpack #scrape, #for, and #comb jobs
                while '#scrape' in job_string_input or '#for' in job_string_input or '#comb' in job_string_input:

                    # Start with a #scrape if it comes before any #for during this round
                    # but otherwise wait until a round where it does
                    scrapeloc = job_string_input.find('#scrape')
                    forloc = job_string_input.find('#for')
                    if scrapeloc != -1 and (scrapeloc < forloc or forloc == -1):
                        scrapefirst = 1
                    else:
                        scrapefirst = 0
                    if '#scrape' in job_string_input and scrapefirst == 1:
                        start_scrape = job_string_input.find('#scrape')
                        if job_string_input[start_scrape+7] == '=':
                            extension = job_string_input[start_scrape+8:].split(' ')[
                                0]
                        else:
                            extension = ''
                        if extension!='':
                            job_string_input = job_string_input.replace('='+extension,'')
                        end_scrape = job_string_input[start_scrape:].find(':')
                        before_part = job_string_input[:start_scrape]
                        after_part = job_string_input[end_scrape+start_scrape:]

                        if job_string_input[start_scrape+7:start_scrape+10] == 'dir':
                            scrape_segment = job_string_input[start_scrape +
                                                              11:end_scrape+start_scrape].split(' ', maxsplit=1)
                            scrapedir = 1
                        else:
                            scrape_segment = job_string_input[start_scrape +
                                                              8:end_scrape+start_scrape].split(' ', maxsplit=1)
                            scrapedir = 0
                        scrape_variable = scrape_segment[0].strip()
                        scrape_directory = scrape_segment[1].strip()
                        
                        try:
                            scrape_directory_temp = work_dir+scrape_directory
                            files = os.listdir(scrape_directory_temp)
                        except:
                            scrape_directory_temp = scrape_directory
                            files = os.listdir(scrape_directory_temp)
                        scrape_string = '#for '+scrape_variable+' '
                        for file in files:
                            if file.endswith(extension):
                                if scrapedir == 0 and os.path.isfile(scrape_directory_temp+'/'+file):
                                    # Should the above be limitable to certain file types? How?
                                    scrape_string += scrape_directory+'/'+file+';'
                                elif scrapedir == 1 and os.path.isdir(scrape_directory_temp+'/'+file):
                                    scrape_string += scrape_directory+'/'+file+';'
                        scrape_string = scrape_string[:-1]
                        job_string_input = before_part+scrape_string+after_part
                        job_string_input = evalstring(job_string_input)

                    # Nested here in case a #scrape loop is accessed by another #scrape loop
                    while '#comb' in job_string_input:
                        start_comb = job_string_input.find('#comb')
                        comb_variable = job_string_input[start_comb+6:].split(' ', maxsplit=1)[
                            0]
                        end_comb = job_string_input.find(
                            '#end '+comb_variable, start_comb)
                        string_to_process = job_string_input[start_comb+6:end_comb]
                        before_part = job_string_input[:start_comb]
                        after_part = job_string_input[end_comb +
                                                      6+len(comb_variable):]
                        process_string_parts = string_to_process.split(
                            ' ', maxsplit=2)
                        group_size = int(process_string_parts[1])
                        process_string_parts = process_string_parts[2].split(
                            '/', maxsplit=2)
                        if len(process_string_parts)==3: #if a punctuator is specified
                            punctuator = process_string_parts[1]
                            string_to_unpack = process_string_parts[2].split(':', maxsplit=1)[
                                0]
                            rep_part = process_string_parts[2].split(':', maxsplit=1)[
                                1]
                            elements_to_permute = string_to_unpack.split(';')
                            combination_list = comb_modify(combinations(
                                elements_to_permute, group_size))
                            values_out = []
                            for combination in combination_list:
                                value_out = str(combination[0])
                                for value in combination[1:]:
                                    value_out += punctuator+str(value)
                                values_out.append(value_out)
                            new_string = before_part
                            new_string += ' #for '
                            new_string += comb_variable+' '+values_out[0]
                            for combination in values_out[1:]:
                                new_string += ';'+combination
                            new_string += ':'+rep_part+' #end '+comb_variable+after_part
                            job_string_input = new_string.strip()
                        else:
                            string_to_unpack = process_string_parts[0].split(':', maxsplit=1)[
                                0]
                            rep_part = process_string_parts[0].split(':', maxsplit=1)[
                                1]
                            elements_to_permute = string_to_unpack.split(';')
                            combination_list = comb_modify(combinations(
                                elements_to_permute, group_size))
                            new_string = before_part
                            for combination in combination_list:
                                rep_part_temp=rep_part
                                for var_part in range(group_size):
                                    rep_part_temp=rep_part_temp.replace(comb_variable+str(var_part+1),combination[var_part])
                                new_string+=rep_part_temp
                        new_string+=after_part
                        job_string_input=new_string
                        job_string_input = evalstring(job_string_input)

                    while '#for' in job_string_input:
                        # Note, if only one option listed under #for there's an error -- does this need fixing?
                        # Unpack multiple jobs from all specified 'for' loops, if any, including nested ones
                        start_for = job_string_input.find('#for')
                        for_variable = job_string_input[start_for +
                                                        5:].split(' ', maxsplit=1)
                        end_for = job_string_input.find(
                            '#end '+for_variable[0], start_for)
                        string_to_unpack = job_string_input[start_for:end_for]
                        before_part = job_string_input[:start_for]
                        after_part = job_string_input[end_for +
                                                      5+len(for_variable[0]):]
                        string_to_unpack = string_to_unpack.split(
                            ':', maxsplit=1)
                        substitutions = string_to_unpack[0].split(';')
                        substitutions[0] = substitutions[0].replace(
                            '#for '+for_variable[0], '').strip()
                        new_string = before_part
                        for substitution in substitutions:
                            new_string += string_to_unpack[1].replace(
                                for_variable[0], substitution)
                        new_string += after_part
                        job_string_input = new_string
                        job_string_input = evalstring(job_string_input)

                while '#incr' in job_string_input:
                    start_incr = job_string_input.find('#incr')
                    incr_variable = job_string_input[start_incr +
                                                     6:].split(' ', maxsplit=1)
                    end_incr = job_string_input.find('#end '+incr_variable[0])
                    string_to_unpack = job_string_input[start_incr:end_incr]
                    before_part = job_string_input[:start_incr]
                    after_part = job_string_input[end_incr +
                                                  5+len(incr_variable[0]):] #was 5
                    string_to_unpack = string_to_unpack.split(':', maxsplit=1)
                    incr_variables = string_to_unpack[0].split(';')
                    incr_variables[0] = incr_variables[0].replace(
                        '#incr '+incr_variable[0], '').strip()
                    new_string = before_part
                    if len(incr_variables) == 1:
                        try:
                            incr_end = int(incr_variables[0])
                        except:
                            incr_end = float(incr_variables[0])
                        incr_start = 1
                        incr_step = 1
                        eval_phase = 'x'
                    elif len(incr_variables) == 2:
                        try:
                            incr_end = int(incr_variables[1])
                        except:
                            incr_end = float(incr_variables[1])
                        try:
                            incr_start = int(incr_variables[0])
                        except:
                            incr_start = float(incr_variables[0])
                        incr_step = 1
                        eval_phase = 'x'
                    elif len(incr_variables) == 3:
                        try:
                            incr_end = int(incr_variables[1])
                        except:
                            incr_end = float(incr_variables[1])
                        try:
                            incr_start = int(incr_variables[0])
                        except:
                            incr_start = float(incr_variables[0])
                        try:
                            incr_step = int(incr_variables[2])
                        except:
                            incr_step = float(incr_variables[2])
                        eval_phase = 'x'
                    elif len(incr_variables) == 4:
                        try:
                            incr_end = int(incr_variables[1])
                        except:
                            incr_end = float(incr_variables[1])
                        try:
                            incr_start = int(incr_variables[0])
                        except:
                            incr_start = float(incr_variables[0])
                        try:
                            incr_step = int(incr_variables[2])
                        except:
                            incr_step = float(incr_variables[2])
                        eval_phase = incr_variables[3]
                    incr = incr_start
                    while incr*np.sign(incr_step) <= incr_end*np.sign(incr_step):
                        ldict = {}
                        exec('incr_eval='+eval_phase.replace('x',
                             str(incr)), globals(), ldict)
                        incr_eval = ldict['incr_eval']
                        new_string += string_to_unpack[1].replace(
                            incr_variable[0], str(incr_eval))
                        incr += incr_step
                    new_string += after_part
                    job_string_input = new_string
                    job_string_input = evalstring(job_string_input)

                set_insertions=['']
                while '#set' in job_string_input:
                    # Isolate the bounded segment of text, and the segments before and after it
                    start_for = job_string_input.find('#set')
                    end_for = job_string_input.find('#unset')
                    string_to_unpack = job_string_input[start_for:end_for]
                    before_part = job_string_input[:start_for]
                    after_part = job_string_input[end_for+6:]
                    # Split the segment at the colon into:
                    # (0) the part to copy and
                    # (1) the part into which to copy at each #new
                    string_to_unpack = string_to_unpack.split(':', maxsplit=1)
                    # Split off any $contrast
                    # Remove initial #new from (1) if it's present
                    # (which would otherwise yield an extra job of (0) by itself)
                    if string_to_unpack[1].strip().startswith('#new'):
                        string_to_unpack[1] = string_to_unpack[1].strip()[4:]
                    # Change (0) to begin with #new instead of #set and to end with a space
                    string_to_unpack[0] = string_to_unpack[0].replace(
                        '#set ', '#new ')+' '
                    # If there's a $contrast segment, split it off to go to the end
                    if '$contrast ' in string_to_unpack[0]:
                        [string_to_unpack[0], contrast] = string_to_unpack[0].split(
                            '$contrast ', maxsplit=1)
                        contrast = "$contrast "+contrast
                    else:
                        contrast = ''
                    set_insertions.append(string_to_unpack[0][5:].strip())
                    set_insertions.append(contrast.strip())
                    # Add #break at the end of (0) if it contains any #steps
                    if '#step' in string_to_unpack[0]:
                        string_to_unpack[0] += ' #break ' 
                    # Create backup of current string_to_unpack[0]
                    backup_string_to_unpack = string_to_unpack[0]
                    # Append contrast segment to start of string_to_unpack[0] (to occur at end of preceding job)
                    string_to_unpack[0] = contrast+string_to_unpack[0]
                    # Replace every #new in (1) with the altered (0)
                    # Which effectively inserts (0) after each #new, minus its original #set
                    # and add a final contrast segment at the end
                    string_to_unpack[1] = string_to_unpack[1].replace(
                        '#new', string_to_unpack[0])+contrast
                    
                    # FIGURE OUT THE FOLLOWING BUT IT WORKS FOR NOW
                    # Rejoin before part, initial #set string minus contrast, #set sequence, and after part
                    new_string = before_part+backup_string_to_unpack+string_to_unpack[1]+after_part
                    job_string_input = new_string  
                    job_string_input = evalstring(job_string_input)
                      
                while '#each' in job_string_input:
                    current_variable=0
                    start_each = job_string_input.find('#each ')
                    each_variable = job_string_input[start_each +
                                                     6:].split(' ', maxsplit=1)[0]
                    end_each = job_string_input.find('#end '+each_variable[0])
                    string_to_unpack = job_string_input[start_each+6+len(each_variable):end_each]
                    before_part = job_string_input[:start_each]
                    after_part = job_string_input[end_each +
                                                  7+len(each_variable[0]):]
                    string_to_unpack = string_to_unpack.split(':', maxsplit=1)
                    each_variables = string_to_unpack[0].split(';')
                    while each_variable in string_to_unpack[1]:
                        string_to_unpack[1]=string_to_unpack[1].replace(each_variable,each_variables[current_variable].strip(),1)
                        current_variable+=1
                        if current_variable==len(each_variables):
                            current_variable=0
                    new_string = before_part+string_to_unpack[1]+after_part
                    job_string_input = new_string
                    job_string_input = evalstring(job_string_input)

                while '#copy' in job_string_input:
                    start_exp = job_string_input.find('#copy')
                    beforepart = job_string_input[:start_exp]
                    [copy_source,addendum] = job_string_input[start_exp +
                                                   6:].split('@ ', maxsplit=1)
                    try:
                        [addendum,afterpart]= addendum.split('#copy',maxsplit=1)
                        afterpart="#new #copy"+afterpart
                    except:
                        afterpart=''
                    try:
                        package = torch.load(work_dir+copy_source)
                        copy_string = package[1].strip()
                    except:
                        try:
                            package = torch.load(copy_source)
                            copy_string = package[1].strip()
                        except:
                            copy_string = '***CANCEL***'  # to stop jobs from being run
                        
                    job_string_input = job_string_input.replace(
                        '#copy '+copy_source,'',1)
                    
                    #copy_string is now the original job string from the pt
                    #addendum is now the string contents to add (up to next #copy)
                    
                    set_insertions.append(("#break "+addendum).strip()) 
                    
                    #split copy_string by #new
                    copy_string_pieces=copy_string.split("#new")
                    
                    new_job_string=beforepart
                    for copy_piece_count,copy_string_piece in enumerate(copy_string_pieces):
                        
                        if "$contrast" in copy_string_piece:
                            [copy_string_piece,copy_contrast]=copy_string_piece.split("$contrast")
                            if "$contrast" not in addendum:
                                copy_contrast=" $contrast"+copy_contrast
                            else:
                                copy_contrast=""
                        else:
                            copy_contrast=""
                        
                        if copy_piece_count==0:
                            new_job_string+=copy_string_piece+' #break '+addendum+copy_contrast 
                        else:
                            # Handle case with closing #unstagger
                            if "#unstagger" in copy_string_piece:
                                copy_string_piece_temp=copy_string_piece.replace("#unstagger","")
                                new_job_string+="#new"+copy_string_piece_temp+' #break '+addendum+" #unstagger"
                            else:
                                new_job_string+="#new"+copy_string_piece+' #break '+addendum
                        
                    job_string_input=new_job_string+afterpart
                    job_string_input = evalstring(job_string_input)


                job_string_input_pieces = []
                job_string_input_staggermode = []
                job_string_input_h_shift = []
                job_string_input_v_shift = []
                job_string_input_stagger_verticals = []
                while '#stagger' in job_string_input:
                    # Get unstaggered preceding piece
                    start_stagger = job_string_input.find('#stagger ')
                    job_string_input_pieces.append(
                        job_string_input[:start_stagger])
                    job_string_input_staggermode.append(0)
                    job_string_input_h_shift.append(0)
                    job_string_input_v_shift.append(0)
                    job_string_input_stagger_verticals.append(0)
                    # Get staggered piece
                    stagger_shift = job_string_input[start_stagger+9:].split(':')[
                        0]
                    stagger_shift_split = stagger_shift.split(';')
                    stagger_h_shift_split = stagger_shift_split[0].split(',')
                    stagger_h_shift=[]
                    for stagger_h_step in stagger_h_shift_split:
                        stagger_h_shift.append(int(stagger_h_step))
                    if len(stagger_shift_split) >= 2:
                        stagger_v_shift_split = stagger_shift_split[1].split(',')
                        stagger_v_shift=[]
                        for stagger_v_step in stagger_v_shift_split:
                            stagger_v_shift.append(int(stagger_v_step))
                    else:
                        stagger_v_shift = [0]
                    if len(stagger_shift_split) == 3:
                        job_string_input_stagger_verticals.append(int(stagger_shift_split[2]))
                    else:
                        job_string_input_stagger_verticals.append(1)
                    job_string_input_staggermode.append(1)
                    job_string_input_h_shift.append(stagger_h_shift)
                    job_string_input_v_shift.append(stagger_v_shift)
                    end_stagger = job_string_input.find('#unstagger')
                    job_string_input_pieces.append(
                        job_string_input[start_stagger:end_stagger])
                    job_string_input_pieces[-1] = job_string_input_pieces[-1].replace(
                        '#stagger '+str(stagger_shift)+':', '')
                    job_string_input = job_string_input[end_stagger+10:]
                if job_string_input.strip() != '':
                    job_string_input_pieces.append(job_string_input)
                    job_string_input_staggermode.append(0)
                    job_string_input_h_shift.append(0)
                    job_string_input_v_shift.append(0)
                    job_string_input_stagger_verticals.append(0)

                for piece_count, job_string_input_piece in enumerate(job_string_input_pieces):
                    filename_root = 'unassigned'
                    subfolder = 'unassigned'
                    if continue_script == 1:
                        try:
                            staggermode = job_string_input_staggermode[piece_count]
                            stagger_h_shift = job_string_input_h_shift[piece_count]
                            stagger_v_shift = job_string_input_v_shift[piece_count]
                            stagger_verticals = job_string_input_stagger_verticals[piece_count]
                            # Split (unpacked) individual job strings into separate items in a list
                            job_string_list = job_string_input_piece.split(
                                '#new')

                            # Parse individual job strings and convert them into an actionable data structure
                            jobs = []
                            raw_job_strings = []
                            for raw_job_string in job_string_list:
                                # First condition guards against an initial '#new' or equivalent, as well as 
                                # jobs that consist only of a #set insertion
                                if raw_job_string.strip() not in set_insertions and '***CANCEL***' not in raw_job_string:
                                    raw_job_strings.append(raw_job_string)
                                    if engage_mode == 0 or engage_mode == 2:
                                        raw_job_string = raw_job_string.replace(
                                            '$', '^$')
                                        raw_job_string = raw_job_string.replace(
                                            '#', '^#')
                                        raw_job_string = raw_job_string.split('^')[
                                            1:]
                                        job_string = []
                                        for segment in raw_job_string:
                                            if segment.startswith('$'):
                                                segment_parts = segment.split(
                                                    ' ', maxsplit=1)
                                                job_string.append(
                                                    segment_parts[0])
                                                job_string.append(
                                                    segment_parts[1].strip())
                                            else:
                                                job_string.append(segment)

                                        cell_count = 0
                                        step_count = 0
                                        loop_count = 0
                                        mix_count = 0
                                        mixmax = 0
                                        job = {}
                                        jobtemp = {}

                                        while cell_count < len(job_string)-1: #was just job_string
                                            if '#break' in job_string[cell_count]:
                                                # Use to flag end of a #step sequence applied via #set
                                                loop_count += 1
                                                step_count = 0
                                                mix_count = 0
                                                if loop_count not in jobtemp:
                                                    jobtemp[loop_count] = {}
                                                if step_count not in jobtemp[loop_count]:
                                                    jobtemp[loop_count][step_count] = {
                                                    }
                                                if mix_count not in jobtemp[loop_count][step_count]:
                                                    jobtemp[loop_count][step_count][mix_count] = {}
                                                cell_count += 1
                                            if '#step' in job_string[cell_count] or '#start' in job_string[cell_count] or '#mix' in job_string[cell_count] or '$contrast' in job_string[cell_count]:
                                                if '$contrast' not in job_string[cell_count]:
                                                    if '#start' in job_string[cell_count]:
                                                        loop_count += 1
                                                        step_count = 0
                                                        mix_count = 0
                                                    cell_count += 1
                                                if loop_count not in jobtemp:
                                                    jobtemp[loop_count] = {}
                                                if step_count not in jobtemp[loop_count]:
                                                    jobtemp[loop_count][step_count] = {
                                                    }
                                                if mix_count not in jobtemp[loop_count][step_count]:
                                                    jobtemp[loop_count][step_count][mix_count] = {}
                                                con_count = 0  # count of contrastive entries
                                                with_on = 0  # switch for turning on a 'with' entry
                                                unbroke=1
                                                while cell_count < len(job_string) and unbroke==1:
                                                    if '#' not in job_string[cell_count]:
                                                        if '$with' in job_string[cell_count]:
                                                            with_on = 1
                                                            jobtemp[loop_count][step_count][mix_count][con_count]['with'] = {
                                                            }
                                                            jobtemp[loop_count][step_count][mix_count][con_count]['with']['prompt'] = job_string[cell_count+1]
                                                            cell_count += 2
                                                        elif '$contrast' in job_string[cell_count]:
                                                            con_count += 1
                                                            jobtemp[loop_count][step_count][mix_count][con_count] = {
                                                            }
                                                            jobtemp[loop_count][step_count][mix_count][con_count]['prompt'] = job_string[cell_count+1]
                                                            cell_count += 2
                                                            with_on = 0
                                                        elif with_on == 1:
                                                            jobtemp[loop_count][step_count][mix_count][con_count]['with'][job_string[cell_count]
                                                                                                               [1:]] = job_string[cell_count+1]
                                                            cell_count += 2
                                                        elif con_count == 0:
                                                            jobtemp[loop_count][step_count][mix_count][job_string[cell_count]
                                                                                            [1:]] = job_string[cell_count+1]
                                                            cell_count += 2
                                                        else:
                                                            jobtemp[loop_count][step_count][mix_count][con_count][job_string[cell_count]
                                                                                                       [1:]] = job_string[cell_count+1]
                                                            cell_count += 2
                                                    elif '#mix' in job_string[cell_count]:
                                                        mix_count += 1
                                                        if mix_count>=mixmax:
                                                            mixmax=mix_count
                                                            jobtemp[loop_count][step_count]['mixmax']=mix_count
                                                            jobtemp[loop_count][step_count][mix_count]={}
                                                        cell_count += 1
                                                    elif '#step' in job_string[cell_count]:
                                                        cell_count += 1
                                                        step_count += 1
                                                        mix_count = 0
                                                        if step_count not in jobtemp[loop_count]:
                                                            jobtemp[loop_count][step_count] = {
                                                            }
                                                        if mix_count not in jobtemp[loop_count][step_count]:
                                                            jobtemp[loop_count][step_count][mix_count] = {}
                                                    elif '#start' in job_string[cell_count]:
                                                        cell_count += 1
                                                        loop_count += 1
                                                        step_count = 0
                                                        mix_count = 0
                                                        if loop_count not in jobtemp:
                                                            jobtemp[loop_count] = {
                                                            }
                                                        if step_count not in jobtemp[loop_count]:
                                                            jobtemp[loop_count][step_count] = {
                                                            }
                                                        if mix_count not in jobtemp[loop_count][step_count]:
                                                            jobtemp[loop_count][step_count][mix_count]={}
                                                    elif '#break' in job_string[cell_count]:
                                                        unbroke=0 #break out of loop
                                            else:
                                                job[job_string[cell_count][1:]] = job_string[cell_count+1]
                                                cell_count += 2
                                        
                                        if 'steps' not in job:
                                            job['steps'] = 20
                                        
                                        #sets up all needed steps    
                                        for step in range(int(job['steps'])):
                                            if step not in job:
                                                job[step] = {}
                                            if 'mixmax' not in job[step]:
                                                job[step]['mixmax']=0
                                           
                                        for loop in jobtemp: #goes through each loop
                                            cycler = 0 #handles steps that are present 
                                            for step in range(int(job['steps'])): #goes through steps
                                                
                                                for mix in jobtemp[loop][cycler]: #goes through mixes
                                                    if mix=='mixmax':
                                                        if jobtemp[loop][cycler]['mixmax']>job[step]['mixmax']:
                                                            job[step]['mixmax']=jobtemp[loop][cycler]['mixmax']
                                                    else:
                                                        if mix not in job[step]:
                                                            job[step][mix]={}
                                                        for entry in jobtemp[loop][cycler][mix]: #goes through entries
                                                            # cycler is the modulo step
                                                            # mix is the mix
                                                            # entry is the coordinator name
                                                            # jobtemp[loop][cycler][mix][entry] is its value
    
                                                            if entry in job[step][mix] and isinstance(entry,dict):
                                                                job[step][mix][entry].update(jobtemp[loop][cycler][mix][entry])
                                                            else:
                                                                job[step][mix][entry] = jobtemp[loop][cycler][mix][entry]
                                                cycler += 1
                                                if cycler >= len(jobtemp[loop]):
                                                    cycler = 0          
                                        # To catch any jobs without explicitly stated steps
                                        if 'steps' not in job:
                                            job['steps'] = 20
                                        for step in range(int(job['steps'])):
                                            if step not in job:
                                                job[step] = {}
                                        if job != {}:
                                            jobs.append(job)

                            if engage_mode == 1:
                                display_status('JOBS PARSED FROM SCRIPT =')
                                for jobcount, job in enumerate(raw_job_strings):
                                    display_status(
                                        'JOB '+str(jobcount)+' = '+job)
                            elif engage_mode == 0 or engage_mode == 2:
                                if engage_mode == 2:
                                    resumption = 1
                                # Otherwise, convert the parsed jobs into numerical variables to be used by the script
                                # and then (afterwards) proceed with processing
                                skipjobs = []

                                if staggermode == 1:
                                    concat_started = 0
                                    concat_stepstarts = [0]
                                    concat_stepvars = []
                                    stagger_jobs = []
                                    stagger_jobcount = []

                                for jobcount, job in enumerate(jobs):
                                    cumulative_jobcount += 1
                                    if staggermode == 1:
                                        stagger_jobs.append(job)
                                        stagger_jobcount.append(
                                            cumulative_jobcount)
                                    display_job(
                                        "JOB "+str(cumulative_jobcount)+":\n"+raw_job_strings[jobcount].strip())
                                    #status_text.delete("1.0", tk.END)
                                    # job_text.update()
                                    # status_text.update()
                                    display_status(
                                        "===================\nJOB "+str(cumulative_jobcount)+'\n===================')
                                    try:
                                        # Set non-step-specific variables
                                        if staggermode == 0 or filename_root == 'unassigned':
                                            if 'name' in job:
                                                filename_root = job['name']
                                            else:
                                                filename_root = ''

                                            if 'namestrip' in job:
                                                filename_root = filename_root.split(
                                                    '/')[-1]
                                                filename_root = filename_root[::-1]
                                                filename_root = filename_root.split('.', 1)[
                                                    1]
                                                filename_root = filename_root[::-1]
                                                filename_root = filename_root[:-27]

                                            if 'prename' in job:
                                                filename_root = job['prename'] + \
                                                    '_'+filename_root

                                            if 'postname' in job:
                                                filename_root = filename_root + \
                                                    '_'+job['postname']

                                            if 'outskip' in job:
                                                stagger_nodecode = 1
                                                stagger_noimgsave = 1
                                            else:
                                                stagger_nodecode = 0
                                                stagger_noimgsave = 0

                                            if 'shiftback' in job:
                                                stagger_shiftback = job['shiftback']
                                            else:
                                                stagger_shiftback = 1

                                        if staggermode == 0 or subfolder == 'unassigned':
                                            if 'dir' in job:
                                                if job['dir'].startswith(work_dir):
                                                    subfolder = job['dir']
                                                else:
                                                    subfolder = work_dir + \
                                                        '/'+job['dir']
                                            else:
                                                subfolder = work_dir

                                        if 'seed' in job:
                                            seed = int(float(job['seed']))
                                        else:
                                            seed = 0

                                        if 'seed2' in job:
                                            secondary_seed = int(
                                                float(job['seed2']))
                                        else:
                                            secondary_seed = None

                                        if 'height' in job:
                                            height = int(float(job['height']))
                                        else:
                                            height = 512

                                        if 'width' in job:
                                            width = int(float(job['width']))
                                        else:
                                            width = 512

                                        if 'steps' in job:
                                            num_inference_steps = int(
                                                float(job['steps']))
                                        else:
                                            num_inference_steps = 20
                                            
                                        if 'skipstep' in job:
                                            skipstep=int(job['skipstep'])
                                        else:
                                            skipstep=0

                                        if 'sched' in job:
                                            scheduler_number = int(
                                                job['sched'])
                                        else:
                                            scheduler_number = 3

                                        if 'schmod' in job:
                                            scheduler_model = int(
                                                job['schmod'])
                                        else:
                                            scheduler_model = None

                                        if 'audio' in job:
                                            audio_out = 1
                                            audio_channels = int(job['audio'])
                                        else:
                                            audio_out = 0
                                            audio_channels = 2
                                            
                                        if 'audinv' in job:
                                            audio_inv = int(job['audinv'])
                                        else:
                                            audio_inv = 0

                                        if 'nodecode' in job:
                                            nodecode = 1
                                        else:
                                            nodecode = 0

                                        if 'noimgsave' in job:
                                            noimgsave = 1
                                        else:
                                            noimgsave = 0

                                        if 'mem' in job:
                                            models_in_memory = int(job['mem'])
                                        else:
                                            models_in_memory = 1

                                        if 'samplerate' in job:
                                            sample_rate = int(
                                                job['samplerate'])
                                        else:
                                            sample_rate = 44100

                                        if 'imtype' in job:
                                            imtype = job['imtype']
                                        else:
                                            imtype = 'tif'

                                        if 'autype' in job:
                                            autype = job['autype']
                                        else:
                                            autype = 'wav'

                                        # Set noise variables
                                        arglist = ['nx', 'n+',
                                                   'ncat', 'nshift', 'npart']
                                        noise_variables = [[[1, 1, 1, 1]], [[0, 0, 0, 0]], [
                                            [0]], [[0, 0]], [[0, 0, 0, 0, 0]], [['', 0]], [[0,0,0,0,0]]]
                                        for argnum, arg in enumerate(arglist):
                                            if arg in job:
                                                noise_variables[argnum] = num_parse(
                                                    job[arg])
                                        if 'i2i' in job:
                                            i2iparse = job['i2i'].split(';')
                                            noise_variables[5][0][0] = i2iparse[0].strip(
                                            )
                                            noise_variables[5][0][1] = float(
                                                i2iparse[1])
                                            if noise_variables[5][0][1]==1:
                                                noise_variables[5][0][1]=0.99999
                                            if len(i2iparse)==3:
                                                noise_variables[5][0].append(i2iparse[2])
                                        if 'rgb' in job:
                                            rgbparse = job['rgb'].split(',')
                                            for rgbcount,rgbvalue in enumerate(rgbparse):
                                                noise_variables[6][0][rgbcount]=float(rgbvalue)
                                            if noise_variables[6][0][3]>=1:
                                                noise_variables[6][0][3]=0.9
                                            elif noise_variables[6][0][3]<0:
                                                noise_variables[6][0][3]=0

                                        # Initialize step-specific variables
                                        prompts = []
                                        contrastives = []
                                        neg_prompts = []
                                        neg_prompt_variables = []
                                        guidance_scales = []
                                        model_numbers = []
                                        step_latent_variables = []
                                        text_models = []
                                        mixvalues = []
                                        mixmax_list=[]

                                        step = 0
                                        snshift_check = 0
                                        while step == 0 or step in job:  # Cycle through individual steps in the job, or run just once if only one step
                                            # Load arguments specific to this step, which will supersede any universal ones
                                            if step in job:
                                                step_args = job[step]
                                            else:
                                                step_args = job
                                                
                                            #Populate the following with defaults to avoid later errors    
                                            if 'mixmax' not in step_args:
                                                step_args['mixmax']=0
                                            if 0 not in step_args:
                                                step_args[0]=[]

                                            # Set prompt and prompt variables
                                            model_numbers_temp={}
                                            text_models_temp={}
                                            prompts_temp={}
                                            neg_prompts_temp={}
                                            step_latent_variables_temp={}
                                            guidance_scales_temp={}
                                            contrastives_temp={}
                                            mixvalues_temp={}
                                            try:
                                                mixmax_list.append(job[step]['mixmax'])
                                            except:
                                                mixmax_list.append(0)
                                            
                                            for mix in range(step_args['mixmax']+1):                                              
                                                # Set model
                                                if 'model' in step_args[mix]:
                                                    model = num_parse(
                                                        step_args[mix]['model'])[0][0]
                                                    model_numbers_temp[mix]=model
                                                elif 'model' in step_args[0]:
                                                    model = num_parse(
                                                        step_args[0]['model'])[0][0]
                                                    model_numbers_temp[mix]=model
                                                elif 'model' in job:
                                                    model = num_parse(
                                                        job['model'])[0][0]
                                                    model_numbers_temp[mix]=model
                                                else:
                                                    model = 0
                                                    model_numbers_temp[mix]=0
                                                    
                                                if 'at' in step_args[mix]:
                                                    mixvalues_temp[mix]=num_parse(step_args[mix]['at'])[0][0]
                                                else:
                                                    mixvalues_temp[mix]=None
                                                    
                                                if 'txtmod' in step_args[mix]:
                                                    text_models_temp[mix]=num_parse(step_args[mix]['txtmod'])[0][0]
                                                elif 'txtmod' in step_args[0]:
                                                    text_models_temp[mix]=num_parse(step_args[0]['txtmod'])[0][0]
                                                
                                                elif 'txtmod' in job:
                                                    text_models_temp[mix]=num_parse(job['txtmod'])[0][0]
                                                else:
                                                    text_models_temp[mix]=model
    
                                                # Set prompt
                                                if 'prompt' in step_args[mix]:
                                                    prompt = step_args[mix]['prompt']
                                                elif 'prompt' in step_args[0]:
                                                    prompt = step_args[0]['prompt']
                                                elif 'prompt' in job:
                                                    prompt = job['prompt']
                                                else:
                                                    prompt = ''
    
                                                if '2prompt' in step_args[mix]:
                                                    prompt = [
                                                        prompt, job['2prompt']]
                                                elif '2prompt' in step_args[0]:
                                                    prompt = [
                                                        prompt, job['2prompt']]
                                                elif '2prompt' in job:
                                                    prompt = [
                                                        prompt, job['2prompt']]
    
                                                if '3prompt' in step_args[mix]:
                                                    try:
                                                        prompt.append(
                                                            job['3prompt'])
                                                    except:
                                                        prompt = [
                                                            prompt, prompt, job['3prompt']]
                                                if '3prompt' in step_args[0]:
                                                    try:
                                                        prompt.append(
                                                            job['3prompt'])
                                                    except:
                                                        prompt = [
                                                            prompt, prompt, job['3prompt']]
                                                elif '3prompt' in job:
                                                    try:
                                                        prompt.append(
                                                            job['3prompt'])
                                                    except:
                                                        prompt = [
                                                            prompt, prompt, job['3prompt']]
    
                                                # Make any specified model-specific adjustment to the prompt
                                                if model_prompts[model_numbers_temp[mix]] != '':
                                                    prompt = model_prompts[model_numbers_temp[mix]].replace(
                                                        '*', prompt)
    
                                                # Set prompt variables
                                                prompt_variables = prompt_defaults.copy()
                                                arglist = ['raw+', 'proc+', '*', 'pad+', 'dyna-pad', 'avg-pad', 'padx', 'posx',
                                                           'rawx', 'procx', 'endtok', '&', 'clipskip', 'cliprange', 'textx', 'text+', 'clipmore']
                                                for argnum, arg in enumerate(arglist):
                                                    if arg in step_args[mix]:
                                                        prompt_variables[argnum] = num_parse(
                                                            step_args[mix][arg])
                                                    elif arg in step_args[0]:
                                                        prompt_variables[argnum] = num_parse(
                                                            step_args[0][arg])
                                                    elif arg in job:
                                                        prompt_variables[argnum] = num_parse(
                                                            job[arg])
                                                if 'padfrom' in step_args[mix]:
                                                    prompt_variables[17]=step_args[mix]['padfrom']
                                                elif 'padfrom' in step_args[0]:
                                                    prompt_variables[17]=step_args[0]['padfrom']
                                                elif 'padfrom' in job:
                                                    prompt_variables[17]=job['padfrom']
                                                        
                                                # Add prompt and prompt variables for this step to the list
                                                # Adjust these to respect hierarchy?
                                                if '2copy' in step_args[mix] or '2copy' in step_args[0] or '2copy' in job:
                                                    prompt_variables2 = prompt_variables
                                                else:
                                                    prompt_variables2 = prompt_defaults.copy()
                                                    arglist2 = ['2raw+', '2proc+', '2*', '2pad+', '2dyna-pad', '2avg-pad', '2padx', '2posx',
                                                                '2rawx', '2procx', '2endtok', '2&', '2clipskip', '2cliprange', '2textx', '2text+', '2clipmore']
                                                    for argnum, arg in enumerate(arglist2):
                                                        if arg in step_args[mix]:
                                                            prompt_variables2[argnum] = num_parse(
                                                                step_args[mix][arg])
                                                        elif arg in step_args[0]:
                                                            prompt_variables2[argnum] = num_parse(
                                                                step_args[0][arg])
                                                        elif arg in job:
                                                            prompt_variables2[argnum] = num_parse(
                                                                job[arg])
                                                    if '2padfrom' in step_args[mix]:
                                                        prompt_variables2[16]=step_args[mix]['2padfrom']
                                                    elif '2padfrom' in step_args[0]:
                                                        prompt_variables2[16]=step_args[0]['2padfrom']
                                                    
                                                    elif '2padfrom' in job:
                                                        prompt_variables2[16]=job['2padfrom']
                                                prompts_temp[mix]=[prompt, [prompt_variables, prompt_variables2], 1]
    
                                                # Set negative prompt and negative prompt variables
                                                if 'neg_prompt' in step_args[mix]:
                                                    neg_prompt = step_args[mix]['neg_prompt']
                                                elif 'neg_prompt' in step_args[0]:
                                                    neg_prompt = step_args[0]['neg_prompt']
                                                elif 'neg_prompt' in job:
                                                    neg_prompt = job['neg_prompt']
                                                else:
                                                    neg_prompt = ''
                                                if '2neg_prompt' in step_args[mix]:
                                                    neg_prompt = [
                                                        neg_prompt, step_args[mix]['2neg_prompt']]
                                                elif '2neg_prompt' in step_args[0]:
                                                    neg_prompt = [
                                                        neg_prompt, step_args[mix]['2neg_prompt']]
                                                elif '2neg_prompt' in job:
                                                    neg_prompt = [
                                                        neg_prompt, job['2neg_prompt']]
                                                if '3neg_prompt' in step_args[mix]:
                                                    try:
                                                        neg_prompt.append(
                                                            step_args[mix]['3neg_prompt'])
                                                    except:
                                                        neg_prompt = [
                                                            neg_prompt, neg_prompt, step_args[mix]['3neg_prompt']]
                                                elif '3neg_prompt' in step_args[0]:
                                                    try:
                                                        neg_prompt.append(
                                                            step_args[0]['3neg_prompt'])
                                                    except:
                                                        neg_prompt = [
                                                            neg_prompt, neg_prompt, step_args[0]['3neg_prompt']]
                                                elif '3neg_prompt' in job:
                                                    try:
                                                        neg_prompt.append(
                                                            job['3neg_prompt'])
                                                    except:
                                                        neg_prompt = [
                                                            neg_prompt, neg_prompt, job['3neg_prompt']]
    
                                                neg_prompt_variables = prompt_defaults.copy()
                                                neg_arglist = ['neg_raw+', 'neg_proc+', 'neg_*', 'neg_pad+', 'neg_dyna-pad', 'neg_avg-pad', 'neg_padx', 'neg_posx',
                                                               'neg_rawx', 'neg_procx', 'neg_endtok', 'neg_&', 'neg_clipskip', 'neg_cliprange', 'neg_textx', 'neg_text+', 'neg_clipmore']
                                                for argnum, arg in enumerate(neg_arglist):
                                                    if arg in step_args[mix]:
                                                        neg_prompt_variables[argnum] = num_parse(
                                                            step_args[mix][arg])
                                                    elif arg in step_args[0]:
                                                        neg_prompt_variables[argnum] = num_parse(
                                                            step_args[0][arg])
                                                    elif arg in job:
                                                        neg_prompt_variables[argnum] = num_parse(
                                                            job[arg])
                                                if 'neg_padfrom' in step_args[mix]:
                                                    neg_prompt_variables[17]=step_args[mix]['neg_padfrom']
                                                elif 'neg_padfrom' in step_args[0]:
                                                    neg_prompt_variables[17]=step_args[0]['neg_padfrom']
                                                elif 'neg_padfrom' in job:
                                                    neg_prompt_variables[17]=job['neg_padfrom']
                                                if '2neg_copy' in step_args[mix] or '2neg_copy' in step_args[0] or '2neg_copy' in job:
                                                    neg_prompt_variables2 = neg_prompt_variables
                                                else:
                                                    neg_prompt_variables2 = prompt_defaults.copy()
                                                    neg_arglist2 = ['2neg_raw+', '2neg_proc+', '2neg_*', '2neg_pad+', '2neg_dyna-pad', '2neg_avg-pad', '2neg_padx', '2neg_posx',
                                                                    '2neg_rawx', '2neg_procx', '2neg_endtok', '2neg_&', '2neg_clipskip', '2neg_cliprange', '2neg_textx', '2neg_text+', '2neg_clipmore']
                                                    for argnum, arg in enumerate(neg_arglist2):
                                                        if arg in step_args[mix]:
                                                            neg_prompt_variables2[argnum] = num_parse(
                                                                step_args[mix][arg])
                                                        elif arg in step_args[0]:
                                                            neg_prompt_variables2[argnum] = num_parse(
                                                                step_args[0][arg])
                                                        elif arg in job:
                                                            neg_prompt_variables2[argnum] = num_parse(
                                                                job[arg])
                                                    if '2_neg_padfrom' in step_args[mix]:
                                                        neg_prompt_variables[17]=step_args[mix]['2_neg_padfrom']
                                                    elif '2_neg_padfrom' in step_args[0]:
                                                        neg_prompt_variables[17]=step_args[0]['2_neg_padfrom']
                                                    elif '2_neg_padfrom' in job:
                                                        neg_prompt_variables[17]=job['2_neg_padfrom']
                                                if 'negx' in step_args[mix]:
                                                    negx = num_parse(
                                                        step_args[mix]['negx'])[0][0]
                                                elif 'negx' in step_args[0]:
                                                    negx = num_parse(
                                                        step_args[0]['negx'])[0][0]
                                                elif 'negx' in job:
                                                    negx = num_parse(
                                                        job['negx'])[0][0]
                                                else:
                                                    negx = 1
                                                neg_prompts_temp[mix]=[neg_prompt, [neg_prompt_variables, neg_prompt_variables2], negx]
    
                                                # Set guidance scales
                                                if 'guid' in step_args[mix]:
                                                    guidance = num_parse(
                                                        step_args[mix]['guid'])[0][0]
                                                    guidance_scales_temp[mix]=guidance
                                                elif 'guid' in step_args[0]:
                                                    guidance = num_parse(
                                                        step_args[0]['guid'])[0][0]
                                                    guidance_scales_temp[mix]=guidance
                                                elif 'guid' in job:
                                                    guidance = num_parse(
                                                        job['guid'])[0][0]
                                                    guidance_scales_temp[mix]=guidance
                                                else:
                                                    guidance_scales_temp[mix]=9
    
                                                # Set step-by-step latent adjustments
                                                arglist_lat = [
                                                    'snx', 'sn+', 'sncat', 'snshift', 'snpart']
                                                step_latent_variable = [[[1, 1, 1, 1]], [
                                                    [0, 0, 0, 0]], [[0]], [[0, 0]], [[0, 0, 0, 0, 0]]]
                                                for argnum, arg in enumerate(arglist_lat):
                                                    if arg in step_args[mix]:
                                                        step_latent_variable[argnum] = num_parse(
                                                            step_args[mix][arg])
                                                    elif arg in step_args[0]:
                                                        step_latent_variable[argnum] = num_parse(
                                                            step_args[0][arg])
                                                    elif arg in job:
                                                        step_latent_variable[argnum] = num_parse(
                                                            job[arg])
                                                step_latent_variables_temp[mix]=step_latent_variable
                                                if 'snshift' in job or 'snshift' in step_args[mix] or 'snshift' in step_args[0]:
                                                    snshift_check = 1
    
                                                # Get contrastives
                                                contrastive_step = [[[['', prompt_defaults.copy()], 0]], [
                                                    [['', prompt_defaults.copy()], ['', prompt_defaults.copy()], 0]]]
                                                con_counter = 1
                                            
    
                                                while con_counter in step_args[mix]:
                                                    contrastive = step_args[mix][con_counter]
                                                    if 'prompt' in contrastive:
                                                        con_prompt = contrastive['prompt']
                                                    else:
                                                        con_prompt = ''
                                                    if '2prompt' in contrastive:
                                                        con_prompt = [
                                                            con_prompt, contrastive['2prompt']]
                                                    if '3prompt' in contrastive:
                                                        try:
                                                            con_prompt.append(
                                                                job['3prompt'])
                                                        except:
                                                            con_prompt = [
                                                                con_prompt, con_prompt, contrastive['3prompt']]
                                                    con_prompt_variables = prompt_defaults.copy()
                                                    for argnum, arg in enumerate(arglist):
                                                        if arg in contrastive:
                                                            con_prompt_variables[argnum] = num_parse(
                                                                contrastive[arg])
                                                    if '2copy' in contrastive:
                                                        con_prompt_variables2 = con_prompt_variables.copy()
                                                    else:
                                                        con_prompt_variables2 = prompt_defaults.copy()
                                                        for argnum, arg in enumerate(arglist2):
                                                            if arg in contrastive:
                                                                con_prompt_variables2[argnum] = num_parse(
                                                                    contrastive[arg])
                                                    con_prompts = [con_prompt, [
                                                        con_prompt_variables, con_prompt_variables2]]
                                                    if 'with' in contrastive:
                                                        if contrastive_step[1][0] == [['', prompt_defaults], ['', prompt_defaults], 0]:
                                                            # If the first entry is empty / default, remove it (for replacement)
                                                            contrastive_step[1] = [
                                                            ]
                                                        if 'prompt' in contrastive['with']:
                                                            with_prompt = contrastive['with']['prompt']
                                                        else:
                                                            with_prompt = ''
                                                        if '2prompt' in contrastive['with']:
                                                            with_prompt = [
                                                                with_prompt, contrastive['with']['2prompt']]
                                                        if '3prompt' in contrastive['with']:
                                                            try:
                                                                with_prompt.append(
                                                                    contrastive['with']['3prompt'])
                                                            except:
                                                                with_prompt = [
                                                                    with_prompt, with_prompt, contrastive['with']['3prompt']]
                                                        with_prompt_variables = prompt_defaults.copy()
                                                        for argnum, arg in enumerate(arglist):
                                                            if arg in contrastive['with']:
                                                                with_prompt_variables[argnum] = num_parse(
                                                                    contrastive['with'][arg])
                                                        if '2copy' in contrastive['with']:
                                                            with_prompt_variables2 = with_prompt_variables
                                                        else:
                                                            with_prompt_variables2 = prompt_defaults.copy()
                                                            for argnum, arg in enumerate(arglist2):
                                                                if arg in contrastive['with']:
                                                                    con_prompt_variables2[argnum] = num_parse(
                                                                        contrastive['with'][arg])
                                                        with_prompts = [with_prompt, [
                                                            with_prompt_variables, with_prompt_variables2]]
                                                        if 'by' in contrastive['with']:
                                                            con_by = num_parse(
                                                                contrastive['with']['by'])[0][0]
                                                        elif 'by' in contrastive:  # dispreferred notation
                                                            con_by = num_parse(
                                                                contrastive['by'])[0][0]
                                                        else:
                                                            con_by = 1
                                                        contrastive_step[1].append(
                                                            [con_prompts, with_prompts, con_by])
                                                    else:
                                                        if contrastive_step[0][0] == [['', prompt_defaults], 0]:
                                                            # If the first entry is empty / default, remove it (for replacement)
                                                            contrastive_step[0] = [
                                                            ]
                                                        if 'by' in contrastive:
                                                            con_by = num_parse(
                                                                contrastive['by'])[0][0]
                                                        else:
                                                            con_by = 1
                                                        contrastive_step[0].append(
                                                            [con_prompts, con_by])
                                                    con_counter += 1
                                                # Append this step's contrastives to the multi-step contrastive array
                                                contrastives_temp[mix]=contrastive_step
                                                
                                                if 'contracopy' in step_args[mix]:
                                                    contrastives_temp[mix]=contrastives_temp[0]
                                                    
                                                # process mixvalues_temp here so values for each step total 1
                                                
                                            model_numbers.append(model_numbers_temp)
                                            text_models.append(text_models_temp)
                                            prompts.append(prompts_temp)
                                            neg_prompts.append(neg_prompts_temp)
                                            guidance_scales.append(guidance_scales_temp)
                                            step_latent_variables.append(step_latent_variables_temp)
                                            contrastives.append(contrastives_temp)
                                            
                                            #process mixvalues_temp
                                            for mixcounter,mixquantity in enumerate(mixmax_list):
                                                if mixquantity==0:
                                                    for val in mixvalues_temp:
                                                        
                                                        if mixvalues_temp[val]==None:
                                                            mixvalues_temp[val]=1
                                                else:
                                                    mixtotal=1
                                                    gapcounter=0
                                                    for val in mixvalues_temp:
                                                        
                                                        if mixvalues_temp[val]!=None:
                                                            mixtotal-=mixvalues_temp[val]
                                                        else:
                                                            gapcounter+=1
                                                    for val in mixvalues_temp:
                                                        
                                                        if mixvalues_temp[val]==None:
                                                            mixvalues_temp[val]=mixtotal/gapcounter
                                            
                                            mixvalues.append(mixvalues_temp)

                                            step += 1
                                        if 'shiftback' in job[0]:
                                            shiftback = int(
                                                job[0]['shiftback'])
                                        elif snshift_check == 1:
                                            shiftback = 1
                                        else:
                                            shiftback = 0
                                    except Exception as e:
                                        display_status("Error: "+str(e))
                                        display_status(
                                            "Error parsing job variables for JOB "+str(cumulative_jobcount)+" -- skipping.")
                                        skipjobs.append(jobcount)

                                    if jobcount not in skipjobs and (continue_script == 1 or staggermode == 1) and cumulative_jobcount >= resume_point:
                                        try:
                                            # Intercept concatenation jobs that don't involve SD inference
                                            if 'wavcat' in job:
                                                wavcat_directory = job['wavcat']
                                                vae_option = model_ids[0]
                                                if 'chunk' in job:
                                                    chunk_size = int(
                                                        job['chunk'])
                                                else:
                                                    chunk_size = 100  # this is a better default than 50
                                                if 'overlap' in job:
                                                    overlap_size = int(
                                                        job['overlap'])
                                                else:
                                                    overlap_size = 10
                                                if 'overlap_type' in job:
                                                    overlap_type = int(
                                                        job['overlap_type'])
                                                else:
                                                    overlap_type = 1
                                                if 'windur' in job:
                                                    win_dur = int(
                                                        job['windur'])
                                                else:
                                                    win_dur = 100
                                                    
                                                if 'deldir' in job:
                                                    folder_delete = 1
                                                else:
                                                    folder_delete = 0
                                                if torch_device == 'cuda':
                                                    # To clear CUDA memory
                                                    models = torch.zeros(
                                                        1).to('cpu')
                                                    torch.cuda.empty_cache()
                                                    models_in_memory = 0
                                                    old_model_numbers = -1
                                                wavcat(wavcat_directory, vae_option, chunk_size, overlap_size, overlap_type,
                                                       sample_rate, win_dur, subfolder, audio_channels, autype, raw_job_strings[jobcount],audio_inv,folder_delete)
                                                if continue_script == 0:
                                                    display_status(
                                                        "Script stopped by request.  Click 'Resume Script' to resume with the next job number.")
                                                    resume_point = cumulative_jobcount+1
                                                    file = open(
                                                        path_to_resume_point, 'w')
                                                    file.write(
                                                        str(cumulative_jobcount+1))
                                                    file.close()
                                            elif 'wavprep' in job:
                                                wavprep_source = job['wavprep']
                                                if 'est_dur' in job:
                                                    estimatedDuration = num_parse(
                                                        job['est_dur'])[0][0]
                                                else:
                                                    estimatedDuration = 5
                                                if 'test_margin' in job:
                                                    testMargin = num_parse(
                                                        job['test_margin'])[0][0]
                                                else:
                                                    testMargin = 0.7
                                                if 'reverse' in job:
                                                    reverse = 1
                                                else:
                                                    reverse = 0
                                                if 'norm' in job:
                                                    normalize = num_parse(
                                                        job['norm'])[0][0]
                                                else:
                                                    normalize = 7.7
                                                if 'keep_width' in job:
                                                    rewidth = 0
                                                else:
                                                    rewidth = 1
                                                if 'def_width' in job:
                                                    defwidth = int(
                                                        job['def_width'])
                                                else:
                                                    defwidth = 0
                                                if 'freqblur' in job:
                                                    freqblur = num_parse(
                                                        job['freqblur'])[0][0]
                                                else:
                                                    freqblur = None
                                                if 'double' in job:
                                                    double = 1
                                                else:
                                                    double = 0
                                                prep_audio_input(wavprep_source, subfolder, estimatedDuration, testMargin,
                                                                 reverse, normalize, rewidth, defwidth, freqblur, double, imtype)
                                                if continue_script == 0:
                                                    display_status(
                                                        "Script stopped by request.  Click 'Resume Script' to resume with the next job number.")
                                                    resume_point = cumulative_jobcount+1
                                                    file = open(
                                                        path_to_resume_point, 'w')
                                                    file.write(
                                                        str(cumulative_jobcount+1))
                                                    file.close()
                                            elif 'makevid' in job:
                                                if 'nopt' in job:
                                                    nopt=1
                                                else:
                                                    nopt=0
                                                makevid = job['makevid']
                                                make_video(
                                                    makevid, raw_job_strings[jobcount], job, nopt)
                                                if continue_script == 0:
                                                    display_status(
                                                        "Script stopped by request.  Click 'Resume Script' to resume with the next job number.")
                                                    resume_point = cumulative_jobcount+1
                                                    file = open(
                                                        path_to_resume_point, 'w')
                                                    file.write(
                                                        str(cumulative_jobcount+1))
                                                    file.close()
                                            elif 'chop' in job:
                                                chop_variables=job['chop'].split(';')
                                                im_src=chop_variables[0]
                                                chop_dir=chop_variables[1]
                                                vert_parts=int(chop_variables[2])
                                                horiz_parts=int(chop_variables[3])
                                                vert_height=int(chop_variables[4])
                                                chop_image(im_src,chop_dir,vert_parts,horiz_parts,vert_height)
                                                
                                            # Handle jobs that do involve SD inference
                                            else:
                                                cumulative_shift = [0, 0]
                                                if 'vae' in job:
                                                    vae_number = int(
                                                        job['vae'])
                                                else:
                                                    # if not specified, use the VAE of the model used in the final step
                                                    vae_number = model_numbers[-1][0] #selects first

                                                if model_numbers != old_model_numbers or resumption == 1:
                                                    if staggermode == 0:  # To avoid redundant model loads
                                                        models = prepare_model_data(
                                                            model_numbers, model_ids, model_prompts, models_in_memory)
                                                    elif text_models == old_text_models:  # only if staggermode
                                                        resumption = 0
                                                else:
                                                    display_status(
                                                        'Using same UNet arrangement as last job')

                                                if text_models != old_text_models or resumption == 1:
                                                    if staggermode == 0:  # To avoid redundant model loads
                                                        t_models = prepare_text_model_data(
                                                            text_models, model_ids, models_in_memory)
                                                    resumption = 0
                                                else:
                                                    display_status(
                                                        'Using same text model arrangement as last job')

                                                if scheduler_number != old_scheduler_number:
                                                    scheduler = set_scheduler(
                                                        scheduler_number, model_numbers[0][0], model_ids, scheduler_model)
                                                if vae_number != old_vae_number:
                                                    display_status(
                                                        "Using VAE from model "+str(vae_number))
                                                    vae = AutoencoderKL.from_pretrained(
                                                        model_ids[vae_number], subfolder="vae")
                                                    vae = vae.to(torch_device)
                                                scheduler.set_timesteps(
                                                    num_inference_steps)

                                                # Generate image based on input arguments
                                                latent = generate_noise_latent(
                                                    seed, secondary_seed, height, width, noise_variables, scheduler, vae, num_inference_steps)

                                                if staggermode == 0:
                                                    latent = generate_image_latent(latent, scheduler, num_inference_steps, guidance_scales, models, prompts, neg_prompts, step_latent_variables,
                                                                                   noise_variables[5][0],noise_variables[6][0][3], contrastives, models_in_memory, models_in_memory_old, model_ids, shiftback, t_models, skipstep,mixmax_list,mixvalues)
                                                    display_status(
                                                        'Inference complete.')
                                                    filename = filename_root+'_'+get_datestring()
                                                    if subfolder != None:
                                                        if not os.path.exists(subfolder):
                                                            os.makedirs(
                                                                subfolder)
                                                        filename = subfolder+'/'+filename
                                                    display_status(
                                                        'Saving latent and metadata package.')
                                                    package = [
                                                        latent, raw_job_strings[jobcount], job, __file__]
                                                    torch.save(
                                                        package, filename+'.pt')
                                                    # Delete model data to free up memory
                                                    if nodecode == 0:
                                                        display_status(
                                                            'Generating image from latent.')
                                                        image = generate_image_from_latent(
                                                            vae, latent)
                                                        refresh_image_display(
                                                            image)
                                                        if noimgsave == 0:
                                                            display_status(
                                                                'Saving image.')
                                                            image.save(
                                                                filename+'.'+imtype)
                                                            display_caption(
                                                                filename+'.'+imtype)
                                                        else:
                                                            display_caption(
                                                                filename+'.pt')
                                                    else:
                                                        display_caption(
                                                            filename+'.pt\nNOT DISPLAYED')
                                                    if audio_out == 1:
                                                        if nodecode == 0:
                                                            display_status(
                                                                'Generating audio.')
                                                            if audio_inv==1:
                                                                image=ImageOps.invert(image)
                                                            spectrophone(
                                                                image, filename, audio_channels, sample_rate, autype)
                                                        else:
                                                            display_status(
                                                                "Can't generate audio because image latent not decoded.")
                                                elif staggermode == 1:
                                                    if concat_started == 0:
                                                        concat_latent = latent
                                                        concat_started = 1
                                                    else:
                                                        concat_latent = torch.cat(
                                                            (concat_latent, latent), dim=3)
                                                    concat_stepstarts.append(
                                                        concat_latent.shape[3])
                                                    concat_stepvars.append([scheduler, num_inference_steps, guidance_scales, model_numbers, prompts, neg_prompts, step_latent_variables,
                                                                           noise_variables[5][0], contrastives, models_in_memory, models_in_memory_old, model_ids, shiftback, text_models, skipstep, noise_variables[6][0][3]])
                                        except Exception as e:
                                            display_status("Error: "+str(e))
                                            display_status(
                                                "Error running JOB "+str(cumulative_jobcount)+" -- skipping")
                                    elif continue_script == 0 and staggermode == 0:
                                        display_status(
                                            "Script stopped by request.  Click 'Resume Script' to resume with the next job number.")
                                        resume_point = cumulative_jobcount+1
                                        file=open(path_to_resume_point,'w')
                                        file.write(str(jobcount+1))
                                        file.close()
                                        break
                                    else:
                                        display_status(
                                            "JOB "+str(cumulative_jobcount)+" skipped.")
                                    old_model_numbers = model_numbers
                                    old_text_models = text_models
                                    file=open(path_to_resume_point,'w') #<-- added in
                                    file.write(str(cumulative_jobcount+1))
                                    file.close()
                                # In stagger mode, will reach this point with concatenated latent and list of stepvars
                                if staggermode == 1:
                                    if resumption == 0:
                                        latent = stagger_generate_image_latent(
                                            concat_latent, concat_stepstarts, concat_stepvars, stagger_h_shift, stagger_v_shift, stagger_verticals, stagger_shiftback,mixmax_list,mixvalues)
                                        display_status('Inference complete.')
                                        filename = filename_root+'_'+get_datestring()
                                        if subfolder != None:
                                            if not os.path.exists(subfolder):
                                                os.makedirs(subfolder)
                                            filename = subfolder+'/'+filename
                                            raw_job_strings_recast = '#stagger ' + \
                                                str(stagger_shift)+': '
                                            for item in raw_job_strings:
                                                raw_job_strings_recast += '#new '+item.strip()+' '
                                            raw_job_strings_recast += '#unstagger'
                                            package = [
                                                latent, raw_job_strings_recast, stagger_jobs, __file__]
                                            torch.save(package, filename+'.pt')

                                        # Following mostly cut and pasted from above
                                        if stagger_nodecode == 0:
                                            display_status(
                                                'Generating image from latent.')
                                            image = generate_image_from_latent(
                                                vae, latent)
                                            refresh_image_display(image)
                                            if stagger_noimgsave == 0:
                                                display_status('Saving image.')
                                                image.save(filename+'.'+imtype)
                                                display_caption(
                                                    filename+'.'+imtype)
                                            else:
                                                display_caption(filename+'.pt')
                                        else:
                                            display_caption(
                                                filename+'.pt\nNOT DISPLAYED')
                                        if audio_out == 1:
                                            if stagger_nodecode == 0:
                                                display_status(
                                                    'Generating audio.')
                                                spectrophone(
                                                    image, filename, audio_channels, sample_rate, autype)
                                            else:
                                                display_status(
                                                    "Can't generate audio because image latent not decoded.")

                                        old_scheduler_number = scheduler_number
                                        cumulative_shift = [0, 0]
                                        models_in_memory_old = models_in_memory
                                        file = open(path_to_resume_point, 'w')
                                        file.write(str(cumulative_jobcount+1))
                                        file.close()
                                        if continue_script == 0:
                                            display_status(
                                                "Script stopped by request.  Click 'Resume Script' to resume with the next job number.")
                                            resume_point = cumulative_jobcount+1
                                            break
                        except Exception as e:
                            display_status("Error: "+str(e))
                            display_status(
                                "Error running job group -- please review.")
            except Exception as e:
                display_status("Error: "+str(e))
                if engage_mode == 1 and len(job_string_inputs) > 1:
                    display_status(
                        "Error parsing job log.  However, the script contains a #restart, so parsing later parts of the script may depend on actions taken during earlier parts.")
                else:
                    display_status("Error parsing job log -- please review.")
                engage_mode = 2


def get_datestring():
    datestring = str(datetime.datetime.now())
    datestring = datestring.replace(":", "-")
    datestring = datestring.replace(".", "-")
    datestring = datestring.replace(" ", "-")
    return datestring

def save_script():
    to_save = 1
    if os.path.exists(path_to_script):
        file = open(path_to_script, 'r')
        current_text = file.read()
        file.close()
        if current_text.strip() == script_text.get(1.0, tk.END).strip():
            display_caption('Current saved script is up to date.')
            to_save = 0
        else:
            backup_datestring=get_datestring()
            os.rename(path_to_script, path_to_backups +
                      script_backup_prefix+backup_datestring+'.txt')
            if os.path.exists(path_to_resume_point):
                os.rename(path_to_resume_point, path_to_resume_backups+backup_resume_prefix+backup_datestring+'.txt')
    if to_save == 1:
        file = open(path_to_script, 'w')
        file.write(script_text.get(1.0, tk.END).strip())
        file.close()
        if os.path.exists(path_to_resume_point):
            os.remove(path_to_resume_point)

def revert_script():
    file = open(path_to_script, 'r')
    script_text.delete(1.0, tk.END)
    script_text.insert(1.0, file.read())
    file.close()


def refresh_image_display(image):
    image = ImageTk.PhotoImage(image)
    image_label.config(image=image)
    image_label.photo_ref = image
    image_label.update()


def display_job(job):
    job_text.delete("1.0", tk.END)
    job_text.insert("1.0", job)
    job_text.update()


def display_status(status):
    status_text.insert(tk.END, status+'\n')
    status_text.update()
    status_text.see("end")
    
def display_step(step):
    steps_text.delete("1.0", tk.END)
    steps_text.insert(tk.END, step+'\n')
    steps_text.update()
    steps_text.see("end")

def display_caption(caption):
    caption_text.delete("1.0", tk.END)
    caption_text.insert("1.0", caption)
    caption_text.update()


def query_folder():
    d = fd.askdirectory()
    if d == '':
        display_caption("Cancelled folder query.")
    else:
        file = open(d+'/folder_query.txt', 'w')
        file.write('Contents of '+d+' as of '+get_datestring()+'\n\n')
        for x in os.listdir(d):
            if x.endswith('.pt'):
                file.write(x+'\n')
                display_status(x)
                try:
                    package = torch.load(d+'/'+x)
                    file.write(package[1].strip()+'\n')
                    display_status(package[1].strip())
                except Exception as e:
                    display_status(package[1].strip())
                    file.write('---\n')
                    display_status("Error: "+str(e))
                    display_status(
                        "Image query failed -- couldn't access information in SyDNEy .pt format.")
                file.write('\n')
                display_status('')
        file.close()


def query_image():
    x = fd.askopenfilename(filetypes=[("SyDNEy .pt files", "*.pt")])
    if x == '':
        display_caption("Cancelled image query.")
    else:
        try:
            package = torch.load(x)
            package[1] = str(package[1])
            display_caption(package[1].strip())
        except Exception as e:
            display_status("Error: "+str(e))
            display_caption(
                "Image query failed -- couldn't access information in SyDNEy .pt format.")


def stop_script():
    global continue_script
    continue_script = 0

def load_backup():
    if os.path.exists(path_to_backups):
        x = fd.askopenfilename(
            filetypes=[("Script .txt files", "*.txt")], initialdir=path_to_backups)
        file = open(x)
        script_text.delete(1.0, tk.END)
        script_text.insert(1.0, file.read())
        file.close()
        save_script()
        resume_filepath=path_to_resume_backups+backup_resume_prefix+x.split(script_backup_prefix)[1]
        if os.path.exists(resume_filepath):
            file = open(resume_filepath)
            resume_point = int(file.read())
            file = open(
                path_to_resume_point, 'w')
            file.write(
                str(resume_point))
            file.close()
            display_status("=====\nLOADED BACKUP\n'Resume Script' will begin with job "+str(resume_point))
        else:
            display_status("=====\nLOADED BACKUP\nNo resume point recorded.")

def list_models():
    try:
        with open(path_to_config, 'r') as file:
            configuration = file.read().replace('\n', '')
            display_status("Loading configuration from sydney_config.txt")
    except:
        display_status(
            "No sydney_config.txt file found, using default configuration.")
        configuration = ''
    if configuration != '':
        config_parts = configuration.split('#')
        for config_part in config_parts:
            if config_part.startswith('models'):
                display_status("Creating model index....")
                config_part = config_part[7:]
                model_strings = config_part.split(';')
                model_counter = 0
                for model_string in model_strings:
                    model_string = model_string.split(',')
                    if len(model_string) == 1:
                        display_status("Model "+str(model_counter) +
                                       ": "+model_string[0].strip())
                    elif len(model_string) == 2:
                        display_status("Model "+str(model_counter)+": " +
                                       model_string[0].strip()+" +prompts: "+model_string[1].strip())
                    model_counter += 1

def list_schedulers():
    scheduler_list = ['PNDMScheduler', 'DDIMScheduler', 'LMSDiscreteScheduler', 'EulerDiscreteScheduler', 'EulerAncestralDiscreteScheduler', 'DPMSolverMultistepScheduler', 'DDPMScheduler', 'KDPM2DiscreteScheduler',
                      'DPMSolverSinglestepScheduler', 'DEISMultistepScheduler', 'UniPCMultistepScheduler', 'HeunDiscreteScheduler', 'KDPM2AncestralDiscreteScheduler', 'KDPM2DiscreteScheduler', 'DPMSolverSDEScheduler']
    for sched_count, scheduler_name in enumerate(scheduler_list):
        display_status('Scheduler '+str(sched_count)+': '+scheduler_name)


def popup_input(title, text):
    temp_window = tk.Tk()
    temp_window.withdraw()
    input_text = sd.askstring(title, text, parent=temp_window)
    temp_window.destroy()
    return input_text


def add_model():
    model_to_add = popup_input(
        'Add Model', 'Enter model ID, for example: dreamlike-art/dreamlike-photoreal-2.0')
    prompt_addition = popup_input(
        'Prompt Addition', 'Optionally enter text to add to prompts, with an asterisk (*) representing the position of the prompt; for example, to add "mdjrny-v4 style" at the end of each prompt: * mdjrny-v4 style')
    if prompt_addition != '':
        model_to_add += ','+prompt_addition
    if model_to_add != '':
        configuration = ''
        new_configuration = ''
        try:
            with open(path_to_config, 'r') as file:
                configuration = file.read()
            if configuration.endswith('\n'):
                configuration = configuration[:-1]
            if configuration != '':
                config_parts = configuration.split('#')
                for config_part in config_parts:
                    if config_part.startswith('models'):
                        config_part += ';\n'+model_to_add
                    if config_part != '':
                        new_configuration += '#'+config_part
        except:
            display_status("No configuration file found; creating new one.")
        if new_configuration == '':
            new_configuration = '#models '+model_to_add
        os.rename(path_to_config, ref_dir +
                  config_backup_prefix+get_datestring()+'.txt')
        with open(path_to_config, 'w') as file:
            file.write(new_configuration)
            file.close()
    list_models()
    
def color_picker():
    from tkinter import colorchooser
    color_code=colorchooser.askcolor(title="Choose color")
    display_status("RGB: "+str(color_code[0]))
    
def rgb_to_latent(r,g,b,vae):
    img = Image.new('RGB', [8,8], 255)
    for x in range(8):
        for y in range(8):
            img.putpixel((x,y),(r,g,b)
        )
    img_encoded=to_latent(img,vae)
    return[float(img_encoded[0][0][0][0]),float(img_encoded[0][1][0][0]),float(img_encoded[0][2][0][0]),float(img_encoded[0][3][0][0])]
    
# GUI setup

root = tk.Tk()
root.title('SyDNEy: Stable Diffusion Numerical Explorer')
window_width = 1500
window_height = 625

# center screen
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
center_x = int(screen_width/2 - window_width / 2)
center_y = int(screen_height/2 - window_height / 2)
root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')

# add text of script with scrollbar and button
frame = tk.Frame(root)
frame.place(x=10, y=0)
scrollbar = tk.Scrollbar(frame)
script_text = tk.Text(frame, bg="white", fg="green",
                      height=10, width=100, yscrollcommand=scrollbar.set)
scrollbar.config(command=script_text.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
script_text.pack(side="left")
try:
    file = open(path_to_script, 'r')
    sydney_script = file.read()
    file.close()
except:
    sydney_script = 'SCRIPT WINDOW.  Compose a script here.'
script_text.insert(tk.END, sydney_script)

# create image display
image_label = tk.Label(root)
image_label.place(x=950, y=55)

# create text displays
job_text = tk.Text(root, bg="gray35", fg="white", height=5, width=100)
job_text.place(x=10, y=180)

frame2 = tk.Frame(root)
frame2.place(x=10, y=280)
scrollbar2 = tk.Scrollbar(frame2)
status_text = tk.Text(frame2, bg="black", fg="white",
                      height=20, width=100, yscrollcommand=scrollbar2.set)
scrollbar2.config(command=status_text.yview)
scrollbar2.pack(side=tk.RIGHT, fill=tk.Y)
status_text.pack(side="left")

# Report any missing dependencies
if missing_dependencies != []:
    display_status(
        "Didn't find the following libraries required for basic functionality:")
    for dependency in missing_dependencies:
        display_status(" "+dependency)
else:
    display_status("All libraries required for basic functionality found.")
if missing_dependencies_audio != []:
    display_status(
        "Didn't find the following libraries required for audio output:")
    for dependency in missing_dependencies_audio:
        display_status(" "+dependency)
else:
    display_status("All libraries required for audio output found.")
if missing_dependencies_audio != []:
    display_status(
        "Didn't find the following libraries required for video output:")
    for dependency in missing_dependencies_video:
        display_status(" "+dependency)
else:
    display_status("All libraries required for video output found.")

caption_text = tk.Text(root, bg="gray35", fg="white", height=3, width=60)
caption_text.place(x=950, y=0)

# create buttons
run_script_button = tk.Button(root, text="Run Script", command=run_script)
run_script_button.place(x=835, y=0)
save_script_button = tk.Button(root, text="Save Script", command=save_script)
save_script_button.place(x=835, y=30)
revert_script_button = tk.Button(
    root, text="Revert to Saved", command=revert_script)
revert_script_button.place(x=835, y=60)
load_backup_button = tk.Button(root, text="Load Backup", command=load_backup)
load_backup_button.place(x=835, y=90)
stop_script_button = tk.Button(root, text="Stop Script", command=stop_script)
stop_script_button.place(x=835, y=120)
query_image_button = tk.Button(root, text="Query Image", command=query_image)
query_image_button.place(x=835, y=150)
query_folder_button = tk.Button(
    root, text="Query Folder", command=query_folder)
query_folder_button.place(x=835, y=180)
list_models_button = tk.Button(root, text="List Models", command=list_models)
list_models_button.place(x=835, y=210)
add_model_button = tk.Button(root, text="Add Model", command=add_model)
add_model_button.place(x=835, y=240)
list_schedulers_button = tk.Button(
    root, text="List Schedulers", command=list_schedulers)
list_schedulers_button.place(x=835, y=270)
parse_script_button = tk.Button(
    root, text="Parse Script", command=parse_script)
parse_script_button.place(x=835, y=310)
resume_script_button = tk.Button(
    root, text="Resume Script", command=resume_script)
resume_script_button.place(x=835, y=340)
color_picker_button = tk.Button(
    root, text="Color Picker", command=color_picker)
color_picker_button.place(x=835, y=370)

steps_text = tk.Text(root, bg="gray35", fg="white", height=2, width=7,font=("Arial",15))
steps_text.place(x=835, y=410)

def launch_gui():
    tk.mainloop()
    
def main():
    launch_gui()
    
if __name__ == "__main__":
    main()
