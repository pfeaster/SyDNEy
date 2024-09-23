# We'll assume the following basic libraries are installed
import os,datetime,csv,tkinter as tk
from tkinter import simpledialog as sd,filedialog as fd
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"

# Other dependencies we'll check for and report if missing
missing_dependencies=[]
missing_dependencies_audio=[]
missing_dependencies_video=[]

# For basic functionality
try:
    from diffusers import StableDiffusionPipeline,UNet2DConditionModel,AutoencoderKL
except:
    missing_dependencies.append('diffusers')
try:
    from transformers import CLIPTextModel,CLIPTokenizer,CLIPTextModelWithProjection
except:
    missing_dependencies.append('transformers')
try:
    import torch
    torch_device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    torch.use_deterministic_algorithms(True)
except:
    missing_dependencies.append('torch')    
try:
    from torchvision import transforms as tfms
except:
    missing_dependencies.append('torchvision')
try:
    from PIL import Image,ImageTk
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
    
global prompt_defaults
global old_vae_number
global continue_script
global cumulative_shift
os.chdir(os.path.dirname(__file__))

def write_config(path_to_config):
    config_lines=['#models ',
                  'stable-diffusion-v1-5/stable-diffusion-v1-5;'
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
                  'Lykon/dreamshaper-8']
    with open(path_to_config, 'w') as file:
        for line in config_lines:
            file.write(line+'\n')

# Set up names of reference files and create reference directories
ref_dir='SyDNEy_ref'
if not os.path.exists(ref_dir):
    os.mkdir(ref_dir)
path_to_script=ref_dir+'/SyDNEy_script.txt'
path_to_config=ref_dir+'/SyDNEy_config.txt'
if not os.path.exists(path_to_config):
    write_config(path_to_config)
path_to_csv_log=ref_dir+"/SyDNEy_log.csv"
path_to_resume_point=ref_dir+"/resume_point.txt"
path_to_backups=ref_dir+"/script_backups"
path_to_scheduler_config=ref_dir+"/scheduler_config_"
if not os.path.exists(path_to_backups):
    os.mkdir(path_to_backups)
script_backup_prefix="/script_backup_"
config_backup_prefix="/config_backup_"
work_dir='SyDNEy_work/'
     
def set_scheduler(scheduler_number,first_model,model_ids,scheduler_model):
    # Save and re-use scheduler config to avoid needing to reload pipeline
    if scheduler_model==None:
        scheduler_model=first_model
    try:
        sched_config=torch.load(path_to_scheduler_config+str(scheduler_model)+'.pt')
    except:
        pipeline=StableDiffusionPipeline.from_pretrained(model_ids[scheduler_model])
        display_status("Setting scheduler "+str(scheduler_number)+" using model "+str(scheduler_model))
        torch.save(pipeline.scheduler.config,path_to_scheduler_config+str(scheduler_model)+'.pt')
        sched_config=pipeline.scheduler.config
    if scheduler_number==0:
        from diffusers import PNDMScheduler
        scheduler = PNDMScheduler.from_config(sched_config)
    elif scheduler_number==1:
        from diffusers import DDIMScheduler
        scheduler = DDIMScheduler.from_config(sched_config)
    elif scheduler_number==2:
        from diffusers import LMSDiscreteScheduler
        scheduler = LMSDiscreteScheduler.from_config(sched_config)
    elif scheduler_number==3:
        from diffusers import EulerDiscreteScheduler
        scheduler = EulerDiscreteScheduler.from_config(sched_config)
    elif scheduler_number==4:
        from diffusers import EulerAncestralDiscreteScheduler
        scheduler = EulerAncestralDiscreteScheduler.from_config(sched_config)
    elif scheduler_number==5:
        from diffusers import DPMSolverMultistepScheduler
        scheduler = DPMSolverMultistepScheduler.from_config(sched_config)
    elif scheduler_number==6:
        from diffusers import DDPMScheduler
        scheduler = DDPMScheduler.from_config(sched_config)
    elif scheduler_number==7:
        from diffusers import KDPM2DiscreteScheduler
        scheduler = KDPM2DiscreteScheduler.from_config(sched_config)
    elif scheduler_number==8:
        from diffusers import DPMSolverSinglestepScheduler
        scheduler = DPMSolverSinglestepScheduler.from_config(sched_config)
    elif scheduler_number==9:
        from diffusers import DEISMultistepScheduler
        scheduler = DEISMultistepScheduler.from_config(sched_config)
    elif scheduler_number==10:
        from diffusers import UniPCMultistepScheduler
        scheduler = UniPCMultistepScheduler.from_config(sched_config)
    elif scheduler_number==11:
        from diffusers import HeunDiscreteScheduler
        scheduler = HeunDiscreteScheduler.from_config(sched_config)
    elif scheduler_number==12:
        from diffusers import KDPM2AncestralDiscreteScheduler
        scheduler = KDPM2AncestralDiscreteScheduler.from_config(sched_config)
    elif scheduler_number==13:
        from diffusers import KDPM2DiscreteScheduler
        scheduler = KDPM2DiscreteScheduler.from_config(sched_config)
    elif scheduler_number==14:
        from diffusers import DPMSolverSDEScheduler
        scheduler = DPMSolverSDEScheduler.from_config(sched_config)
    return scheduler

def build_causal_attention_mask(bsz, seq_len, dtype):
    mask = torch.empty(bsz, seq_len, seq_len, dtype=dtype)
    mask.fill_(torch.tensor(torch.finfo(dtype).min))
    mask.triu_(1)
    mask = mask.unsqueeze(1)
    return mask

def vary_embedding(embedding,mult_prompt_variables,add_prompt_variables):
    #MULTIPLICATIONS 
    if mult_prompt_variables!=[[1,0,0,0,0]]:
        for entry in mult_prompt_variables:
            if len(entry)>=2:
                if entry[1]==-1:
                    entry[1]=embedding.shape[2]-1
            if len(entry)>=3:
                if entry[2]==-1:
                    entry[2]=embedding.shape[2]-1
            if len(entry)==5:
                embedding[0,entry[3]:entry[4]+1,entry[1]:entry[2]+1]*=entry[0]
            elif len(entry)==4:
                embedding[0,entry[3],entry[1]:entry[2]+1]*=entry[0]
            elif len(entry)==3:
                embedding[0,:,entry[1]:entry[2]+1]*=entry[0]
            elif len(entry)==2:
                embedding[0,:,entry[1]]*=entry[0]
            elif len(entry)==1:
                embedding*=entry[0]
    #ADDITIONS    
    if add_prompt_variables!=[[0,0,0,0,0]]:
        for entry in add_prompt_variables:
            if len(entry)>=2:
                if entry[1]==-1:
                    entry[1]=embedding.shape[2]-1
            if len(entry)>=3:
                if entry[2]==-1:
                    entry[2]=embedding.shape[2]-1
            if len(entry)==5:
                embedding[0,entry[3]:entry[4]+1,entry[1]:entry[2]+1]+=entry[0]
            elif len(entry)==4:
                embedding[0,entry[3],entry[1]:entry[2]+1]+=entry[0]
            elif len(entry)==3:
                embedding[0,:,entry[1]:entry[2]+1]+=entry[0]
            elif len(entry)==2:
                embedding[0,:,entry[1]]+=entry[0]
            elif len(entry)==1:
                embedding+=entry[0]
    return embedding

def build_raw_embedding(token_numbers,token_emb_layer,prompt_variables):
    raw_prompt_embedding = torch.zeros((1,77,token_emb_layer.embedding_dim))
    asterisk_counter=0 #counts asterisks encountered in prompt
    adjust_padding=0 #turned on after reaching (first) end token
    for counter in range(77):
        
        #Check for asterisk (vocabulary list item 9 or 265)
        if int(token_numbers[counter])==9 or int(token_numbers[counter])==265:
            raw_prompt_embedding[0,counter,:]=torch.zeros((1,token_emb_layer.embedding_dim))
            if prompt_variables[2]!=[[0,0,0]]:
                for entry in prompt_variables[2]:
                    if entry[2]==asterisk_counter:
                        display_status("Row "+str(counter)+": Applying "+str(entry[0:2])+" at asterisk "+str(entry[2]))
                        raw_prompt_embedding[0,counter,entry[0]]+=entry[1]
            asterisk_counter+=1 #iterate asterisk count  
        
        #Padding row adjustment    
        elif adjust_padding==1:
            if prompt_variables[6]!=[[0]]:
                raw_prompt_embedding[0,counter,:]=token_emb_layer(token_numbers[counter])*prompt_variables[6][0][0] ####
            if prompt_variables[3]!=[[0,0]]:
                for entry in prompt_variables[3]:
                    if len(entry)==1:
                        raw_prompt_embedding[0,counter,:]+=entry[0]
                    if len(entry)==2:
                        raw_prompt_embedding[0,counter,entry[1]]+=entry[0] #order inverted from earlier draft                        
                    elif len(entry)==3:
                        raw_prompt_embedding[0,counter,entry[1]:entry[2]+1]+=entry[0]
            if prompt_variables[4]!=[[0]]:
                raw_prompt_embedding[0,counter,raw_prompt_embedding_order[-order_count]]+=raw_prompt_embedding_mean_alt[raw_prompt_embedding_order[-order_count]]
                order_count+=1
            if prompt_variables[5]!=[[0]]:
                raw_prompt_embedding[0,counter,:]+=raw_prompt_embedding_mean*prompt_variables[5][0][0]
        else:
            raw_prompt_embedding[0,counter,:]=token_emb_layer(token_numbers[counter]) ####
            #Steps to take once end token is reached
            if int(token_numbers[counter])==49407:
                #if prompt_variables[3]!=[[0,0]] or prompt_variables[4]!=[[0]] or prompt_variables[5]!=[[0]] or prompt_variables[6]==[[0]] or prompt_variables[10]!=[[0]]:
                if prompt_variables[3]!=[[0,0]] or prompt_variables[4]!=[[0]] or prompt_variables[5]!=[[0]] or prompt_variables[6]!=[[1]] or prompt_variables[10]!=[[0]]:
                    endtok_current=counter
                    display_status("Row "+str(counter)+": Found first end token, will adjust rows accordingly")
                    adjust_padding=1
                #Calculate variables for dynamic padding
                if prompt_variables[4]!=[[0]] or prompt_variables[5]!=[[0]]:
                    display_status("Calculating variables for dynamic padding")
                    raw_prompt_embedding_sum=torch.sum(raw_prompt_embedding[0,1:counter-1,:],0)
                    raw_prompt_embedding_mean=raw_prompt_embedding_sum/(counter-1) #convert to mean
                    if prompt_variables[4]!=[[0]]:
                        raw_prompt_embedding_mean_alt=raw_prompt_embedding_mean*prompt_variables[4][0][0] #multiply by designated factor
                        raw_prompt_embedding_sum_abs=abs(raw_prompt_embedding_sum)
                        raw_prompt_embedding_order=raw_prompt_embedding_sum_abs.argsort()
                        order_count=1 #had been 0
    if prompt_variables[10]!=[[0]]: #move end token to new location
        endtok_target=prompt_variables[10][0][0]
        if endtok_target<0:
            endtok_target=endtok_current-endtok_target
        if endtok_target>76:
            endtok_target=76
        embedding_temp=torch.zeros_like(raw_prompt_embedding)
        embedding_temp[0,:endtok_current,:]=raw_prompt_embedding[0,:endtok_current,:]
        embedding_temp[0,endtok_current:-1,:]=raw_prompt_embedding[0,endtok_current+1,:]
        if endtok_target!=76:
            embedding_temp[0,endtok_target+1:,:]=embedding_temp[0,endtok_target:-1,:]
            embedding_temp[0,endtok_target,:]=raw_prompt_embedding[0,endtok_current,:]
        else:
            embedding_temp[0,76,:]=raw_prompt_embedding[0,endtok_current,:]
        raw_prompt_embedding=embedding_temp
    return raw_prompt_embedding

def generate_prompt_embedding(prompt,tokenizer,text_encoder,tokenizer_2,text_encoder_2):    
    prompt_text=prompt[0]
    prompt_variables=prompt[1] #consists of [[first],[second]]
    
    if not isinstance(prompt_text,list):
        prompt_text=[prompt_text,prompt_text,prompt_text]
    elif len(prompt[0])==2:
        prompt_text.append(prompt_text[1])
        
    # FIRST EMBEDDING (OR ONLY EMBEDDING)
    
    if prompt_text[0]!='~0':
        display_status("Generating embedding for prompt "+str([prompt_text[0]]))
        
        prompt_text[0]=prompt_text[0].replace('*','* ')
        prompt_text[0]=prompt_text[0].replace('  ',' ')
        
        #Translate natural-language prompt into sequence of numerical tokens
        text_inputs = tokenizer(
            prompt_text[0],
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        token_numbers = text_inputs.input_ids[0,:]
    
        #Build raw prompt embedding by looking up the numerical array for each token    
        token_emb_layer = text_encoder.text_model.embeddings.token_embedding
        raw_prompt_embedding=build_raw_embedding(token_numbers,token_emb_layer,prompt_variables[0])
        
        #Multiplications and additions to raw embedding    
        raw_prompt_embedding=vary_embedding(raw_prompt_embedding,prompt_variables[0][8],prompt_variables[0][0])
            
        #Add positional embedding to raw prompt embedding to create combined raw embedding
        if prompt_variables[0][7]!=[[0]]:
            pos_emb_layer = text_encoder.text_model.embeddings.position_embedding
            position_ids = text_encoder.text_model.embeddings.position_ids[:, :77]
            position_embedding = pos_emb_layer(position_ids)
            final_raw_embedding=raw_prompt_embedding+(position_embedding*prompt_variables[0][7][0][0])
        else:
            final_raw_embedding=raw_prompt_embedding
            
        final_raw_embedding=final_raw_embedding
        
        #Generate processed embedding from combined raw embedding
        bsz, seq_len = final_raw_embedding.shape[:2]
        causal_attention_mask = build_causal_attention_mask(bsz, seq_len, dtype=final_raw_embedding.dtype)
        encoder_outputs = text_encoder.text_model.encoder(
            inputs_embeds=final_raw_embedding,
            attention_mask=None,
            causal_attention_mask=causal_attention_mask,
            output_attentions=None,
            output_hidden_states=True,
            return_dict=None,
        )
        
        if tokenizer_2=='': #for non-XL models
            # Get output hidden state
            output = encoder_outputs[0]
            # Pass through final layer norm
            processed_embedding = text_encoder.text_model.final_layer_norm(output)
        else: #for XL models
            processed_embedding=encoder_outputs.hidden_states[-2]
            
        # Multiplications and additions to processed embedding    
        processed_embedding=vary_embedding(processed_embedding,prompt_variables[0][9],prompt_variables[0][1])
    else:
        display_status('Primary embedding zeroed out.')
        processed_embedding=torch.zeros(1,77,text_encoder.config.projection_dim) #get the actual dimension
    if tokenizer_2!='':
        prompt_embeds_list=[processed_embedding]
        
        #SECOND EMBEDDING FOR XL MODELS
        if prompt_text[1]!='~0':
            if prompt_text[1]=='~':
                prompt_text[1]=''
            display_status("Generating second embedding for prompt "+str([prompt_text[1]]))
            prompt_text[1]=prompt_text[1].replace('*','* ')
            prompt_text[1]=prompt_text[1].replace('  ',' ')
            text_inputs = tokenizer_2(
                prompt_text[1],
                padding="max_length",
                max_length=tokenizer_2.model_max_length,
                truncation=True,
                return_tensors="pt"
            )
            text_input_ids = text_inputs.input_ids
            token_numbers=text_inputs.input_ids[0,:]            
            token_emb_layer=text_encoder_2.text_model.embeddings.token_embedding
            raw_prompt_embedding=build_raw_embedding(token_numbers,token_emb_layer,prompt_variables[1])
            raw_prompt_embedding=vary_embedding(raw_prompt_embedding,prompt_variables[1][8],prompt_variables[1][0])
            if prompt_variables[1][7]!=[[0]]:
                pos_emb_layer = text_encoder_2.text_model.embeddings.position_embedding
                position_ids = text_encoder_2.text_model.embeddings.position_ids[:, :77]
                position_embedding = pos_emb_layer(position_ids)
                final_raw_embedding=raw_prompt_embedding+(position_embedding*prompt_variables[1][7][0][0])
            else:
                final_raw_embedding=raw_prompt_embedding
            
            bsz, seq_len=final_raw_embedding.shape[:2]
            causal_attention_mask=build_causal_attention_mask(bsz, seq_len, dtype=final_raw_embedding.dtype)
            encoder_outputs = text_encoder_2.text_model.encoder(
                inputs_embeds=final_raw_embedding,
                attention_mask=None,
                causal_attention_mask=causal_attention_mask,
                output_attentions=None,
                output_hidden_states=True,
                return_dict=None,
            )
            processed_embedding=encoder_outputs.hidden_states[-2]
            processed_embedding=vary_embedding(processed_embedding,prompt_variables[1][9],prompt_variables[1][1])
        else: #zero out the embedding
            display_status("Second embedding zeroed out.")
            processed_embedding=torch.zeros(1,77,text_encoder_2.config.projection_dim)
        prompt_embeds_list.append(processed_embedding)
        processed_embedding=torch.concat(prompt_embeds_list,dim=-1)
        
        #THIRD EMBEDDING FOR XL MODELS
        if prompt_text[2]!='~0':
            if prompt_text[2]=='~':
                prompt_text[2]=''
            display_status("Generating third embedding for prompt "+str([prompt_text[2]]))
            text_inputs = tokenizer_2(
                prompt_text[2],
                padding="max_length",
                max_length=tokenizer_2.model_max_length,
                truncation=True,
                return_tensors="pt"
            )
            #NONVARIABLE PLACEHOLDER UNTIL I CAN FIGURE THIS OUT
            text_input_ids = text_inputs.input_ids
            prompt_embeds_2 = text_encoder_2(
                text_input_ids.to("cpu"),
                output_hidden_states=True
            )
            pooled_embedding=prompt_embeds_2[0]
        else: #zero out the embedding
            display_status("Third embedding zeroed out.")
            pooled_embedding=torch.zeros(1,text_encoder_2.config.projection_dim)
    else:
        pooled_embedding=''        
    return [processed_embedding,pooled_embedding]

def latent_shift(latent,v_shift,h_shift):
    latent_temp=torch.zeros_like(latent)
    latent_temp[:,:,:latent.shape[2]-v_shift,:]=latent[:,:,v_shift:,:]   
    latent_temp[:,:,latent.shape[2]-v_shift:,:]=latent[:,:,:v_shift,:]
    latent=latent_temp
    latent_temp=torch.zeros_like(latent)
    latent_temp[:,:,:,:latent.shape[3]-h_shift]=latent[:,:,:,h_shift:]
    latent_temp[:,:,:,latent.shape[3]-h_shift:]=latent[:,:,:,:h_shift]
    return latent_temp

def get_white_latent(latent):
    if latent.ndim==4:
        white_latent=torch.zeros_like(latent)
        white_latent[:,0,:,:]+=0.238
        white_latent[:,1,:,:]+=0.156
        white_latent[:,3,:,:]+=-0.126
    elif latent.ndim==3:
        white_latent=torch.zeros_like(latent)
        white_latent[0,:,:]+=0.238
        white_latent[1,:,:]+=0.156
        white_latent[3,:,:]+=-0.126
    return white_latent

def vary_latent(latent,noise_variables):
    global cumulative_shift
    if noise_variables!=[[[1,1,1,1]],[[0,0,0,0]],[[0]],[[0,0]],[[0,0,0,0,0]]]:
        if noise_variables[0]!=[[1,1,1,1]]:
            if len(noise_variables[0][0])==1:
                latent*=noise_variables[0][0][0]
            else:
                for channel in range(4):
                    latent[:,channel,:,:]*=noise_variables[0][0][channel]
        if noise_variables[1]!=[[0,0,0,0]]:
            if len(noise_variables[1][0])==1:
                latent+=noise_variables[1][0][0]
            else:
                for channel in range(4):
                    latent[:,channel,:,:]+=noise_variables[1][0][channel]
        latent_to_add=latent #to hold stable through the following processes
        for alteration in noise_variables[2]:
            if alteration==[1]: #2x width
                latent=latent.tile((1,2))
            elif alteration==[2]: #2x height
                latent=latent.tile((2,1))
            elif alteration==[3]: #flip LR
                latent=latent.flip((3,0))
            elif alteration==[4]: #flip UD
                latent=latent.flip((2,0))
            elif alteration==[5]: #flip LR and UD
                latent=latent.flip((3,0))
                latent=latent.flip((2,0)) #can these be combined?
            elif alteration==[6]: #2x = RIGHT FLIPPED LR
                latent_flip=latent.flip((3,0))
                latent=torch.cat((latent,latent_flip),dim=3)
            elif alteration==[7]: #2x = BOTTOM FLIPPED LR
                latent_flip=latent.flip((3,0))
                latent=torch.cat((latent,latent_flip),dim=2)
            elif alteration==[8]: #2x = RIGHT FLIPPED UD
                latent_flip=latent.flip((2,0))
                latent=torch.cat((latent,latent_flip),dim=3)
            elif alteration==[9]: #2x = BOTTOM FLIPPED UD
                latent_flip=latent.flip((2,0))
                latent=torch.cat((latent,latent_flip),dim=2)
            elif alteration==[10]: #2x = BOTTOM FLIPPED LR AND UD
                latent_flip=latent.flip((2,0))
                latent_flip=latent_flip.flip((3,0))
                latent=torch.cat((latent,latent_flip),dim=3)    
            elif alteration==[11]: #2x = RIGHT FLIPPED UD AND LR
                latent_flip=latent.flip((2,0))
                latent_flip=latent_flip.flip((3,0))
                latent=torch.cat((latent,latent_flip),dim=2)
            elif alteration==[12]: #ROTATE CLOCKWISE
                latent=torch.rot90(latent,1,(3,2))
            elif alteration==[13]: #ROTATE COUNTERCLOCKWISE
                latent=torch.rot90(latent,1,(2,3))
            elif alteration==[14]: #INVERT RIGHT HALF
                latent_half_width=int(latent.shape[3]/2)
                latent[:,:,:,latent_half_width:]=latent[:,:,:,latent_half_width:]*-1
            elif alteration==[15]: #INVERT BOTTOM HALF
                latent_half_height=int(latent.shape[2]/2)
                latent[:,:,latent_half_height:,:]=latent[:,:,latent_half_height:,:]*-1
            elif alteration==[16]: #CROP TO MIDDLE HALF
                latent_third_width=int(latent.shape[3]/3)
                latent_half_width=int(latent.shape[3]/2)
                latent=latent[:,:,:,latent_third_width:latent_third_width+latent_half_width+1]
            elif alteration==[17]: #CROP TO LEFT HALF
                latent_half_width=int(latent.shape[3]/2)
                latent=latent[:,:,:,:latent_half_width]
            elif alteration==[18]: #CROP TO TOP HALF
                latent_half_height=int(latent.shape[2]/2)
                latent=latent[:,:,:latent_half_height,:]
            # ADDED 20240207:    
            elif alteration==[20]: #add regular copy to right
                latent=torch.cat((latent,latent_to_add),dim=3)
            elif alteration==[21]: #add flip UD copy to right
                latent=torch.cat((latent,latent_to_add.flip((2,0))),dim=3)
            elif alteration==[22]: #add flip LR copy to right
                latent=torch.cat((latent,latent_to_add.flip((3,0))),dim=3)
            elif alteration==[23]: #add flip LR+UD copy to right
                latent_flip=latent_to_add.flip((2,0))
                latent_flip=latent_flip.flip((3,0))
                latent=torch.cat((latent,latent_flip),dim=3)
            elif alteration==[24]: #add inverted copy to right
                latent=torch.cat((latent,latent_to_add*-1),dim=3)
            elif alteration==[25]: #add white latent to right
                latent=torch.cat((latent,get_white_latent(latent_to_add)),dim=3)

        if noise_variables[3]!=[[0,0]]:
            if len(noise_variables[3][0])==1:
                v_shift=noise_variables[3][0][0]
                h_shift=0
            elif len(noise_variables[3][0])==2:
                v_shift=noise_variables[3][0][0]
                h_shift=noise_variables[3][0][1]
            cumulative_shift[0]+=v_shift
            cumulative_shift[1]+=h_shift
            if v_shift!=0:
                display_status("Vertical shift "+str(v_shift))
                #specified row becomes the top row, rest moved to bottom
                if v_shift<0:
                    v_shift=latent.shape[2]-v_shift
                if v_shift<0 or v_shift>=latent.shape[2]:
                    display_status("Requested vertical shift exceeds latent dimension, not applied")
                else:
                    latent=latent_shift(latent,v_shift,0)
            if h_shift!=0:
                display_status("Horizontal shift "+str(h_shift))
                if h_shift<0:
                    h_shift=latent.shape[3]-h_shift
                if h_shift<0 or h_shift>=latent.shape[3]:
                    display_status("Requested horizontal shift exceeds latent dimensions, not applied")
                #specified row becomes the leftmost row, rest moved to right
                else:
                    latent=latent_shift(latent,0,h_shift)
            display_status("Cumulative shift "+str(cumulative_shift))
        if noise_variables[4]!=[[0,0,0,0,0]]:
            for entry in noise_variables[4]:
                if entry[4]==5: #crop to rectangle
                    latent=latent[:,:,entry[0]:latent.shape[2]-entry[1],entry[2]:latent.shape[3]-entry[3]]
                elif entry[4]==6: #fill in white
                    latent[0,:,entry[0]:latent.shape[2]-entry[1],entry[2]:latent.shape[3]-entry[3]]=get_white_latent(latent[0,:,entry[0]:latent.shape[2]-entry[1],entry[2]:latent.shape[3]-entry[3]])
                else:
                    for i in range(4):
                        latent_piece=latent[0,i,entry[0]:latent.shape[2]-entry[1],entry[2]:latent.shape[3]-entry[3]]
                        if entry[4]==0:
                            latent_piece*=0
                        elif entry[4]==1:
                            latent_piece*=-1
                        elif entry[4]==2:
                            latent_piece=torch.fliplr(latent_piece)
                        elif entry[4]==3: 
                            latent_piece=torch.flipud(latent_piece)
                        elif entry[4]==4: 
                            latent_piece=torch.fliplr(torch.flipud(latent_piece))
                        # ADDED 20240208
                        elif entry[4]==6:
                            latent_piece=get_white_latent(latent_piece)
                        latent[0,i,entry[0]:latent.shape[2]-entry[1],entry[2]:latent.shape[3]-entry[3]]=latent_piece
    return latent

def to_latent(img:Image,vae):
  generator = torch.Generator("cpu").manual_seed(0) #does appear deterministic
  #copied encoding method below from SD Deep Dive
  with torch.no_grad():
        latent = vae.encode(tfms.ToTensor()(img).unsqueeze(0).to(torch_device)*2-1) # Note scaling
  return vae.config.scaling_factor * latent.latent_dist.sample() 
    
def generate_noise_latent(seed_to_use,secondary_seed,height,width,noise_variables,scheduler,vae,num_inference_steps):
    if noise_variables[5]==[['',0]]:
        # Create noise latent from scratch
        generator = torch.Generator('cpu').manual_seed(seed_to_use)
        latent = torch.randn((1,4,int(height/8),int(width/8)),generator=generator)
    else:
        image_source=noise_variables[5][0][0]
        display_status('image2image: '+image_source)
        try:
            # Load as PIL image, encode it, add appropriate noise level
            try:
                image_input=Image.open(work_dir+image_source)
            except:
                image_input=Image.open(image_source)
            image_input=image_input.convert("RGB")
            torch.manual_seed(0)
            latent=to_latent(image_input,vae)
        except:
            #load .pt file
            try:
                package=torch.load(work_dir+image_source)
            except:
                package=torch.load(image_source)
            latent=package[0]
        strength=noise_variables[5][0][1]
        start_step=min(int(num_inference_steps*strength),num_inference_steps)
        #The above effectively inverts the usual strength scale
        #Ordinarily num_inference_steps*strength is number of steps to run
        #Here it's the step number to start with
        generator = torch.Generator('cpu').manual_seed(seed_to_use)
        noise = torch.empty_like(latent.to('cpu')).normal_(generator=generator)
        noise = noise.to(torch_device)
        #latent = scheduler.add_noise(latent,noise,timesteps=torch.tensor([scheduler.timesteps[start_step-1]])) 
        latent = scheduler.add_noise(latent,noise,timesteps=torch.tensor([scheduler.timesteps[start_step]]))
    if secondary_seed!=None:
        torch.manual_seed(secondary_seed)
    latent=vary_latent(latent,noise_variables)
    return latent

def generate_image_latent(latent,scheduler,num_inference_steps,guidance_scales,models,prompts,neg_prompts,step_latent_variables,i2i,contrastives,models_in_memory,models_in_memory_old,model_ids,shiftback):
    global prompt_defaults
    global cumulative_shift
    #Set counting variables
    prompt_counter=0
    neg_prompt_counter=0
    model_counter=0
    guidance_counter=0
    contrastives_counter=0
    step_latent_counter=0

    if i2i[0]!='':
        strength=i2i[1]
        start_step=min(int(num_inference_steps*strength),num_inference_steps)
        latent = latent.to(torch_device).float()
    else:
        start_step=0
        latent = latent.to(torch_device).float()
        latent=latent*scheduler.init_noise_sigma
    
    #Cycle through inference steps
    last_model=-1
    last_prompts=[]
    for tcount,t in enumerate(scheduler.timesteps):
        if tcount>=start_step:
            display_status("Inference timestep: "+str(tcount+1)+"/"+str(len(scheduler.timesteps)))
            if tcount==start_step or len(step_latent_variables)>1:
                latent=vary_latent(latent,step_latent_variables[step_latent_counter])
                height=latent.shape[2]*8
                width=latent.shape[3]*8
            if tcount==start_step or len(models['sequence'])>1: #If multiple models
                if models_in_memory==1:
                    if models['sequence'][model_counter]!=last_model or models_in_memory_old==0:
                        models_in_memory_old=1
                        display_status("Loading stored model "+str(models['sequence'][model_counter]))
                        if len(models[models['sequence'][model_counter]])==3:
                            [unet,tokenizer,text_encoder]=models[models['sequence'][model_counter]] #Get the model for this step
                            tokenizer_2=''
                            text_encoder_2=''
                        elif len(models[models['sequence'][model_counter]])==5:
                            [unet,tokenizer,text_encoder,tokenizer_2,text_encoder_2]=models[models['sequence'][model_counter]]
                    model_value=models['sequence'][model_counter]
                else:
                    if model_ids[models['sequence'][model_counter]]!=last_model:
                        display_status("Loading model "+str(models['sequence'][model_counter]))
                        unet=UNet2DConditionModel.from_pretrained(model_ids[models['sequence'][model_counter]],subfolder="unet")
                        unet = unet.to(torch_device)
                        text_encoder=CLIPTextModel.from_pretrained(model_ids[models['sequence'][model_counter]],subfolder="text_encoder")
                        #text_encoder = text_encoder.to(torch_device)
                        tokenizer=CLIPTokenizer.from_pretrained(model_ids[models['sequence'][model_counter]],subfolder="tokenizer")
                        try:
                            text_encoder_2=CLIPTextModelWithProjection.from_pretrained(model_ids[models['sequence'][model_counter]],subfolder="text_encoder_2")
                            #text_encoder_2 = text_encoder_2.to(torch_device) #OOM
                            tokenizer_2=CLIPTokenizer.from_pretrained(model_ids[models['sequence'][model_counter]],subfolder="tokenizer_2")
                        except:
                            text_encoder_2=''
                            tokenizer_2='' 
                    model_value=model_ids[models['sequence'][model_counter]]
            
            if tcount==start_step or (model_value!=last_model or prompts[prompt_counter]!=last_prompts): #If multiple models or prompts
                last_prompts=prompts[prompt_counter]
                display_status("EMBEDDING:")
                [embedding,pooled_embedding]=generate_prompt_embedding(prompts[prompt_counter],tokenizer,text_encoder,tokenizer_2,text_encoder_2) #Get the embedding for this step
                #CONTRASTIVE MANIPULATIONS HERE
                if contrastives[contrastives_counter][0]!=[[['',prompt_defaults],0]]:
                    display_status("Generating single-prompt contrastives:")
                    embedding_reference=embedding #Hold embedding stable
                    pooled_embedding_reference=pooled_embedding
                    for entry in contrastives[contrastives_counter][0]:
                        [alt_embedding,pooled_alt_embedding]=generate_prompt_embedding(entry[0],tokenizer,text_encoder,tokenizer_2,text_encoder_2)
                        embedding_diff=embedding_reference-alt_embedding #What's distinctive about 'main' embedding
                        embedding_diff*=entry[1]
                        embedding+=embedding_diff
                        if pooled_embedding_reference!='':
                            pooled_embedding_diff=pooled_embedding_reference-pooled_alt_embedding
                            pooled_embedding_diff*=entry[1]
                            pooled_embedding+=pooled_embedding_diff
                if contrastives[contrastives_counter][1]!=[[['',prompt_defaults],['',prompt_defaults],0]]:
                    display_status("Generating prompt-pair contrastives:")
                    for entry in contrastives[contrastives_counter][1]:
                        [alt_embedding_1,pooled_alt_embedding_1]=generate_prompt_embedding(entry[0],tokenizer,text_encoder,tokenizer_2,text_encoder_2)
                        [alt_embedding_2,pooled_alt_embedding_2]=generate_prompt_embedding(entry[1],tokenizer,text_encoder,tokenizer_2,text_encoder_2)
                        embedding_diff=alt_embedding_1-alt_embedding_2
                        embedding+=(embedding_diff*entry[2])
                        if pooled_embedding!='':
                            pooled_embedding_diff=pooled_alt_embedding_1-pooled_alt_embedding_2
                            pooled_embedding+=(pooled_embedding_diff*entry[2])
                if text_encoder_2=='' or neg_prompts[neg_prompt_counter][0]!='':
                    display_status("UNCOND/NEG EMBEDDING:")
                    [uncond_embedding,pooled_uncond_embedding]=generate_prompt_embedding(neg_prompts[neg_prompt_counter],tokenizer,text_encoder,tokenizer_2,text_encoder_2)        
                    # If negative prompt is assigned a strength other than 1
                    if neg_prompts[neg_prompt_counter][2]!=1 and neg_prompts[neg_prompt_counter][0]!='':
                        if text_encoder_2!='': #ForSDXL-type models just multiply
                            uncond_embedding*=neg_prompts[neg_prompt_counter][2]
                        else: #otherwise use difference from unconditional embedding
                            [ref_uncond_embedding,ref_pooled_uncond_embedding]=generate_prompt_embedding(['',[prompt_defaults,prompt_defaults]],tokenizer,text_encoder,tokenizer_2,text_encoder_2)
                            embedding_diff=uncond_embedding-ref_uncond_embedding
                            uncond_embedding=ref_uncond_embedding+(embedding_diff*neg_prompts[neg_prompt_counter][2])
                else:
                    display_status("Using zeroed-out UNCOND/NEG EMBEDDING")
                    uncond_embedding=torch.zeros_like(embedding)
                    pooled_uncond_embedding=torch.zeros_like(pooled_embedding)
                    
                embedding = torch.cat([uncond_embedding,embedding])
                
            if tcount==start_step or len(guidance_scales)>1: #If multiple guidance scales
                guidance_scale=guidance_scales[guidance_counter] #Get the guidance scale for this step
            latent_model_input=torch.cat([latent]*2)
            latent_model_input=scheduler.scale_model_input(latent_model_input,timestep=t)
            with torch.no_grad():
                if text_encoder_2=='':
                    noise_pred=unet(latent_model_input.to(torch_device),t,encoder_hidden_states=embedding.to(torch_device)).sample
                else:
                    original_size=(width,height)
                    target_size=(width,height)
                    crops_coords_top_left=(0,0)
                    add_time_ids = torch.tensor([list(original_size + crops_coords_top_left + target_size)])
                    negative_add_time_ids = add_time_ids
                    add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0).to(torch_device)
                    add_text_embeds=torch.cat([pooled_uncond_embedding, pooled_embedding], dim=0).to(torch_device)
                    added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                    noise_pred=unet(latent_model_input.to(torch_device),t,encoder_hidden_states=embedding.to(torch_device),added_cond_kwargs=added_cond_kwargs,return_dict=False)[0]
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond+guidance_scale*(noise_pred_text - noise_pred_uncond)
            
            latent = scheduler.step(noise_pred,t,latent).prev_sample
            
            #if last round and option is selected, 
            if tcount==num_inference_steps-1 and cumulative_shift!=[0,0] and shiftback==1:
                display_status("Re-setting latent to original boundaries....")
                v_shiftback=latent.shape[2]-cumulative_shift[0]
                h_shiftback=latent.shape[3]-cumulative_shift[1]
                while v_shiftback<0:
                    v_shiftback+=latent.shape[2]
                while h_shiftback<0:
                    h_shiftback+=latent.shape[3]
                latent=latent_shift(latent,v_shiftback,h_shiftback)
    
            #Iterate and/or reset counters
            last_model=model_value
            last_prompt=prompts[prompt_counter]
            model_counter+=1
            if model_counter==len(models['sequence']):
                model_counter=0
            prompt_counter+=1
            if prompt_counter==len(prompts):
                prompt_counter=0
            neg_prompt_counter+=1
            if neg_prompt_counter==len(neg_prompts):
                neg_prompt_counter=0
            guidance_counter+=1
            if guidance_counter==len(guidance_scales):
                guidance_counter=0
            contrastives_counter+=1
            if contrastives_counter==len(contrastives):
                contrastives_counter=0
            step_latent_counter+=1
            if step_latent_counter==len(step_latent_variables):
                step_latent_counter=0
            
    return latent   
        
def generate_image_from_latent(vae,latent):
    with torch.no_grad():
        image = vae.decode(latent / vae.config.scaling_factor, return_dict=False)[0]
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()[0]
    image = (image * 255).round().astype("uint8")
    image = Image.fromarray(image)
    return image

def prepare_model_data(model_numbers,model_ids,model_prompts,models_in_memory):
    models={}
    if models_in_memory==1: #Stores models in memory to avoid repeated loading at expense of memory usage
        display_status('Loading models into memory.')
        models['sequence']=[]
        models['prompt']=[]
        model_counter=0
        number_pool=[]
        for mcount,model_number in enumerate(model_numbers):
            models['prompt'].append(model_prompts[model_number])
            if model_number not in number_pool:
                display_status('Model sequence '+str(mcount)+': Storing model number '+str(model_number)+' in position '+str(model_counter))
                unet=UNet2DConditionModel.from_pretrained(model_ids[model_number],subfolder="unet")
                unet = unet.to(torch_device)
                text_encoder=CLIPTextModel.from_pretrained(model_ids[model_number],subfolder="text_encoder")
                #text_encoder = text_encoder.to(torch_device)
                tokenizer=CLIPTokenizer.from_pretrained(model_ids[model_number],subfolder="tokenizer")
                try:
                    text_encoder_2=CLIPTextModelWithProjection.from_pretrained(model_ids[model_number],subfolder="text_encoder_2")
                    #text_encoder_2=text_encoder_2.to(torch_device) #OOM error
                    tokenizer_2=CLIPTokenizer.from_pretrained(model_ids[model_number],subfolder="tokenizer_2")
                    models[model_counter]=[unet,tokenizer,text_encoder,tokenizer_2,text_encoder_2]    
                except:
                    models[model_counter]=[unet,tokenizer,text_encoder]
                number_pool.append(model_number)
                models['sequence'].append(model_counter)
                model_counter+=1
            else:
                display_status('Model sequence '+str(mcount)+': Model number '+str(model_number)+' already stored in memory') 
                models['sequence'].append(number_pool.index(model_number))
    else:
        models['sequence']=model_numbers
        models['prompt']=[]
        for model_number in model_numbers:
            models['prompt'].append(model_prompts[model_number])
    return models

def spectrophone(image,filename,audio_channels,sample_rate,autype):
    max_volume = 50
    power_for_image = 0.25
    n_mels = 512
    padded_duration_ms = 400  # [ms] 
    step_size_ms = 10  
    mel_scale=True
    max_mel_iters=200
    num_griffin_lim_iters=32
    window_duration_ms=100
    n_fft = int(padded_duration_ms / 1000.0 * sample_rate)
    hop_length = int(step_size_ms / 1000.0 * sample_rate)
    win_length = int(window_duration_ms / 1000.0 * sample_rate)
    data_set = np.array(image).astype(np.float32)
    waveform_group={}
    for color_channel in range(audio_channels):
        #reverses direction of vertical axis, selects only first color channel
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
                #max_iter=max_mel_iters #Works in some versions, raises an error in others
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
        waveform=waveform/np.max(waveform)
        if audio_channels>1:
            waveform_group[color_channel]=waveform
    if audio_channels==1:
        audio_filename=filename+'.'+autype
        sf.write(audio_filename, waveform.T, sample_rate)
    elif audio_channels==2:
        stereo_waveform=np.vstack((waveform_group[0],waveform_group[1]))
        audio_filename=filename+'_stereo.'+autype
        sf.write(audio_filename, stereo_waveform.T, sample_rate)
    elif audio_channels==3:
        for channel in range(3):
            audio_filename=filename+'_'+str(channel)+'.'+autype
            sf.write(audio_filename,waveform_group[channel].T,sample_rate)
            
def make_spectrogram_from_audio(waveform,filename,sample_rate,normalize,rewidth):
    power_for_image = 0.25
    n_mels = 512
    padded_duration_ms = 400  
    step_size_ms = 10  
    window_duration_ms=100
    n_fft = int(padded_duration_ms / 1000.0 * sample_rate)
    hop_length = int(step_size_ms / 1000.0 * sample_rate)
    win_length = int(window_duration_ms / 1000.0 * sample_rate)
    spectrogram_func = torchaudio.transforms.Spectrogram(
        n_fft=n_fft,
        power=None,
        hop_length=hop_length,
        win_length=win_length,
    )
    waveform_tensor = torch.from_numpy(waveform.astype(np.float32)).reshape(1, -1)
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
    #If normalize = 0 then divide by maximum of each clip
    if normalize==0:
        data = data-np.min(data)
        data = data / np.max(data)
    else:
        data = data / normalize 
        data[data>1]=1
    data = data * 255
    data = 255 - data
    image = Image.fromarray(data.astype(np.uint8))
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    #image = image.convert("RGB")
    if rewidth==1:
        if image.width/8!=int(image.width/8):
            new_width=(int(image.width/8)+1)*8
            image=image.resize((new_width,512))
    return image

def prep_audio_input(path,subfolder,estimatedDuration,testMargin,reverse,normalize,rewidth,defwidth,freqblur,double,imtype):
    try:
        y_in, sr = a2n.audio_from_file(path)
    except:
        y_in, sr = a2n.audio_from_file(work_dir+path)    
    try:
        if reverse==1:
            y_in=np.flipud(y_in)
        y=y_in[:,0]
        channels=2
    except:
        if reverse==1:
            y_in=np.flip(y_in)
        y=y_in
        channels=1
    if defwidth==0:
        frac=10
        #Low-resolution pass
        estimate20=(estimatedDuration*sr)/frac
        testMargin20=(testMargin*sr)/frac
        startLoopLength=int(estimate20-testMargin20)
        stopLoopLength=int(estimate20+testMargin20)
        if(len(y.shape))==2:
            yrectified=np.abs(np.sum(y,axis=1)/2)
        else:
            yrectified=np.abs(y)    
        yrectdown=signal.resample(yrectified,int(yrectified.shape[0]/20))        
        maxval=[]
        for testwidth in range(startLoopLength,stopLoopLength):
            testheight=int(np.floor(yrectdown.shape[0])/testwidth)
            testshape=yrectdown[0:(testwidth*testheight)].reshape((testheight,testwidth))
            testsum=np.sum(testshape,axis=0)/testheight
            maxval.append(np.max(testsum)-np.min(testsum[testsum>0]))
        loresval=int((np.argmax(maxval))+startLoopLength)*frac
        maxval=[]
        startLoopLength=int(loresval-frac)
        stopLoopLength=int(loresval+frac)
        for testwidth in range(startLoopLength,stopLoopLength):
            testheight=int(np.floor(yrectified.shape[0])/testwidth)
            testshape=yrectified[0:(testwidth*testheight)].reshape((testheight,testwidth))
            testsum=np.sum(testshape,axis=0)/testheight
            maxval.append(np.max(testsum)-np.min(testsum))  
        topval=int(np.argmax(maxval)+startLoopLength)    
    else:
        topval=defwidth
    if subfolder=='':
        filename_start='audio_'
    else:
        filename_start=subfolder+'/audio_'
    if double==1:
        if channels==1:
            y_in=np.pad(y_in,topval)
        else:
            y_in=np.pad(y_in,((topval,topval),(0,0)))
    counter=100000
    for rowcount in range(0,y_in.shape[0],topval):
        for channel in range(channels):
            if double==0:
                if channels==1:
                    weftrow=y_in[rowcount:rowcount+topval]
                else:
                    weftrow=y_in[rowcount:rowcount+topval,channel]
            else:
                if channels==1:
                    weftrow=y_in[rowcount:rowcount+(topval*2)]
                else:
                    weftrow=y_in[rowcount:rowcount+(topval*2),channel]
            if counter==100000:
                row_width=weftrow.shape[0]
                weftrow_temp=np.zeros_like(weftrow)
            elif weftrow.shape[0]!=row_width:
                weftrow_temp[:weftrow.shape[0]]=weftrow
                weftrow=weftrow_temp
            filename=filename_start+str(counter)+'.'+imtype
            image=make_spectrogram_from_audio(weftrow,filename,sr,normalize,rewidth)
            if freqblur!=None:
                orig_imwidth=image.width
                orig_imheight=image.height
                image=image.resize((orig_imwidth,int(orig_imheight/freqblur)))
                image=image.resize((orig_imwidth,orig_imheight))
            if channel==0:
                image_out=image
            elif channel==1:
                image_out=Image.merge('RGB',(image_out,image,image))
        if not os.path.isdir(subfolder):
            os.makedirs(subfolder)
        image_out.convert('RGB')
        image_out.save(filename)
        counter+=1
            
def make_video(path,raw_job_string,job):
    img_array=[]
    recreated_job_string=''
    path=work_dir+path
    for image in os.listdir(path): 
        if not image.endswith('.pt'):
            try:
                display_status("Added "+image)
                frame = cv2.imread(path+'/'+image)
                height, width, layers = frame.shape
                size = (width,height)
                img_array.append(frame)
            except:
                display_status("Skipped "+image)
        else:
            package_contents=torch.load(path+'/'+image)
            recreated_job_string+='#new '+package_contents[1]
    recreated_job_string+='#new '+raw_job_string
    package=[img_array,recreated_job_string,job,__file__]
    torch.save(package,path+'.pt')
    out = cv2.VideoWriter(path+'.avi',cv2.VideoWriter_fourcc(*'MJPG'), 30, size)
    for i in range(len(img_array)):
        x=out.write(img_array[i])
    out.release()
    display_status('Video written.')

# Helper function for converting comma-delimited string inputs into numerical lists
def num_parse(instring):
    out_list=[]
    separate_strings=instring.split('+')
    for string in separate_strings:
        string_list=[]
        string_parts=string.split(',')
        for string_part in string_parts:
            try:
                string_list.append(int(string_part))
            except:
                string_list.append(float(string_part))
        out_list.append(string_list)
    return out_list

## WAVCAT FUNCTIONS

def altGriffinLim(specgram,sample_rate,win_dur):
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
    window=torch.hann_window(win_length, periodic=True).to(torch_device)
    shape = specgram.size()
    specgram = specgram.reshape([-1] + list(shape[-2:]))
    specgram = specgram.pow(1 / 1.0)
    angles = torch.full(specgram.size(), 1, dtype=_get_complex_dtype(specgram.dtype), device=specgram.device)
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
    
def spectralize(image,sample_rate,win_dur,audio_channels):
    data_set = np.array(image).astype(np.float32)
    angles={}
    specgram={}
    n_fft = int(400 / 1000.0 * sample_rate)
    win_length = int(win_dur / 1000.0 * sample_rate)
    hop_length = int(10 / 1000.0 * sample_rate)
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
        #reverses direction of vertical axis, selects only designated color channel
        data = data_set[::-1, :, color_channel]
        data = 255 - data
        data = data * 50 / 255
        data = np.power(data, 1 / 0.25)
        Sxx_torch = torch.from_numpy(data).to(torch_device)
        Sxx_torch = mel_inv_scaler(Sxx_torch)
        specgram[color_channel]=Sxx_torch.to('cpu')
        angles[color_channel] = altGriffinLim(Sxx_torch,sample_rate,win_dur).to('cpu')   
    #To clear CUDA memory
    if torch_device=='cuda':
        Sxx_torch=torch.zeros(1)
        Sxx_torch=Sxx_torch.to('cpu')
        torch.cuda.empty_cache()
    return specgram,angles

def toWaveform(specgram,angles,sample_rate,win_dur):
    n_fft = int(400 / 1000.0 * sample_rate)
    win_length = int(win_dur / 1000.0 * sample_rate)
    hop_length = int(10 / 1000.0 * sample_rate)
    window=torch.hann_window(win_length, periodic=True).to(torch_device)
    specgram=torch.from_numpy(specgram).to(torch_device)
    angles=torch.from_numpy(angles).to(torch_device)
    waveform = torch.istft(
        specgram * angles, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, length=None
    )
    shape = specgram.size()
    waveform = waveform.reshape(shape[:-2] + waveform.shape[-1:])
    waveform = waveform.to('cpu')
    return waveform

def wavWrite(waveform,filename,sample_rate):
    waveform_adjusted=waveform/np.max(waveform)
    sf.write(filename, waveform_adjusted.T, sample_rate)
    
def wavcat(filepath,vae_option,chunk_size,overlap_size,overlap_type,sample_rate,win_dur,subfolder,audio_channels,autype,raw_job_string):
    job=[filepath,vae_option,chunk_size,overlap_size,overlap_type,sample_rate,win_dur,subfolder,audio_channels,autype]
    if filepath.startswith(work_dir):
        folder_filepath=filepath
    else:
        folder_filepath=work_dir+filepath
    vae=AutoencoderKL.from_pretrained(vae_option,subfolder="vae")
    vae=vae.to(torch_device)
    imlist=[]
    for image_location in os.listdir(folder_filepath):
        if image_location.endswith('.pt'):
            imlist.append(image_location)
    counter=0
    tiled_image=''
    recreated_job_string=''
    for image_location in imlist:
        try:
            package_contents=torch.load(folder_filepath+'/'+image_location)
            recreated_job_string+='#new '+package_contents[1]
            image_file=package_contents[0]
            if counter==0:
                tiled_image=image_file
                counter=1
            else:
                tiled_image=torch.cat((tiled_image,image_file),3)
            display_status("Concatenated "+image_location)
        except:
            display_status("Failed to concatenate "+image_location)
    recreated_job_string+='#new '+raw_job_string
    display_status("Tiled image size = "+str(tiled_image.shape))
    if tiled_image!='':
        output_string=filepath+'_wavcat_'
        datestring=str(datetime.datetime.now())
        datestring=datestring.replace(":","-")
        datestring=datestring.replace(".","-")
        datestring=datestring.replace(" ","-")
        output_string=output_string+'_'+datestring
        if subfolder!=None:
            if not os.path.exists(subfolder):
                os.makedirs(subfolder)
            output_string=subfolder+'/'+output_string
        package=[tiled_image,recreated_job_string,job,__file__]
        torch.save(package,output_string+'.pt')
        chunk_count=int(tiled_image.shape[3]/chunk_size)
        mag_sequence={}
        angle_sequence={}
        for i in range(chunk_count):
            display_status("Processing chunk "+str(i+1)+" of "+str(chunk_count))
            image_part=tiled_image[:,:,:,(i)*chunk_size:((i+1)*chunk_size)+overlap_size] 
            image_part=generate_image_from_latent(vae,image_part)
            mags,angles=spectralize(image_part,sample_rate,win_dur,audio_channels)
            if i==0:
                display_status("Getting unpadded size for cropping")
                test_mags,test_angles=spectralize(generate_image_from_latent(vae,tiled_image[:,:,:,(i)*chunk_size:((i+1)*chunk_size)]),sample_rate,win_dur,1)
                cropping_size=mags[0].shape[1]-test_mags[0].shape[1]
                cropping_size_1=int(cropping_size/2)
                cropping_size_2=cropping_size-cropping_size_1
                mags_forward={}
                angles_forward={}
                for k in range(audio_channels):
                    mags_forward[k]=mags[k][:,-cropping_size_2:]
                    angles_forward[k]=angles[k][:,:,-cropping_size_2:]
                    mag_sequence[k]=mags[k][:,:-cropping_size_2]
                    angle_sequence[k]=angles[k][:,:,:-cropping_size_2]
            else:
                mags_next={}
                angles_next={}
                for k in range(audio_channels):
                    mags_next[k]=mags[k][:,cropping_size_1:-cropping_size_2]
                    angles_next[k]=angles[k][:,:,cropping_size_1:-cropping_size_2]
                    lapwidth=int(mags_forward[k].shape[1])
                    difflist=[]
                    for j in range(lapwidth):
                        if overlap_type==1:
                            #find most similar column in the overlap
                            difflist.append(np.sum(np.abs(mags_next[k].numpy()[:,j]-mags_forward[k].numpy()[:,j])))
                        elif overlap_type==2:
                            #find column with mutually lowest amplitude in the overlap
                            difflist.append(np.sum(np.abs(mags_next[k].numpy()[:,j]+mags_forward[k].numpy()[:,j])))
                    #get the index of that column
                    diffmin=difflist.index(min(difflist))
                    #substitute the portion of the previous segment leading up to that column
                    mags_next[k][:,:diffmin]=mags_forward[k][:,:diffmin]
                    angles_next[k][:,:,:diffmin]=angles_forward[k][:,:,:diffmin]
                    mag_sequence[k]=np.hstack((mag_sequence[k],mags_next[k]))
                    angle_sequence[k]=np.dstack((angle_sequence[k],angles_next[k]))
                    mags_forward[k]=mags[k][:,-cropping_size_2:]
                    angles_forward[k]=angles[k][:,:,-cropping_size_2:]
        display_status("Processing remainder")
        image_part=tiled_image[:,:,:,(i+1)*chunk_size:]
        if image_part.shape[3]>0:
            image_part=generate_image_from_latent(vae,image_part)
            mags,angles=spectralize(image_part,sample_rate,win_dur,audio_channels)
            for k in range(audio_channels):
                mags_next[k]=mags[k][:,cropping_size_1:]
                angles_next[k]=angles[k][:,:,cropping_size_1:]
                lapwidth=int(mags_forward[k].shape[1])
                try:
                    difflist=[]
                    for j in range(lapwidth):
                        if overlap_type==1:
                            #find most similar column in the overlap
                            difflist.append(np.sum(np.abs(mags_next[k].numpy()[:,j]-mags_forward[k].numpy()[:,j])))
                        elif overlap_type==2:
                            #find column with mutually lowest amplitude in the overlap
                            difflist.append(np.sum(np.abs(mags_next[k].numpy()[:,j]+mags_forward[k].numpy()[:,j])))
                    #get the index of that column
                    diffmin=difflist.index(min(difflist))
                    mags_next[k][:,:diffmin]=mags_forward[k][:,:diffmin]
                    angles_next[k][:,:,:diffmin]=angles_forward[k][:,:,:diffmin]
                    mag_sequence[k]=np.hstack((mag_sequence[k],mags_next[k]))
                    angle_sequence[k]=np.dstack((angle_sequence[k],angles_next[k]))
                except: #if anything goes wrong with the final step
                    mag_sequence[k]=np.hstack((mag_sequence[k],mags_forward[k]))
                    angle_sequence[k]=np.dstack((angle_sequence[k],angles_forward[k]))
        display_status("Image width = "+str(mag_sequence[0].shape[1]))
        display_status("Processing channel one")
        waveform=toWaveform(mag_sequence[0],angle_sequence[0],sample_rate,win_dur)
        display_status("WAV width = "+str(waveform.shape[0]))
        if audio_channels==1 or audio_channels==3:
            wavWrite(np.array(waveform),output_string+'_01.'+autype,sample_rate)
        if audio_channels==2:
            display_status("Processing channel two")
            stereo_waveform=np.vstack((waveform,toWaveform(mag_sequence[1],angle_sequence[1],sample_rate,win_dur)))
            wavWrite(stereo_waveform,output_string+'.'+autype,sample_rate)
        else:
            display_status("Processing channel two")
            wavWrite(np.array(toWaveform(mag_sequence[1],angle_sequence[1],sample_rate,win_dur)),output_string+'_02.'+autype,sample_rate)
            display_status("Processing channel three")
            wavWrite(np.array(toWaveform(mag_sequence[2],angle_sequence[2],sample_rate,win_dur)),output_string+'_03.'+autype,sample_rate) 
        display_status("Wavcat complete.")

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
    continue_script=1
    save_script()
    if os.path.exists(path_to_resume_point) and engage_mode==2:
        file=open(path_to_resume_point)
        resume_point=int(file.read())
        file.close()
    else:
        resume_point=0
    models_in_memory_old=0
    old_model_numbers=[-1]
    old_scheduler_number=[-1]
    
    #Read in a SyDNEy script
    with open(path_to_script, 'r') as file:
        init_job_string_input = file.read().replace('\n', '')
        init_job_string_input+='   '
        
    # Import configuration if given
    job_string_input_whole=init_job_string_input.split('#body')
    configuration=''
    if len(job_string_input_whole)>1:
        configuration=job_string_input_whole[0]
        job_string_input_whole=job_string_input_whole[1]
    else:
        job_string_input_whole=job_string_input_whole[0]
        try:
            with open(path_to_config,'r') as file:
                configuration = file.read().replace('\n', '')
        except:
            display_status("No sydney_config.txt file found, using default configuration.")
    
    job_string_inputs=job_string_input_whole.split('#restart')

    for job_string_input in job_string_inputs:
        if True:       
        #try:    
            if configuration!='':        
                config_parts=configuration.split('#')
                for config_part in config_parts:
                    if config_part.startswith('models'):
                        config_part=config_part[6:]
                        model_strings=config_part.split(';')
                        model_ids=[]
                        model_prompts=[]
                        model_counter=0
                        for model_string in model_strings:
                            model_string=model_string.split(',')
                            if len(model_string)==1:
                                model_ids.append(model_string[0].strip())
                                model_prompts.append('')
                            elif len(model_string)==2:
                                model_ids.append(model_string[0].strip())
                                model_prompts.append(model_string[1].strip())
                            model_counter+=1  
                
            # Supply placeholder variables for settings that don't otherwise need any
                job_string_input=job_string_input.replace('$2copy','$2copy 0')
                job_string_input=job_string_input.replace('$2neg_copy','$2neg_copy 0')
                job_string_input=job_string_input.replace('$reverse','$reverse 1')
                job_string_input=job_string_input.replace('$keep_width','$keep_width 1')
                job_string_input=job_string_input.replace('$double','$double 1')  
                job_string_input=job_string_input.replace('$nodecode','$nodecode 1')
                job_string_input=job_string_input.replace('$noimgsave','$noimgsave 1')
            
            # Unpack scrape jobs
            while '#scrape' in job_string_input or '#for' in job_string_input:
                if '#scrape' in job_string_input:
                    start_scrape=job_string_input.find('#scrape')
                    end_scrape=job_string_input[start_scrape:].find(':')
                    before_part=job_string_input[:start_scrape]
                    after_part=job_string_input[end_scrape+start_scrape:]
                    
                    if job_string_input[start_scrape+7:start_scrape+10]=='dir':
                        scrape_segment=job_string_input[start_scrape+11:end_scrape+start_scrape].split(' ',maxsplit=1)
                        scrapedir=1
                    else:
                        scrape_segment=job_string_input[start_scrape+8:end_scrape+start_scrape].split(' ',maxsplit=1)
                        scrapedir=0    
                    scrape_variable=scrape_segment[0].strip()
                    scrape_directory=scrape_segment[1].strip()
                    try:
                        scrape_directory_temp=work_dir+scrape_directory
                        files=os.listdir(scrape_directory_temp)
                    except:
                        scrape_directory_temp=scrape_directory
                        files=os.listdir(scrape_directory_temp)
                    scrape_string='#for '+scrape_variable+' '
                    for file in files:
                        if scrapedir==0 and os.path.isfile(scrape_directory_temp+'/'+file): 
                            # Should the above be limitable to certain file types? How?
                            scrape_string+=scrape_directory+'/'+file+';'
                        elif scrapedir==1 and os.path.isdir(scrape_directory_temp+'/'+file):
                            scrape_string+=scrape_directory+'/'+file+';'
                    scrape_string=scrape_string[:-1]
                    job_string_input=before_part+scrape_string+after_part

                # Nested here in case a #scrape loop is accessed by another #scrape loop
                while '#for' in job_string_input:
                    # Note, if only one option listed under #for there's an error -- does this need fixing?
                    # Unpack multiple jobs from all specified 'for' loops, if any, including nested ones
                    start_for=job_string_input.find('#for')
                    for_variable=job_string_input[start_for+5:].split(' ',maxsplit=1)
                    end_for=job_string_input.find('#end '+for_variable[0],start_for)
                    string_to_unpack=job_string_input[start_for:end_for]
                    before_part=job_string_input[:start_for]
                    after_part=job_string_input[end_for+5+len(for_variable[0]):]
                    string_to_unpack=string_to_unpack.split(':',maxsplit=1)
                    substitutions=string_to_unpack[0].split(';')
                    substitutions[0]=substitutions[0].replace('#for '+for_variable[0],'').strip()
                    new_string=before_part
                    for substitution in substitutions:
                        new_string+=string_to_unpack[1].replace(for_variable[0],substitution)
                    new_string+=after_part
                    job_string_input=new_string
                    
            while '#incr' in job_string_input:
                start_incr=job_string_input.find('#incr')
                incr_variable=job_string_input[start_incr+6:].split(' ',maxsplit=1)
                end_incr=job_string_input.find('#end '+incr_variable[0])
                string_to_unpack=job_string_input[start_incr:end_incr]
                before_part=job_string_input[:start_incr]
                after_part=job_string_input[end_incr+5+len(incr_variable[0]):]
                string_to_unpack=string_to_unpack.split(':',maxsplit=1)
                incr_variables=string_to_unpack[0].split(';')
                incr_variables[0]=incr_variables[0].replace('#incr '+incr_variable[0],'').strip()
                new_string=before_part
                if len(incr_variables)==1:
                    try:
                        incr_end=int(incr_variables[0])
                    except:
                        incr_end=float(incr_variables[0])
                    incr_start=1
                    incr_step=1
                    eval_phase='x'
                elif len(incr_variables)==2:
                    try:
                        incr_end=int(incr_variables[1])
                    except:
                        incr_end=float(incr_variables[1])
                    try:
                        incr_start=int(incr_variables[0])
                    except:
                        incr_start=float(incr_variables[0])
                    incr_step=1
                    eval_phase='x'
                elif len(incr_variables)==3:
                    try:
                        incr_end=int(incr_variables[1])
                    except:
                        incr_end=float(incr_variables[1])
                    try:
                        incr_start=int(incr_variables[0])
                    except:
                        incr_start=float(incr_variables[0])
                    try:
                        incr_step=int(incr_variables[2])
                    except:
                        incr_step=float(incr_variables[2])
                    eval_phase='x'
                elif len(incr_variables)==4:
                    try:
                        incr_end=int(incr_variables[1])
                    except:
                        incr_end=float(incr_variables[1])
                    try:
                        incr_start=int(incr_variables[0])
                    except:
                        incr_start=float(incr_variables[0])
                    try:
                        incr_step=int(incr_variables[2])
                    except:
                        incr_step=float(incr_variables[2])
                    eval_phase=incr_variables[3]
                incr=incr_start
                while incr*np.sign(incr_step)<=incr_end*np.sign(incr_step):
                    ldict={}
                    exec('incr_eval='+eval_phase.replace('x',str(incr)),globals(),ldict)
                    incr_eval=ldict['incr_eval']
                    new_string+=string_to_unpack[1].replace(incr_variable[0],str(incr_eval))
                    incr+=incr_step
                new_string+=after_part
                job_string_input=new_string

            while '#set' in job_string_input:
                # Isolate the bounded segment of text, and the segments before and after it
                start_for=job_string_input.find('#set')
                end_for=job_string_input.find('#unset')
                string_to_unpack=job_string_input[start_for:end_for]
                before_part=job_string_input[:start_for]
                after_part=job_string_input[end_for+6:]
                # Split the segment at the colon into:
                # (0) the part to copy and
                # (1) the part into which to copy at each #new
                string_to_unpack=string_to_unpack.split(':',maxsplit=1)
                # Split off any $contrast    
                # Remove initial #new from (1) if it's present
                # (which would otherwise yield an extra job of (0) by itself)
                if string_to_unpack[1].startswith('#new'):
                    string_to_unpack[1]=string_to_unpack[1][4:]
                # Change (0) to begin with #new instead of #set and to end with a space
                string_to_unpack[0]=string_to_unpack[0].replace('#set ','#new ')+' '
                # If there's a $contrast segment, split it off to go to the end
                if '$contrast ' in string_to_unpack[0]:
                    [string_to_unpack[0],contrast]=string_to_unpack[0].split('$contrast ',maxsplit=1)
                    contrast="$contrast "+contrast
                else:
                    contrast=''
                # Add #break at the end of (0) if it contains any #steps
                if '#step' in string_to_unpack[0]:
                    string_to_unpack[0]+=' #break '
                # Create backup of current string_to_unpack[0]
                backup_string_to_unpack=string_to_unpack[0]
                # Append contrast segment to start of string_to_unpack[0] (to occur at end of preceding job)
                string_to_unpack[0]=contrast+string_to_unpack[0]
                # Replace every #new in (1) with the altered (0)
                # Which effectively inserts (0) after each #new, minus its original #set
                # and add a final contrast segment at the end
                string_to_unpack[1]=string_to_unpack[1].replace('#new',string_to_unpack[0])+contrast
                # FIGURE OUT THE FOLLOWING BUT IT WORKS FOR NOW
                # Rejoin before part, initial #set string minus contrast, #set sequence, and after part
                new_string=before_part+backup_string_to_unpack+string_to_unpack[1]+after_part
                job_string_input=new_string
                
            while '{' in job_string_input:
                start_exp=job_string_input.find('{')
                end_exp=job_string_input.find('}')
                exp=job_string_input[start_exp+1:end_exp]
                ldict={}
                exec('exp_eval='+exp,globals(),ldict)
                exp_eval=ldict['exp_eval']
                job_string_input=job_string_input.replace('{'+exp+'}',str(exp_eval))  
                
            while '#copy' in job_string_input:
                start_exp=job_string_input.find('#copy')
                copy_source=job_string_input[start_exp+6:].split(' ',maxsplit=1)[0]
                try:
                    package=torch.load(copy_source)
                    copy_string=package[1].strip()
                    #print('>>'+copy_string)
                except:
                    copy_string='***CANCEL***' #to stop jobs from being run
                job_string_input=job_string_input.replace('#copy '+copy_source,copy_string+' #break ')
                
            #Split (unpacked) individual job strings into separate items in a list
            job_string_list=job_string_input.split('#new')
        
            #Parse individual job strings and convert them into an actionable data structure
            jobs=[]
            raw_job_strings=[]
            for raw_job_string in job_string_list:
                if raw_job_string.strip()!='' and '***CANCEL***' not in raw_job_string: #First condition guards against an initial '#new' or equivalent
                    raw_job_strings.append(raw_job_string)
                    if engage_mode==0 or engage_mode==2:
                        raw_job_string=raw_job_string.replace('$','^$')
                        raw_job_string=raw_job_string.replace('#','^#')
                        raw_job_string=raw_job_string.split('^')[1:]
                        job_string=[]
                        for segment in raw_job_string:
                            if segment.startswith('$'):
                                segment_parts=segment.split(' ',maxsplit=1)
                                job_string.append(segment_parts[0])
                                job_string.append(segment_parts[1].strip())
                            else:
                                job_string.append(segment)
                    
                        cell_count=0
                        step_count=0
                        loop_count=0
                        job={}
                        jobtemp={}
                      
                        while cell_count<len(job_string):
                            if '#break' in job_string[cell_count]:
                                # Use to flag end of a #step sequence applied via #set
                                loop_count+=1
                                step_count=0
                                cell_count+=1
                            if '#step' in job_string[cell_count] or '#start' in job_string[cell_count] or '$contrast' in job_string[cell_count]:
                                if '$contrast' not in job_string[cell_count]:
                                    if '#start' in job_string[cell_count]:
                                        loop_count+=1
                                        step_count=0
                                    cell_count+=1
                                if loop_count not in jobtemp:
                                    jobtemp[loop_count]={}
                                if step_count not in jobtemp[loop_count]:
                                    jobtemp[loop_count][step_count]={}
                                con_count=0 #count of contrastive entries
                                with_on=0 #switch for turning on a 'with' entry
                                while cell_count<len(job_string) and '#' not in job_string[cell_count]:
                                    if '$with' in job_string[cell_count]:
                                        with_on=1
                                        jobtemp[loop_count][step_count][con_count]['with']={}
                                        jobtemp[loop_count][step_count][con_count]['with']['prompt']=job_string[cell_count+1]
                                        cell_count+=2
                                    elif '$contrast' in job_string[cell_count]:
                                        con_count+=1
                                        jobtemp[loop_count][step_count][con_count]={}
                                        jobtemp[loop_count][step_count][con_count]['prompt']=job_string[cell_count+1]
                                        cell_count+=2
                                        with_on=0
                                    elif with_on==1:
                                        jobtemp[loop_count][step_count][con_count]['with'][job_string[cell_count][1:]]=job_string[cell_count+1]
                                        cell_count+=2
                                    elif con_count==0:
                                        jobtemp[loop_count][step_count][job_string[cell_count][1:]]=job_string[cell_count+1]
                                        cell_count+=2
                                    else:
                                        jobtemp[loop_count][step_count][con_count][job_string[cell_count][1:]]=job_string[cell_count+1]
                                        cell_count+=2
                                step_count+=1
                            else:
                                job[job_string[cell_count][1:]]=job_string[cell_count+1]
                                cell_count+=2
                        for loop in jobtemp:
                            if 'steps' not in job:
                                job['steps']=20
                            cycler=0
                            for step in range(int(job['steps'])):
                                if step not in job:
                                    job[step]={}
                                for entry in jobtemp[loop][cycler]:
                                    job[step][entry]=jobtemp[loop][cycler][entry]
                                cycler+=1
                                if cycler>=len(jobtemp[loop]):
                                    cycler=0
                        # To catch any jobs without explicitly stated steps
                        if 'steps' not in job: 
                            job['steps']=20
                        for step in range(int(job['steps'])): 
                            if step not in job:
                                job[step]={}
                        if job!={}:
                            jobs.append(job)
        else:
        #except:
            if engage_mode==1 and len(job_string_inputs)>1:
                display_status("Error parsing job log.  However, the script contains a #restart, so parsing later parts of the script may depend on actions taken during earlier parts.")
            else:
                display_status("Error parsing job log -- please review.") 
            engage_mode=2
        
        # Display the set of parsed jobs if that's all that's been requested
        if engage_mode==1:
            display_status('JOBS PARSED FROM SCRIPT =')
            for jobcount,job in enumerate(raw_job_strings):   
                display_status('JOB '+str(jobcount)+' = '+job)
        elif engage_mode==0 or engage_mode==2:
            #Otherwise, convert the parsed jobs into numerical variables to be used by the script
            #and then (afterwards) proceed with processing
            skipjobs=[]
            for jobcount,job in enumerate(jobs):
                display_job("JOB "+str(jobcount)+":\n"+raw_job_strings[jobcount].strip())
                #status_text.delete("1.0", tk.END) 
                #job_text.update()
                #status_text.update()
                display_status("===================\nJOB "+str(jobcount)+'\n===================')
                #if True:
                try:
                    #Set non-step-specific variables
                    if 'name' in job:
                        filename_root=job['name']
                    else:
                        filename_root=''
                        
                    if 'prename' in job:
                        filename_root=job['prename']+'_'+filename_root
                        
                    if 'postname' in job:
                        filename_root=filename_root+'_'+job['postname']
                        
                    if 'dir' in job:
                        if job['dir'].startswith(work_dir):
                            subfolder=job['dir']
                        else:
                            subfolder=work_dir+'/'+job['dir']
                    else:
                        subfolder=work_dir
                    
                    if 'seed' in job:
                        seed=int(float(job['seed']))
                    else:
                        seed=0
                        
                    if 'seed2' in job:
                        secondary_seed=int(float(job['seed2']))
                    else:
                        secondary_seed=None
                        
                    if 'height' in job:
                        height=int(float(job['height']))
                    else:
                        height=512
                        
                    if 'width' in job:
                        width=int(float(job['width']))
                    else:
                        width=512
                        
                    if 'steps' in job:
                        num_inference_steps=int(float(job['steps']))
                    else:
                        num_inference_steps=20
                        
                    if 'sched' in job:
                        scheduler_number=int(job['sched'])
                    else:
                        scheduler_number=3
                        
                    if 'schmod' in job:
                        scheduler_model=int(job['schmod'])
                    else:
                        scheduler_model=None
                        
                    if 'audio' in job:
                        audio_out=1
                        audio_channels=int(job['audio'])
                    else:
                        audio_out=0
                        audio_channels=2
                        
                    if 'nodecode' in job:
                        nodecode=1
                    else:
                        nodecode=0
                        
                    if 'noimgsave' in job:
                        noimgsave=1
                    else:
                        noimgsave=0
                        
                    if 'mem' in job:
                        models_in_memory=int(job['mem'])
                    else:
                        models_in_memory=1
                        
                    if 'samplerate' in job:
                        sample_rate=int(job['samplerate'])
                    else:
                        sample_rate=44100
                        
                    if 'imtype' in job:
                        imtype=job['imtype']
                    else:
                        imtype='tif'
                        
                    if 'autype' in job:
                        autype=job['autype']
                    else:
                        autype='wav'
                        
                    #Set noise variables
                    arglist=['nx','n+','ncat','nshift','npart']
                    noise_variables=[[[1,1,1,1]],[[0,0,0,0]],[[0]],[[0,0]],[[0,0,0,0,0]],[['',0]]]
                    for argnum,arg in enumerate(arglist):
                        if arg in job:
                            noise_variables[argnum]=num_parse(job[arg])
                    if 'i2i' in job:
                        i2iparse=job['i2i'].split(';')
                        noise_variables[5][0][0]=i2iparse[0].strip()
                        noise_variables[5][0][1]=float(i2iparse[1])
                    
                    #Initialize step-specific variables
                    prompt_variables=[]
                    prompts=[]
                    contrastives=[]
                    neg_prompts=[]
                    neg_prompt_variables=[]
                    guidance_scales=[]
                    model_numbers=[]
                    step_latent_variables=[]
                
                    step=0
                    snshift_check=0
                    while step==0 or step in job: #Cycle through individual steps in the job, or run just once if only one step       
                        #Load arguments specific to this step, which will supersede any universal ones
                        if step in job:
                            step_args=job[step]
                        else:
                            step_args=job
                        
                        #Set prompt and prompt variables
                        
                        #Set model
                        if 'model' in step_args:
                            model=num_parse(step_args['model'])[0][0]
                            model_numbers.append(model)
                        elif 'model' in job:
                            model=num_parse(job['model'])[0][0]
                            model_numbers.append(model)
                        else:
                            model_numbers.append(0)
                        
                        #Set prompt
                        if 'prompt' in step_args:
                            prompt=step_args['prompt']
                        elif 'prompt' in job:
                            prompt=job['prompt']
                        else:
                            prompt=''
                            
                        if '2prompt' in step_args:
                            prompt=[prompt,job['2prompt']]
                        elif '2prompt' in job:
                            prompt=[prompt,job['2prompt']]
                
                        if '3prompt' in step_args:
                            try:
                                prompt.append(job['3prompt'])
                            except:
                                prompt=[prompt,prompt,job['3prompt']]
                        elif '3prompt' in job:
                            try:
                                prompt.append(job['3prompt'])
                            except:
                                prompt=[prompt,prompt,job['3prompt']]
                            
                        #Make any specified model-specific adjustment to the prompt
                        if model_prompts[model_numbers[-1]]!='':
                            prompt=model_prompts[model_numbers[-1]].replace('*',prompt)
                            
                        #Set prompt variables    
                        prompt_variables=prompt_defaults.copy()
                        arglist=['raw+','proc+','*','pad+','dyna-pad','avg-pad','padx','posx','rawx','procx','endtok']
                        for argnum,arg in enumerate(arglist):
                            if arg in step_args:
                                prompt_variables[argnum]=num_parse(step_args[arg])
                            elif arg in job:
                                prompt_variables[argnum]=num_parse(job[arg])
                        #Add prompt and prompt variables for this step to the list  
                        if '2copy' in step_args or '2copy' in job:
                            prompt_variables2=prompt_variables
                        else:
                            prompt_variables2=prompt_defaults.copy()
                            arglist2=['2raw+','2proc+','2*','2pad+','2dyna-pad','2avg-pad','2padx','2posx','2rawx','2procx','2endtok']
                            for argnum,arg in enumerate(arglist2):
                                if arg in step_args:
                                    prompt_variables2[argnum]=num_parse(step_args[arg])
                                elif arg in job:
                                    prompt_variables2[argnum]=num_parse(job[arg])
                        prompts.append([prompt,[prompt_variables,prompt_variables2],1])
                        
                        #Set negative prompt and negative prompt variables
                        if 'neg_prompt' in step_args:
                            neg_prompt=step_args['neg_prompt']
                        elif 'neg_prompt' in job:
                            neg_prompt=job['neg_prompt']
                        else:
                            neg_prompt=''
                        if '2neg_prompt' in step_args:
                            neg_prompt=[neg_prompt,step_args['2neg_prompt']]
                        elif '2neg_prompt' in job:
                            neg_prompt=[neg_prompt,job['2neg_prompt']]
                        if '3neg_prompt' in step_args:
                            try:
                                neg_prompt.append(step_args['3neg_prompt'])
                            except:
                                neg_prompt=[neg_prompt,neg_prompt,step_args['3neg_prompt']]
                        elif '3neg_prompt' in job:
                            try:
                                neg_prompt.append(job['3neg_prompt'])       
                            except:
                                neg_prompt=[neg_prompt,neg_prompt,step_args['3neg_prompt']]
                               
                        neg_prompt_variables=prompt_defaults.copy()
                        neg_arglist=['neg_raw+','neg_proc+','neg_*','neg_pad+','neg_dyna-pad','neg_avg-pad','neg_padx','neg_posx','neg_rawx','neg_procx','neg_endtok']
                        for argnum,arg in enumerate(neg_arglist):
                            if arg in step_args:
                                neg_prompt_variables[argnum]=num_parse(step_args[arg])
                            elif arg in job:
                                neg_prompt_variables[argnum]=num_parse(job[arg])
                        if '2neg_copy' in neg_arglist:
                            neg_prompt_variables2=neg_prompt_variables
                        else:
                            neg_prompt_variables2=prompt_defaults.copy()
                            neg_arglist2=['2neg_raw+','2neg_proc+','2neg_*','2neg_pad+','2neg_dyna-pad','2neg_avg-pad','2neg_padx','2neg_posx','2neg_rawx','2neg_procx','2neg_endtok']
                            for argnum,arg in enumerate(neg_arglist2):
                                if arg in step_args:
                                    neg_prompt_variables2[argnum]=num_parse(step_args[arg])
                                elif arg in job:
                                    neg_prompt_variables2[argnum]=num_parse(job[arg])
                                    
                        if 'negx' in step_args:
                            negx=num_parse(step_args['negx'])[0][0]
                        elif 'negx' in job:
                            negx=num_parse(job['negx'])[0][0]
                        else:
                            negx=1
                        neg_prompts.append([neg_prompt,[neg_prompt_variables,neg_prompt_variables2],negx])
                        
                        #Set guidance scales
                        if 'guid' in step_args:
                            guidance=num_parse(step_args['guid'])[0][0]
                            guidance_scales.append(guidance)
                        elif 'guid' in job:
                            guidance=num_parse(job['guid'])[0][0]
                            guidance_scales.append(guidance)
                        else:
                            guidance_scales.append(9)
                            
                        #Set step-by-step latent adjustments
                        arglist_lat=['snx','sn+','sncat','snshift','snpart']
                        step_latent_variable=[[[1,1,1,1]],[[0,0,0,0]],[[0]],[[0,0]],[[0,0,0,0,0]]]
                        for argnum,arg in enumerate(arglist_lat):
                            if arg in step_args:
                                step_latent_variable[argnum]=num_parse(step_args[arg])
                            elif arg in job:
                                step_latent_variable[argnum]=num_parse(job[arg])
                        step_latent_variables.append(step_latent_variable)
                        if 'snshift' in job or 'snshift' in step_args:
                            snshift_check=1
                            
                        # Get contrastives        
                        contrastive_step=[[[['',prompt_defaults.copy()],0]],[[['',prompt_defaults.copy()],['',prompt_defaults.copy()],0]]]
                        con_counter=1
                        
                        while con_counter in step_args:
                            contrastive=step_args[con_counter]
                            if 'prompt' in contrastive:
                                con_prompt=contrastive['prompt']
                            else:
                                con_prompt=''    
                            if '2prompt' in contrastive:
                                con_prompt=[con_prompt,contrastive['2prompt']]
                            if '3prompt' in contrastive:
                                try:
                                    con_prompt.append(job['3prompt'])
                                except:
                                    con_prompt=[con_prompt,con_prompt,contrastive['3prompt']]
                            con_prompt_variables=prompt_defaults.copy()
                            for argnum,arg in enumerate(arglist):
                                if arg in contrastive:
                                    con_prompt_variables[argnum]=num_parse(contrastive[arg])
                            if '2copy' in contrastive:
                                con_prompt_variables2=con_prompt_variables.copy()
                            else:
                                con_prompt_variables2=prompt_defaults.copy()
                                for argnum,arg in enumerate(arglist2):
                                    if arg in contrastive:
                                        con_prompt_variables2[argnum]=num_parse(contrastive[arg])
                            con_prompts=[con_prompt,[con_prompt_variables,con_prompt_variables2]]
                            if 'with' in contrastive:
                                if contrastive_step[1][0]==[['',prompt_defaults],['',prompt_defaults],0]:
                                    #If the first entry is empty / default, remove it (for replacement)
                                    contrastive_step[1]=[]
                                if 'prompt' in contrastive['with']:
                                    with_prompt=contrastive['with']['prompt']
                                else:
                                    with_prompt=''
                                if '2prompt' in contrastive['with']:
                                    with_prompt=[with_prompt,contrastive['with']['2prompt']]
                                if '3prompt' in contrastive['with']:
                                    try:
                                        with_prompt.append(contrastive['with']['3prompt'])
                                    except:
                                        with_prompt=[with_prompt,with_prompt,contrastive['with']['3prompt']]    
                                with_prompt_variables=prompt_defaults.copy()
                                for argnum,arg in enumerate(arglist):
                                    if arg in contrastive['with']:
                                        with_prompt_variables[argnum]=num_parse(contrastive['with'][arg])
                                if '2copy' in contrastive['with']:
                                    with_prompt_variables2=with_prompt_variables
                                else:
                                    with_prompt_variables2=prompt_defaults.copy()
                                    for argnum,arg in enumerate(arglist2):
                                        if arg in contrastive['with']:
                                            con_prompt_variables2[argnum]=num_parse(contrastive['with'][arg])
                                with_prompts=[with_prompt,[with_prompt_variables,with_prompt_variables2]]
                                if 'by' in contrastive['with']:
                                    con_by=num_parse(contrastive['with']['by'])[0][0]
                                elif 'by' in contrastive:  #dispreferred notation
                                    con_by=num_parse(contrastive['by'])[0][0]
                                else:
                                    con_by=1
                                contrastive_step[1].append([con_prompts,with_prompts,con_by])
                            else:
                                if contrastive_step[0][0]==[['',prompt_defaults],0]:
                                    #If the first entry is empty / default, remove it (for replacement)
                                    contrastive_step[0]=[]
                                if 'by' in contrastive:
                                    con_by=num_parse(contrastive['by'])[0][0]
                                else:
                                    con_by=1
                                contrastive_step[0].append([con_prompts,con_by])
                            con_counter+=1
                        #Append this step's contrastives to the multi-step contrastive array
                        contrastives.append(contrastive_step)
                        step+=1
                    if 'shiftback' in job[0]:
                        shiftback=int(job[0]['shiftback'])
                    elif snshift_check==1:
                        shiftback=1
                    else:
                        shiftback=0   
                #else:
                except:
                    display_status("Error parsing job variables for JOB "+str(jobcount)+" -- skipping.")
                    skipjobs.append(jobcount)
                    
                if jobcount not in skipjobs and continue_script==1 and jobcount>=resume_point:
                    #if True:   
                    try:
                        # Intercept concatenation jobs that don't involve SD inference
                        if 'wavcat' in job:
                            wavcat_directory=job['wavcat']
                            vae_option=model_ids[0]
                            if 'chunk' in job:
                                chunk_size=int(job['chunk'])
                            else:
                                chunk_size=100 #this is a better default than 50
                            if 'overlap' in job:
                                overlap_size=int(job['overlap'])
                            else:
                                overlap_size=10
                            if 'overlap_type' in job:
                                overlap_type=int(job['overlap_type'])
                            else:
                                overlap_type=1
                            if 'windur' in job:
                                win_dur=int(job['windur'])
                            else:
                                win_dur=100
                            if torch_device=='cuda':
                                # To clear CUDA memory
                                models=torch.zeros(1).to('cpu')
                                torch.cuda.empty_cache()
                                models_in_memory=0
                                old_model_numbers=-1
                            wavcat(wavcat_directory,vae_option,chunk_size,overlap_size,overlap_type,sample_rate,win_dur,subfolder,audio_channels,autype,raw_job_strings[jobcount])
                        elif 'wavprep' in job:
                            wavprep_source=job['wavprep']
                            if 'est_dur' in job:
                                estimatedDuration=num_parse(job['est_dur'])[0][0]
                            else:
                                estimatedDuration=5
                            if 'test_margin' in job:
                                testMargin=num_parse(job['test_margin'])[0][0]
                            else:
                                testMargin=0.7
                            if 'reverse' in job:
                                reverse=1
                            else:
                                reverse=0
                            if 'norm' in job:
                                normalize=num_parse(job['norm'])[0][0]
                            else:
                                normalize=7.7
                            if 'keep_width' in job:
                                rewidth=0
                            else:
                                rewidth=1
                            if 'def_width' in job:
                                defwidth=int(job['def_width'])
                            else:
                                defwidth=0
                            if 'freqblur' in job:
                                freqblur=num_parse(job['freqblur'])[0][0]
                            else:
                                freqblur=None
                            if 'double' in job:
                                double=1
                            else:
                                double=0
                            prep_audio_input(wavprep_source,subfolder,estimatedDuration,testMargin,reverse,normalize,rewidth,defwidth,freqblur,double,imtype)
                        elif 'makevid' in job:
                            makevid=job['makevid']
                            make_video(makevid,raw_job_strings[jobcount],job)
                        # Handle jobs that do involve SD inference
                        else:
                            cumulative_shift=[0,0]
                            if 'vae' in job:
                                vae_number=int(job['vae'])
                            else:
                                vae_number=model_numbers[-1] #if not specified, use the VAE of the model used in the final step
                            
                            if model_numbers!=old_model_numbers:
                                models=prepare_model_data(model_numbers,model_ids,model_prompts,models_in_memory)
                            else:
                                display_status('Using same model arrangement as last job')
                            if scheduler_number!=old_scheduler_number:
                                scheduler=set_scheduler(scheduler_number,model_numbers[0],model_ids,scheduler_model)
                            if vae_number!=old_vae_number:
                                display_status("Using VAE from model "+str(vae_number))
                                vae=AutoencoderKL.from_pretrained(model_ids[vae_number],subfolder="vae")
                                vae = vae.to(torch_device)
                            scheduler.set_timesteps(num_inference_steps)
                            
                            #Generate image based on input arguments
                            latent=generate_noise_latent(seed,secondary_seed,height,width,noise_variables,scheduler,vae,num_inference_steps)
                            latent=generate_image_latent(latent,scheduler,num_inference_steps,guidance_scales,models,prompts,neg_prompts,step_latent_variables,noise_variables[5][0],contrastives,models_in_memory,models_in_memory_old,model_ids,shiftback)
                            display_status('Inference complete.')
                            filename=filename_root+'_'+get_datestring()
                            if subfolder !=None:
                                if not os.path.exists(subfolder):
                                    os.makedirs(subfolder)
                                filename=subfolder+'/'+filename
                            display_status('Saving latent and metadata package.')
                            package=[latent,raw_job_strings[jobcount],job,__file__]
                            torch.save(package,filename+'.pt')
                            #Delete model data to free up memory
                            display_status('Generating image from latent.')
                            if nodecode==0:
                                image=generate_image_from_latent(vae,latent)
                                refresh_image_display(image)
                                if noimgsave==0:
                                    display_status('Saving image.')
                                    image.save(filename+'.'+imtype)
                                    display_caption(filename+'.'+imtype)
                                else:
                                    display_caption(filename+'.pt')
                            else:
                                display_caption(filename+'.pt\nNOT DISPLAYED')
                            if audio_out==1:
                                if nodecode==0:
                                    display_status('Generating audio.')
                                    spectrophone(image,filename,audio_channels,sample_rate,autype)
                                else:
                                    display_status("Can't generate audio because image latent not decoded.")
                            display_status('Logging job.')
                            if not os.path.exists(path_to_csv_log):
                                csvfile=open(path_to_csv_log, mode='w', newline='')
                                csv_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                                csv_writer.writerow(['filename','job'])
                                csvfile.close()
                            csvfile=open(path_to_csv_log, mode='a', newline='')
                            csv_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                            csv_values=[filename,
                                        raw_job_strings[jobcount],
                                        seed,
                                        secondary_seed,
                                        model_numbers,
                                        neg_prompts,
                                        height,
                                        width,
                                        scheduler_number,
                                        num_inference_steps,
                                        guidance_scales]
                            csv_writer.writerow(csv_values)
                            csvfile.close()
                            old_scheduler=scheduler
                            old_model_numbers=model_numbers
                            cumulative_shift=[0,0]
                            models_in_memory_old=models_in_memory
                            file=open(path_to_resume_point,'w')
                            file.write(str(jobcount+1))
                            file.close()
                    #else:
                    except:
                        display_status("Error running JOB "+str(jobcount)+" -- skipping")
                elif continue_script==0:
                    display_status("Script stopped by request.  Click 'Resume Script' to resume with the next job number.")
                    resume_point=jobcount+1
                    break
                else:
                    display_status("JOB "+str(jobcount)+" skipped.") 

def get_datestring():
    datestring=str(datetime.datetime.now())
    datestring=datestring.replace(":","-")
    datestring=datestring.replace(".","-")
    datestring=datestring.replace(" ","-")
    return datestring

def save_script():
    to_save=1
    if os.path.exists(path_to_script):
        file=open(path_to_script,'r')
        current_text=file.read()
        file.close()
        if current_text.strip()==script_text.get(1.0,tk.END).strip():
            display_caption('Current saved script is up to date.')
            to_save=0
        else:
            os.rename(path_to_script,path_to_backups+script_backup_prefix+get_datestring()+'.txt')
    if to_save==1:
        file=open(path_to_script,'w')
        file.write(script_text.get(1.0,tk.END).strip())
        file.close()
        if os.path.exists(path_to_resume_point):
            os.remove(path_to_resume_point)
    
def revert_script():
    file=open(path_to_script,'r')
    script_text.delete(1.0, tk.END)
    script_text.insert(1.0, file.read())
    file.close()
    
def refresh_image_display(image):
    image=ImageTk.PhotoImage(image)
    image_label.config(image=image)
    image_label.photo_ref=image
    image_label.update()
    
def display_job(job):
    job_text.delete("1.0", tk.END)
    job_text.insert("1.0", job)
    job_text.update()
    
def display_status(status):
    status_text.insert(tk.END,status+'\n')
    status_text.update()
    status_text.see("end")
    
def display_caption(caption):
    caption_text.delete("1.0", tk.END)
    caption_text.insert("1.0", caption)
    caption_text.update()
    
def query_folder():
    d=fd.askdirectory()
    if d=='':
        display_caption("Cancelled folder query.")
    else:
        file=open(d+'/folder_query.txt','w')
        file.write('Contents of '+d+' as of '+get_datestring()+'\n\n')
        for x in os.listdir(d):
            if x.endswith('.pt'):
                file.write(x+'\n')
                display_status(x)
                try:
                    package=torch.load(d+'/'+x)
                    file.write(package[1].strip()+'\n')
                    display_status(package[1].strip())
                except:
                    display_status(package[1].strip())
                    file.write('---\n')
                    display_status("Image query failed -- couldn't access information in SyDNEy .pt format.")
                file.write('\n')
                display_status('')
        file.close()
    
def query_image():
    x=fd.askopenfilename(filetypes=[("SyDNEy .pt files", "*.pt")])
    if x=='':
        display_caption("Cancelled image query.")
    else:
        try:
            package=torch.load(x)
            display_caption(package[1].strip())
        except:
            display_caption("Image query failed -- couldn't access information in SyDNEy .pt format.")
        
    
def stop_script():
    global continue_script
    continue_script=0
    
def load_backup():
    if os.path.exists(path_to_backups):
        x=fd.askopenfilename(filetypes=[("Script .txt files", "*.txt")],initialdir=path_to_backups)
        file=open(x)
        script_text.delete(1.0, tk.END)
        script_text.insert(1.0, file.read())
        file.close()
        
def list_models():
    try:
        with open(path_to_config, 'r') as file:
            configuration = file.read().replace('\n', '')
            display_status("Loading configuration from sydney_config.txt")
    except:
        display_status("No sydney_config.txt file found, using default configuration.")
        configuration=''
    if configuration!='':        
        config_parts=configuration.split('#')
        for config_part in config_parts:
            if config_part.startswith('models'):
                display_status("Creating model index....")
                config_part=config_part[7:]
                model_strings=config_part.split(';')
                model_counter=0
                for model_string in model_strings:
                    model_string=model_string.split(',')
                    if len(model_string)==1:
                        display_status("Model "+str(model_counter)+": "+model_string[0].strip())
                    elif len(model_string)==2:
                        display_status("Model "+str(model_counter)+": "+model_string[0].strip()+" +prompts: "+model_string[1].strip())
                    model_counter+=1
                    
def list_schedulers():
    scheduler_list=['PNDMScheduler','DDIMScheduler','LMSDiscreteScheduler','EulerDiscreteScheduler','EulerAncestralDiscreteScheduler','DPMSolverMultistepScheduler','DDPMScheduler','KDPM2DiscreteScheduler','DPMSolverSinglestepScheduler','DEISMultistepScheduler','UniPCMultistepScheduler','HeunDiscreteScheduler','KDPM2AncestralDiscreteScheduler','KDPM2DiscreteScheduler','DPMSolverSDEScheduler']
    for sched_count,scheduler_name in enumerate(scheduler_list):
        display_status('Scheduler '+str(sched_count)+': '+scheduler_name)
    
def popup_input(title,text):
    temp_window = tk.Tk()
    temp_window.withdraw()
    input_text = sd.askstring(title,text,parent=temp_window)
    temp_window.destroy()
    return input_text

def add_model():
    model_to_add=popup_input('Add Model','Enter model ID, for example: dreamlike-art/dreamlike-photoreal-2.0')
    prompt_addition=popup_input('Prompt Addition','Optionally enter text to add to prompts, with an asterisk (*) representing the position of the prompt; for example, to add "mdjrny-v4 style" at the end of each prompt: * mdjrny-v4 style')
    if prompt_addition!='':
        model_to_add+=','+prompt_addition
    if model_to_add!='':
        configuration=''
        new_configuration=''
        try:
            with open(path_to_config,'r') as file:
                configuration=file.read()
            if configuration.endswith('\n'):
                configuration=configuration[:-1]
            if configuration!='':        
                config_parts=configuration.split('#')
                for config_part in config_parts:
                    if config_part.startswith('models'):
                        config_part+=';\n'+model_to_add
                    if config_part!='':
                        new_configuration+='#'+config_part
        except:
            display_status("No configuration file found; creating new one.")
        if new_configuration=='':
            new_configuration='#models '+model_to_add
        os.rename(path_to_config,ref_dir+config_backup_prefix+get_datestring()+'.txt')
        with open(path_to_config,'w') as file:
            file.write(new_configuration)
            file.close()
    list_models()
    
# MAIN FUNCTION WITH GUI
prompt_defaults=[[[0,0,0,0,0]],[[0,0,0,0,0]],[[0,0,0]],[[0,0]],[[0]],[[0]],[[1]],[[1]],[[1,0,0,0,0]],[[1,0,0,0,0]],[[0]]]
old_vae_number=[-1]
continue_script=1
cumulative_shift=0
    
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
script_text = tk.Text(frame, bg="white", fg="green", height=10, width=100,yscrollcommand=scrollbar.set)
scrollbar.config(command=script_text.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
script_text.pack(side="left")
try:
    file=open(path_to_script,'r')
    sydney_script = file.read()
    file.close()
except:
    sydney_script = 'SCRIPT WINDOW.  Compose a script here.'
script_text.insert(tk.END, sydney_script)

#create image display
image_label=tk.Label(root)
image_label.place(x=950,y=55)

#create text displays
job_text=tk.Text(root, bg="gray35", fg="white",height=5,width=100)
job_text.place(x=10,y=180)

frame2 = tk.Frame(root)
frame2.place(x=10,y=280)
scrollbar2 = tk.Scrollbar(frame2)
status_text=tk.Text(frame2, bg="black", fg="white",height=20,width=100,yscrollcommand=scrollbar2.set)
scrollbar2.config(command=status_text.yview)
scrollbar2.pack(side=tk.RIGHT, fill=tk.Y)
status_text.pack(side="left")

# Report any missing dependencies
if missing_dependencies!=[]:
    display_status("Didn't find the following libraries required for basic functionality:")
    for dependency in missing_dependencies:
        display_status(" "+dependency)
else:
    display_status("All libraries required for basic functionality found.")
if missing_dependencies_audio!=[]:
    display_status("Didn't find the following libraries required for audio output:")
    for dependency in missing_dependencies_audio:
        display_status(" "+dependency)
else:
    display_status("All libraries required for audio output found.")
if missing_dependencies_audio!=[]:
    display_status("Didn't find the following libraries required for video output:")
    for dependency in missing_dependencies_video:
        display_status(" "+dependency)
else:
    display_status("All libraries required for video output found.")

caption_text=tk.Text(root, bg="gray35", fg="white",height=3,width=60)
caption_text.place(x=950,y=0)

# create buttons
run_script_button = tk.Button(root, text="Run Script", command=run_script)
run_script_button.place(x=835,y=0)
save_script_button = tk.Button(root, text="Save Script", command=save_script)
save_script_button.place(x=835,y=30)
revert_script_button = tk.Button(root, text="Revert to Saved", command=revert_script)
revert_script_button.place(x=835,y=60)
load_backup_button = tk.Button(root, text="Load Backup", command=load_backup)
load_backup_button.place(x=835,y=90)
stop_script_button = tk.Button(root, text="Stop Script", command=stop_script)
stop_script_button.place(x=835,y=120)
query_image_button = tk.Button(root, text="Query Image", command=query_image)
query_image_button.place(x=835,y=150)
query_folder_button = tk.Button(root, text="Query Folder", command=query_folder)
query_folder_button.place(x=835,y=180)
list_models_button = tk.Button(root, text="List Models", command=list_models)
list_models_button.place(x=835,y=210)
add_model_button = tk.Button(root, text="Add Model", command=add_model)
add_model_button.place(x=835,y=240)
list_schedulers_button = tk.Button(root, text="List Schedulers", command=list_schedulers)
list_schedulers_button.place(x=835,y=270)
parse_script_button = tk.Button(root, text="Parse Script", command=parse_script)
parse_script_button.place(x=835,y=310)
resume_script_button = tk.Button(root, text="Resume Script", command=resume_script)
resume_script_button.place(x=835,y=340)

tk.mainloop()
