import transformers
import torch

from langchain import PromptTemplate
from langchain.llms import HuggingFacePipeline

from models.Model import ModelInterface

import sys
sys.path.append('..')

import status

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
B_S, E_S = "<s>", "</s>"

class Llama2Model(ModelInterface):
    def __init__(self, model_name, temperature):
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model_config = transformers.AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            config=model_config,
            quantization_config=bnb_config,
            device_map='auto',
            cache_dir=status.cache_dir
        )
        
        self.streamer = transformers.TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        generate_text = transformers.pipeline(
            model=model,
            tokenizer=tokenizer,
            return_full_text=True, 
            task='text-generation',
            temperature=temperature, 
            max_new_tokens=4096,  # max number of tokens to generate in the output
            repetition_penalty=1.1, 
            device_map = "auto",
            do_sample = True,
            streamer=self.streamer
        )

        self.llm = HuggingFacePipeline(pipeline=generate_text)
        
        global B_INST, E_INST, B_SYS, E_SYS, B_S, E_S
        system_prompt_variable = "You are a helpful assistant that provides accurate and concise responses"
        user_prompt_variable = '{question}'
        history_prompt_variable = '{chat_history}'
        
        template = f"{B_S}{B_INST} {B_SYS}{system_prompt_variable}{E_SYS}{history_prompt_variable}{user_prompt_variable} {E_INST}"

        self.prompt = PromptTemplate(input_variables=["human_message", "history"], template=template)
    
    def get_llm(self):
        return self.llm
    
    def get_prompt(self):
        return self.prompt
    
    def get_streamer(self):
        return self.streamer
    
    def set_prompt(self, template, input_variables):
        template = template
        self.prompt = PromptTemplate(input_variables=input_variables, template=template)
        