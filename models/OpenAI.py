from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.callbacks import get_openai_callback

from models.Model import ModelInterface

import sys
sys.path.append('..')

import status

class GPTModel(ModelInterface):
    def __init__(self, model_name, temperature):
        if "ft:gpt" in model_name:
            self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        elif model_name != "":
            self.llm = OpenAI(model_name=model_name, temperature=temperature)
        else:
            # using default OpenAI language model, since it is stable (avaliable for more time) and cheaper then GPT4
            self.llm = OpenAI(temperature=temperature) 
        template = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know. 
Current conversation: 
{history}
Human: {human_message}
AI: """
        self.prompt = PromptTemplate(input_variables=["human_message", "history"], template=template)
    
    def __call__(self, human_message):
        with get_openai_callback() as cb:
            response = self.llm(self.prompt.format(human_message=human_message))
            status.total_tokens_used = status.total_tokens_used + cb.total_tokens
            status.total_cost = status.total_cost + cb.total_cost
        return response
    
    def get_llm(self):
        return self.llm
    
    def get_prompt(self):
        return self.prompt
    
    def set_prompt(self, template, input_variables):
        template = template
        self.prompt = PromptTemplate(input_variables=input_variables, template=template)