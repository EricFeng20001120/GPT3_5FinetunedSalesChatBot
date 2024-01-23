from abc import ABC, abstractmethod

class ModelInterface(ABC):
    
    @abstractmethod
    def get_llm(self):
        pass
    
    @abstractmethod
    def get_prompt(self):
        pass
    
    @abstractmethod
    def set_prompt(self, template, input_variables):
        pass