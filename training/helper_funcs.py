from models.OpenAI import GPTModel
from models.Llama2 import Llama2Model

from langchain.callbacks import get_openai_callback
from langchain import LLMChain
from langchain.chains import ConversationalRetrievalChain
from langchain import PromptTemplate

from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import LLMChainFilter
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers import ContextualCompressionRetriever
import langchain.llms

import time
import multiprocessing

import status
import helper_funcs_retrieval

# debug function
def pretty_print_docs(docs):
    print(f"\n{'-' * 100}\n".join([f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]))

history_parsor = None
LLMchain = None
ConversationalRetrievalchain = None
model_type = None
customer_msg_req_lookup = {}
budget = -1
product_info = "Cannot be specified as the budget is not given. "

# special tokens from llama 2
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
B_S, E_S = "<s>", "</s>"

def create_prompt_template(model_name, system_msg):
    global B_INST, E_INST, B_SYS, E_SYS, B_S, E_S
    global model_type
    if 'gpt' in model_name or 'davinci' in model_name or 'ada' in model_name:
        model_type = "gpt"
        template = system_msg + """
Here are the products, and be sure you first recommend the higher-performance one:
{product_info}
Here is the information you know:
{document}
{chat_history}
###Customer: {question}
###Sales: """
        prompt = PromptTemplate(input_variables=["product_info", "document", "question", "chat_history"], template=template)
        return prompt
    elif 'Llama-2' in model_name:
        model_type = "Llama2"
        product_prompt_variable = "\nHere are the products, and be sure you first recommend the higher-performance one:\n{product_info}"
        document_prompt_variable = "\nHere is the information you know:\n{document}"
        user_prompt_variable = "{question}"
        history_prompt_variable = "{chat_history}"
        
        template = f"{B_S}{B_INST} {B_SYS}{system_msg}{product_prompt_variable}{document_prompt_variable}{E_SYS}{history_prompt_variable}{user_prompt_variable} {E_INST}"
        prompt = PromptTemplate(input_variables=["product_info", "document", "question", "chat_history"], template=template)
        return prompt
    else:
        raise ValueError("Invalid model type")

def create_model(model_name, temperature):
    global model_type
    if 'gpt' in model_name or 'davinci' in model_name or 'ada' in model_name:
        model_type = "gpt"
        return GPTModel(model_name, temperature)
    elif 'Llama-2' in model_name:
        model_type = "Llama2"
        return Llama2Model(model_name, temperature)
    else:
        raise ValueError("Invalid model type")
        
def create_LLMchain(model, human_prefix, AI_prefix, prompt, verbose=False):
    global LLMchain
    LLMchain = LLMChain(
        llm=model.get_llm(), 
        prompt=prompt, 
        verbose=verbose
    )
    # initialize the history parsor
    global history_parsor
    if history_parsor is None:
        history_parsor = my_history_parsor(human_prefix, AI_prefix)
    return LLMchain

def LLMchain_run(product_info, document, human_message, history):
    global LLMchain
    if model_type == "gpt":
        with get_openai_callback() as cb:
            response = LLMchain.predict(product_info=product_info, document=document, question=human_message, chat_history=history_parsor.parse_history_str(history))
            status.total_tokens_used = status.total_tokens_used + cb.total_tokens
            status.total_cost = status.total_cost + cb.total_cost
    elif model_type == "Llama2":
        response = LLMchain.predict(product_info=product_info, document=document, question=human_message, chat_history=history_parsor.parse_history_str(history))
    else:
        raise ValueError("Invalid model type")
    return response

def data_retreve(vectorstore, human_message, history):
    # check budget - powered by gpt-3.5
    recheved_budget = helper_funcs_retrieval.check_budget(human_message)
    global budget
    if recheved_budget != -1:
        budget = recheved_budget
        # manully remove the 'within' and 'under' from the message if it is about budget
        human_message = human_message.replace("within", "of").replace("under", "of")
    # budget will be needed for product retrieve
    if budget != -1:
        budget_str = " Budget = {} ".format(int(budget))
    else:
        budget_str = ""
    
    # user query reformulation powered by gpt-3.5
    reformulated_human_message = helper_funcs_retrieval.get_reformated_customer_question(history_parsor.parse_history_str(history, conversation=True), human_message)
    print("reformulated_human_message = \n{}\n".format(reformulated_human_message))
    if 'budget' in reformulated_human_message:
		# manully remove the 'within' and 'under' from the message if it is about budget
        reformulated_human_message = reformulated_human_message.replace("within", "of").replace("under", "of")
    
    # product recheve - powered by gpt-3.5 and custom function, if failed, will use gpt-4 panda agent
    if budget_str != "":
        global product_info
        if product_info == "Cannot be specified as the budget is not given. ":
            retrieved_product_info = helper_funcs_retrieval.product_retrieve_gpt3_5(reformulated_human_message + budget_str, False)
            if retrieved_product_info != "error":
                product_info = retrieved_product_info
        else:
            bool_new_product_requirement = helper_funcs_retrieval.check_product_requirement(reformulated_human_message)
            print("bool_new_product_requirement = {}".format(bool_new_product_requirement))
            if bool_new_product_requirement:
                retrieved_product_info = helper_funcs_retrieval.product_retrieve_gpt3_5(reformulated_human_message + budget_str, False)
                if retrieved_product_info != "error":
                    product_info = retrieved_product_info
            else:
                pass # not changing the retrieved product information from before
        print("product_info = \n{}".format(product_info))
    
    # data recheving and docs filtering - powered by gpt-3.5
    redundant_filter = EmbeddingsRedundantFilter(embeddings=HuggingFaceBgeEmbeddings())
    # llm_chain_filter = LLMChainFilter.from_llm(langchain.llms.OpenAI(temperature=0)) # gpt-3.5
    # llm_chain_compressor = LLMChainExtractor.from_llm(langchain.llms.OpenAI(temperature=0))
    pipeline_compressor = DocumentCompressorPipeline(
        # transformers=[redundant_filter, llm_chain_filter]
        transformers=[redundant_filter]
    )
    compression_retriever = ContextualCompressionRetriever(base_compressor=pipeline_compressor, base_retriever=vectorstore.get_retriever())
    docs = vectorstore.get_ald_documents() + compression_retriever.get_relevant_documents(reformulated_human_message)
    
    print("Done redundant filtering, doing llm filtering...")
    # doing llm_chain_filter manully since langchain llm_chain_filter does not support chat models
    for doc in docs:
        doc_relavent = helper_funcs_retrieval.check_relavent(reformulated_human_message, doc.page_content)
        if not doc_relavent:
            docs.remove(doc)
        
    # convert recheved docs into string
    related_info = "===\n"
    sources = []
    for doc in docs:
        related_info = related_info + doc.page_content + "\n\n"
        if doc.metadata['source'] not in sources:
            sources.append(doc.metadata['source'])
    related_info = related_info + "===\n"
    print("related_info = \n{}\n".format(related_info))
    
    return product_info, related_info, sources

# if the database is empty, run LLMchain, otherwise run Retrievalchain
def chain_run(vectorstore, human_message, history, print_source):
    if vectorstore.get_vectorstore() is not None:
        product_info, related_info, sources = data_retreve(vectorstore, human_message, history)

        if product_info == "":
            response = LLMchain_run("Sorry, we do not have a product that meets all of your requirements. ", related_info, human_message, history)
        else:
            response = LLMchain_run(product_info, related_info, human_message, history)
        # response = "Test message output"
        if print_source:
            return "{}\nSources:\n{}".format(response, sources)
        else:
            return response
        # return "Retrievalchain is talking"
    else:
        response = LLMchain_run(product_info, "", human_message, history)
        return response
        # return "LLMchain is talking"

# this class parse the history matained by the gradio
# memory from langchain was not used since it does not support (or we cannot 
# find out the way to) removing portion of the history which will be needed for responce re-generation and undo
class my_history_parsor():
    def __init__(self, human_prefix, AI_prefix):
        self.size = status.memory_size
        self.human_prefix = human_prefix
        self.AI_prefix = AI_prefix
    
    # if conversation=True, the output will be in form of conversation reguardless of the model type
    def parse_history_str(self, history, conversation=False):
        history_size = len(history)
        if history_size >= self.size:
            start = -self.size
        else:
            start = -history_size
            
        parsed_history = ""
        global model_type
        # prompt format for GPT and llama are different
        if model_type == "gpt" or conversation:
            for i in range(start, 0): # for gpt model, history updated after responce, so loop to i=-1 will parse all history
                human_message = self.human_prefix + ": " + history[i][0] + "\n"
                parsed_history = parsed_history + human_message
                AI_message = self.AI_prefix + ": " + history[i][1] + "\n"
                parsed_history = parsed_history + AI_message
            parsed_history = parsed_history[:-1]
            return parsed_history
        elif model_type == "Llama2": # Llama2 uses streamer, history updated before responce, so loop to i=-2 will ignore the unfinshed responce
            global B_INST, E_INST, B_SYS, E_SYS, B_S, E_S
            for i in range(start, -1):
                pair = history[i]
                parsed_history = parsed_history + pair[0] + E_INST
                parsed_history = parsed_history + pair[1] + E_S + B_S + B_INST
            return parsed_history
        else:
            return ""
    
    def parse_history_list(self, history):
        history_size = len(history)
        if history_size >= self.size:
            start = -self.size
        else:
            start = -history_size
            
        parsed_history = []
        for i in range(start, 0):
            parsed_history.append((history[i][0], history[i][1]))
        return parsed_history

# def run_check_budget(customer_msg, conn):
#     result = helper_funcs_retrieval.check_budget(customer_msg)
#     conn.send(result)
#     conn.close()

# def run_get_reformated_customer_question(history_msg, customer_msg, conn):
#     result = helper_funcs_retrieval.get_reformated_customer_question(history_msg, customer_msg)
#     conn.send(result)
#     conn.close()

# def run_product_retrieve_gpt3_5(customer_msg, get_comment, conn):
#     result = helper_funcs_retrieval.product_retrieve_gpt3_5(customer_msg, get_comment)
#     conn.send(result)
#     conn.close()

# def run_check_product_requirement(history, customer_msg, customer_msg_req_lookup, conn):
#     result = helper_funcs_retrieval.check_product_requirement(history, customer_msg, customer_msg_req_lookup)
#     conn.send(result)
#     conn.close()

# def run_check_relavent(customer_msg, doc_info, conn):
#     result = helper_funcs_retrieval.check_relavent(customer_msg, doc_info)
#     conn.send(result)
#     conn.close()

# # even slower using multiprocessing, not using it
# def fork_join_all(target_list, args_list, task_name, timeout):
#     parent_conn_list = []
#     child_conn_list = []
#     process_list = []
#     for i in range(len(target_list)):
#         parent_conn, child_conn = multiprocessing.Pipe()
#         parent_conn_list.append(parent_conn)
#         child_conn_list.append(child_conn)
        
#         args_list[i].append(child_conn_list[i])
#         process_list.append(multiprocessing.Process(target=target_list[i], args=tuple(args_list[i])))
    
#     start_time = time.time()
#     for process in process_list:
#         process.start()

#     is_timeout = False
#     while True:
#         curr_time = time.time()
#         if curr_time - start_time > timeout:
#             is_timeout = True
#             break
        
#         is_done = True
#         for i in range(len(process_list)):
#             # print(f"process {i} is_alive = {process_list[i].is_alive()}")
#             if process_list[i].is_alive():
#                 time.sleep(0.1)
#                 is_done = False
#                 break
#         if is_done:
#             break
    
#     results = []
#     if is_timeout:
#         for process in process_list:
#             process.terminate()
#             process.join()
#         raise Exception("Watchdog timeout for {}!".format(task_name))
#     else:
#         for i in range(len(process_list)):
#             results.append(parent_conn_list[i].recv())
    
#     # end after any one finished
#     for process in process_list:
#         process.terminate()

#     return results