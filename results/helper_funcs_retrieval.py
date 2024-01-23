import openai
import status
from  models.vectorstore import my_vectors_store
import json
import time

def create_vectorstore(databases):
    return my_vectors_store(databases)

def get_database_in_docs(vectorstore):
    docs = vectorstore.documents
    # Note: this contain duplicates
    return [doc.metadata['source'].split('\\')[-2] for doc in docs]

def update_database(vectorstore, databases):
    exist_databases = get_database_in_docs(vectorstore)
    # add docs for new database
    for database in databases:
        if database not in exist_databases:
            vectorstore.get_database_documents(database)
    # remove docs for unloaded database
    vectorstore.remove_unneeded_database_documents(databases)
    # update vectorstore based on the updated docs
    vectorstore.update_vectorstore()
    
def remove_database(vectorstore, database):
    vectorstore.remove_datase_documents(database)

def update_cost(response, model="gpt-3.5-turbo"):
    prompt_tokens = response.usage.prompt_tokens
    completion_tokens = response.usage.completion_tokens
    status.total_tokens_used = status.total_tokens_used + prompt_tokens + completion_tokens
    if model == "gpt-3.5-turbo":
        status.total_cost = status.total_cost + prompt_tokens * status.gpt3_5_in + completion_tokens * status.gpt3_5_gen
    elif model == "gpt-4":
        status.total_cost = status.total_cost + prompt_tokens * status.gpt4_in + completion_tokens * status.gpt4_gen

def send_request_openai(model, sys_msg, human_msg, temperature, max_tokens):
    print("Sending request to OpenAI for FYI")
    for retries in range(3):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": sys_msg
                    },
                    {
                        "role": "user",
                        "content": human_msg
                    }
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            return response
        except openai.error.OpenAIError as e:
            print("Error {} encounted, will try again".format(e))
            time.sleep(2)
            continue

def get_reformated_customer_question(history_msg, customer_msg):
    prompt = """
Chat History:
{}
Follow Up Input: ###Customer: {}
Standalone question as the Customer: """.format(history_msg, customer_msg)
    system_msg = "Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question as you are the Customer. You should always include the number of budget if the customer is asking for product. You should mention if the customer does not have specific requirements on CPU, GPU, or storage. "
    
    # send requirements to gpt-3.5
    response = send_request_openai("gpt-3.5-turbo", system_msg, prompt, 0, 256)
    
    update_cost(response)
    return response.choices[0].message.content

# this function checks if the budget is been mentioned, if it is, will return the value 
def check_budget(customer_msg):
    system_msg = "Has the customer mentioned the budget NUMBER? If the customer has, reply just the budget, otherwise, reply 'No' ONLY. "
    
    # send requirements to gpt-3.5
    response = send_request_openai("gpt-3.5-turbo", system_msg, customer_msg, 0, 256)
    
    update_cost(response)
    rst_str = response.choices[0].message.content.replace('$', '')
    if rst_str == 'No':
        return -1
    else:
        budget = float(rst_str)
        print("Budget detected: {}".format(budget))
        return budget # will return True if format is not correct

# this function checks if the new customer message contains product requirements or not
def check_product_requirement(customer_msg):
    system_msg = "Here is a user who wants to purchase a pre-build PC. You need to identify if the user asking for a new request directly about the product, or about services or external products related to the product like upgradability, accessories, warranties, return policies, or shipping. Reply \"Yes\" if it is about the product itself, \"No\" otherwise. "
    
    # send requirements to gpt-3.5
    response = send_request_openai("gpt-3.5-turbo", system_msg, customer_msg, 0, 256)
    
    update_cost(response)
    rst_str = response.choices[0].message.content
    if rst_str == 'No':
        return False
    else:
        print("Check '{}' for product requirement: {}".format(customer_msg, rst_str))
        return True # will return True if format is not correct
    
# this functionn returns customers' requirements in a list format of [budget, CPU brand, GPU brand, minimum storage]
def get_customer_reqirements(customer_msg):
    system_msg = "Given the following conversation and a follow-up question, what is the customer's budget, preferred CPU brand, preferred GPU brand, and minimum storage size? Reply ONLY in this format: [\"budget\", \"CPU brand\", \"GPU brand\", \"minimum storage\"], and use \"\" for fields that the customer did not specify. "

    # send requirements to gpt-3.5
    response = send_request_openai("gpt-3.5-turbo", system_msg, customer_msg, 0, 256)
    
    rst_str = response.choices[0].message.content
    try:
        reqirement_list = json.loads(rst_str)
        reqirement_list[0] = reqirement_list[0].replace("$", "")
        return reqirement_list
    except:
        print("GPT 3.5 reqirement summary failed, output = {}".format(rst_str))
        return []

# def summarize_product_requirement(history, customer_msg, customer_msg_req_lookup):
#     # parse customer's messages
#     customer_reqirements = ""
#     for conversation in history:
#         if customer_msg_req_lookup[conversation[0]]:
#             customer_reqirements = customer_reqirements + conversation[0] + '\n'
#     customer_reqirements = customer_reqirements + customer_msg + '\n'
    
#     # send requirements to gpt-3.5
#     response = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         messages=[
#             {
#                 "role": "system",
#                 "content": "Here is a user who wants to purchase a pre-build PC and some user's messages. Please summarize all the user's requirements into a short sentence, using 'I' for the user, using 'GeForce' to replace Nvidia if mentioned. "
#             },
#             {
#                 "role": "user",
#                 "content": customer_reqirements
#             }
#         ],
#         temperature=0,
#         max_tokens=256,
#         top_p=1,
#         frequency_penalty=0,
#         presence_penalty=0
#     )
    
#     update_cost(response)
#     return response.choices[0].message.content

# doing llm_chain_filter manully since langchain llm_chain_filter does not support chat models
def check_relavent(customer_msg, doc_info):
    system_msg = "Given the following question and context, return YES if the context contains the answer to the question, and NO if it isn't."
    prompt = """
> Question: {}
> Context:
>>>
{}
>>>
> Relevant (YES / NO):""".format(customer_msg, doc_info)

    # send requirements to gpt-3.5
    response = send_request_openai("gpt-3.5-turbo-0613", system_msg, prompt, 0, 256)

    rst_str = response.choices[0].message.content
    if rst_str == "NO":
        return False
    elif rst_str == "YES":
        return True
    else:
        print("Unexpected output from check_relavent: {}, so return True just in case".format(rst_str))
        return True

# product retrieve
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain.llms import OpenAI
import pandas as pd
import re

df = pd.read_csv(status.product_path, encoding='ISO-8859-1')
pd.set_option('display.max_colwidth', None)
agent = create_pandas_dataframe_agent(
    ChatOpenAI(temperature=0, model="gpt-4"),
    # ChatOpenAI(temperature=0, model="gpt-4-1106-preview"),
    df,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
)
q1_prompt = "Find the minimum index of products that satisfies the customer's requirements and price <= budget. Answer ONLY the index, or \"nothing\" if no matches. Customer: "
q2_prompt = "Find the maximum index of products that satisfies the customer's requirements and price > (budget*1.1). Answer ONLY the index, or \"nothing\" if no matches. Customer: "
# q1_prompt = "Using tool 'python_repl_ast', find the 'index.min()' of products that satisfies the customer's requirements and price <= budget. Answer ONLY the index ONLY, or \"nothing\" if no matches. Customer: "
# q2_prompt = "Using tool 'python_repl_ast', find the 'index.max()' of products that satisfies the customer's requirements and price > (budget*1.1). Answer ONLY the index ONLY, or \"nothing\" if no matches. Customer: "

def df_row_to_str(df, index, get_comment=True):
    row = df.iloc[index]
    if get_comment:
        out_str = '\n'.join(f"{index}: {value}" for index, value in row.items())
    else:
        out_str = '\n'.join(f"{index}: {value}" for index, value in row.items() if index != 'Performance Comment')
    return out_str

def get_is_num(text):
    match = re.search(r'is (\d+)\.', text)
    if match:
        number = match.group(1)
        return number
    else:
        return text

def product_retrieve(customer_msg, get_comment):
    print("Doing product query, customer_msg = {}".format(customer_msg))
    quary1 = q1_prompt + customer_msg # under budget product retrieve
    quary2 = q2_prompt + customer_msg # over  budget product retrieve
    
    res1 = agent.run(quary1)
    res2 = agent.run(quary2)
    
    res1 = get_is_num(res1)
    res2 = get_is_num(res2)
    
    prod1 = ""
    prod2 = ""
    
    res1_check_failed = (res1 != "nothing") and not bool(re.match(r'^[0-9]+$', res1))
    res2_check_failed = (res2 != "nothing") and not bool(re.match(r'^[0-9]+$', res2))
    if res1_check_failed and res2_check_failed:
        # if any of the agent output is not a number nor 'nothing', something went wrong
        return "error"
    
    # continue if both responce are either all numbers or 'nothing'
    if res1 != "nothing" and not res1_check_failed:
        prod_index1 = int(res1)
        prod1 = df_row_to_str(df, prod_index1, get_comment)
    if res2 != "nothing" and not res2_check_failed:
        prod_index2 = int(res2)
        prod2 = df_row_to_str(df, prod_index2, get_comment)
    
    prod_str = prod1 + "\n" + prod2
    
    return prod_str

def find_suitable_builds(df, budget, cpu, gpu, storage):
    # Convert budget to integer if it's a string
    budget = int(budget) if isinstance(budget, str) else budget

    # Filter the DataFrame based on the requirements
    filtered_df = df.copy()
    
    # Apply CPU filter if specified
    if cpu:
        filtered_df = filtered_df[filtered_df['Processor (CPU)'].str.contains(cpu, case=False)]
    
    # Apply GPU filter if specified
    if gpu:
        gpu = gpu.replace('Nvidia', 'GeForce')
        filtered_df = filtered_df[filtered_df['Video Card (GPU)'].str.contains(gpu, case=False)]
    
    # Apply Storage filter if specified
    if storage:
        # Assuming only '500GB' uses GB as unit
        if 'GB' in storage:
            pass
        else:
            filtered_df['Storage Size'] = filtered_df['Storage'].str.extract(r'(\d+)TB').astype(float)
            filtered_df = filtered_df[filtered_df['Storage Size'] >= 2]

    # Find the most expensive under budget build
    under_budget_builds = filtered_df[filtered_df['Price with 1 Year Standard Warranty (USD)'] <= budget]
    most_expensive_under_budget = under_budget_builds['Price with 1 Year Standard Warranty (USD)'].idxmax() \
                                  if not under_budget_builds.empty else -1

    # Find the cheapest over budget build
    over_budget_builds = filtered_df[filtered_df['Price with 1 Year Standard Warranty (USD)'] > budget*1.1]
    cheapest_over_budget = over_budget_builds['Price with 1 Year Standard Warranty (USD)'].idxmin() \
                           if not over_budget_builds.empty else -1

    return most_expensive_under_budget, cheapest_over_budget

def product_retrieve_gpt3_5(customer_msg, get_comment):
    reqirement_list = get_customer_reqirements(customer_msg)
    
    if len(reqirement_list) == 0:
        # if gpt3.5 with custom function does not work, use gpt4 agent
        print("GPT 3.5 reqirement summary failed, try GPT 4 panda agent...")
        product_info = product_retrieve(customer_msg, get_comment)
        return product_info
    elif len(reqirement_list) == 4:
        # if gpt3.5 works, find out the product
        under_index, over_index = find_suitable_builds(df, reqirement_list[0], reqirement_list[1], reqirement_list[2], reqirement_list[3])
        
        prod1 = ""
        prod2 = ""
        if under_index != -1:
            prod1 = df_row_to_str(df, under_index, get_comment)
        if over_index != -1:
            prod2 = df_row_to_str(df, over_index, get_comment)
        
        prod_str = prod1 + "\n" + prod2
        
        return prod_str
    else:
        raise Exception("reqirement_list lenght not execpted! reqirement_list = {}".format(reqirement_list))
        
        
        