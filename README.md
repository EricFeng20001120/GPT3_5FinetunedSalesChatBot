# Virtual Sales Representative for Prebuilt PCs
## 1. Introduction
This project aims to develop a virtual sales representative tailored for a prebuilt PC-selling company. The representative is designed to provide personalized product recommendations based on customer requirements and the company's product portfolio. Utilizing advanced techniques like Retrieval Augmented Generation (RAG) and model fine-tuning, this virtual salesperson is capable of understanding customer needs, recommending products, and providing information on service policies.

## 2. System Overview
![1705980087365](https://github.com/EricFeng20001120/GPT3_5FinetunedSalesChatBot/assets/55144601/e2f381e0-b50e-4b9d-bd13-5274a1ac9401)
Figure 1: Simplified Top Level Structure

## 3. Background & Related Work
The project builds upon research in domain-specific fine-tuning of Language Learning Models (LLMs), drawing inspiration from works by Xiaochuang Han, Jacob Eisenstein, Suchin Guruangan, and their colleagues.

## 4. Data and Data Processing
4.1 Information Used by RAG
Data source: Product list from iBUYPOWER.
Data nature: Specifications and prices of pre-built PCs.
4.2 Dataset Used for Fine-tuning
Approach: Use of GPT-4 to generate domain-specific dialogue data.
Example: Training data sample.
![1705980131907](https://github.com/EricFeng20001120/GPT3_5FinetunedSalesChatBot/assets/55144601/7153e075-1286-49f5-9231-f437a03721ca)

Figure 2: Fine-tune Training Data Sample

## 5. Architecture and Software
5.1 Information Retrieval
Structure: Combination of GPT-3.5-turbo and GPT-4 panda-agent for product retrieval.
RAG: Implementation using Sparse Retriever and Embedding Retriever.
![1705980251768](https://github.com/EricFeng20001120/GPT3_5FinetunedSalesChatBot/assets/55144601/2259f0d5-7b16-4b73-b05f-8f26b46d535d)

Figure 3: Information Retrieval Structure

5.2 Fine-tuning Process
Model: GPT-3.5-turbo.
Methodology: Low-Rank Adaptation (LoRA) and Parameter-Efficient Fine-Tuning (PEFT).

## 6. Baseline Model and Comparison
6.1 RAG Evaluation
Approach: Dataset creation and hit rate analysis.
6.2 Fine-tuned Model Evaluation
Baseline: Non-fine-tuned GPT-3.5-turbo.
Criteria: Customer Understanding, Product Recommendation Relevance, Communication and Engagement, Upselling and Cross-Selling Skills, Handling Objections and Closing Sales.

## 7. Quantitative Results
Information Retrieval: 91.67% overall hit rate.
Fine-tuned Model: Highest train accuracy of 0.9837.
![1705980448235](https://github.com/EricFeng20001120/GPT3_5FinetunedSalesChatBot/assets/55144601/7d2530c7-d9bc-4f32-a12a-831a2e99a369)

Figure 4: Train Accuracy Curve for Fine-tuned GPT 3.5 Model

## 8. Qualitative Results
![1705980506153](https://github.com/EricFeng20001120/GPT3_5FinetunedSalesChatBot/assets/55144601/5998ba32-5ef6-42f4-8437-85ad253a6200)

Figure 5: Exception Handling Example
![1705980526118](https://github.com/EricFeng20001120/GPT3_5FinetunedSalesChatBot/assets/55144601/b280a732-4625-4758-b524-be8e8978dcd1)

Figure 6: Upselling Example

## 9. Discussion and Learning
The results from the information retrieval part were reasonably good, which means we can be confident that the LLM will have sufficient information for most of the customer’s questions. The model worked well with a product list that is different from what it was trained on showing that the model is not overfitting, and making changes to products and services information does not require to re-fine-tune the model, which meets expectations. 

It was realized that naive implementations like only doing embedding similarity search would not lead to a good performance. Some more advanced approaches are retrieving some generated relative questions, representing and splitting the document in different formats (like 'Smaller Chunks' or 'Summary' types of Multi-Vector Retriever), and combining different retrievers. It was also learned that too much contextual information can overwhelm the model and decrease the output quality, so implementing filters is a good practice to make sure all the retrieved information is relevant. 

The result from the human evaluation showcases that the fine-tuned model performs well in product relevance, upselling capabilities, and exception handling. One reason is that the fine-tuned model is trained with cases including situations of under budget, over budget, and with conversations where the agent tries to upsell. Therefore, invoking those rules in the prompt would be much more effective than the baseline model. Additionally, the fine-tuned model performs well by using prompt engineering to instruct special cases, while also training the model with special case data.

The performance improvement in the fine-tuned model shows that adding some fake retrieved information into the ‘instruction’ can help the model understand how to generate a good response when fine-tuning a model that will be used in a context-intense case. The fine-tuned model has improved on the hallucination problem when the customer’s budget cannot afford any product, while the baseline model would make up a non-existent product. It was learned that a specific hallucination can be mitigated by fine-tuning a dataset that contains the proper response in that specific case. 

When starting another similar project, the team may choose to use GPT-4-turbo directly and make it a type-2 project. The team has tested using GPT-4-turbo as the main chatbot model for interest, and it works better than the fine-tuned model without fine-tuning. If not considering the price of the model and the knowledge learned from this project, directly using GPT-4-turbo may be an easier and better approach. 
