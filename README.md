Virtual Sales Representative for Prebuilt PCs
1. Introduction
This project aims to develop a virtual sales representative tailored for a prebuilt PC-selling company. The representative is designed to provide personalized product recommendations based on customer requirements and the company's product portfolio. Utilizing advanced techniques like Retrieval Augmented Generation (RAG) and model fine-tuning, this virtual salesperson is capable of understanding customer needs, recommending products, and providing information on service policies.

2. System Overview
Figure 1: Simplified Top Level Structure

3. Background & Related Work
The project builds upon research in domain-specific fine-tuning of Language Learning Models (LLMs), drawing inspiration from works by Xiaochuang Han, Jacob Eisenstein, Suchin Guruangan, and their colleagues.

4. Data and Data Processing
4.1 Information Used by RAG
Data source: Product list from iBUYPOWER.
Data nature: Specifications and prices of pre-built PCs.
4.2 Dataset Used for Fine-tuning
Approach: Use of GPT-4 to generate domain-specific dialogue data.
Example: Training data sample.
Figure 3: Fine-tune Training Data Sample
[Image description of training data sample]

5. Architecture and Software
5.1 Information Retrieval
Structure: Combination of GPT-3.5-turbo and GPT-4 panda-agent for product retrieval.
RAG: Implementation using Sparse Retriever and Embedding Retriever.
Figure 4: Information Retrieval Structure
[Image description of information retrieval structure]

5.2 Fine-tuning Process
Model: GPT-3.5-turbo.
Methodology: Low-Rank Adaptation (LoRA) and Parameter-Efficient Fine-Tuning (PEFT).
Figure 5: Fine-Tuning Process
[Image description of the fine-tuning process]

6. Baseline Model and Comparison
6.1 RAG Evaluation
Approach: Dataset creation and hit rate analysis.
6.2 Fine-tuned Model Evaluation
Baseline: Non-fine-tuned GPT-3.5-turbo.
Criteria: Customer Understanding, Product Recommendation Relevance, Communication and Engagement, Upselling and Cross-Selling Skills, Handling Objections and Closing Sales.
Table 1: Rubric for Human Evaluation
[Table description of human evaluation rubric]

7. Quantitative Results
Information Retrieval: 91.67% overall hit rate.
Fine-tuned Model: Highest train accuracy of 0.9837.
Figure 6: Train Accuracy Curve for Fine-tuned GPT 3.5 Model
[Image description of training accuracy curve]

8. Qualitative Results
Figure 7: Exception Handling Example
[Image description of exception handling sample response]

Figure 8: Upselling Example
[Image description of upselling sample response]

9. Discussion and Learning
Key insights and lessons learned from the project are discussed, focusing on the effectiveness of the RAG component, the benefits of fine-tuning, and considerations for future projects.
