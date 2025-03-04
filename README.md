# Demonstration of RAG-LLM using Langchain and DPR (Dense Passage Retrieval)

- The Dense Passage Retrieval will utilize ChromaDB
- Llama 3.2 1B is used as the LLM Model
- For the example data, I only use the column "Response" as the source of documents to retrieve

### Description
This is a demonstration of a RAG-LLM pipeline using Langchain and ChromaDB. The model is not yet fine-tuned but it can demonstrate how we utilize a certain data for RAG pipeline. You will need a HuggingFace access token and obtain permission to use Llama model.

Instructions:
- Install the required libraries listed in the requirements.txt
- Run the module with main.py from the terminal (Example command is shown below)

### Example command
```shell
python main.py --rag_data_path data/Dataset_Banking_chatbot.csv --hugging_face_key <your hugging face key> --splitter_type Semantic --show_similar_docs --use_rag 
```

