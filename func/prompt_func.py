from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.llms import HuggingFacePipeline
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def define_model(hugging_api):
    """
    Using local model instead of HuggingFaceEndpoint due to constant server error
    error:
    An error occurred: 500 Server Error: Internal Server Error for url: https://api-inference.huggingface.co/models/meta-llama/Llama-3.2-3B (Request ID: U_he4nj2C9do4Ff4sSWlP)
    """
    # llm = HuggingFaceEndpoint(
    #     repo_id = 'meta-llama/Llama-3.2-1B',
    #     task = 'text-generation',
    #     huggingfacehub_api_token=hugging_api,
    #     top_k = 10,
    #     top_p = 0.85,
    #     temperature = 0.7,
    #     max_new_tokens = 512,
    #     seed = 42
    # )
    
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
    
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
    
    llm_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        top_k = 30,
        top_p = 0.9,
        temperature = 0.1,
        max_new_tokens = 512,
        device=0,
        repetition_penalty=1
    )
    llm = HuggingFacePipeline(pipeline = llm_pipeline)
    return llm

def ask_question(question, llm_mod, retriever, use_rag = True):
    if use_rag:
        prompt = """
        You are a customer service representative of a bank. 
        When answering, please refer heavily to the context provided. 
        If you are not sure, please ask for the customer to contact a human representative.
        Context: {context}
        Question: {question}
        Answer:
        """

        prompt_template = ChatPromptTemplate.from_template(prompt)
        chain = (
            {'context': retriever, 'question': RunnablePassthrough()}
            | prompt_template
            | llm_mod
            | StrOutputParser()
        )
    else:
        prompt = """
        Question: {question}
        Answer:
        """
        prompt_template = PromptTemplate.from_template(prompt)
        chain = (
            prompt_template
            | llm_mod
            | StrOutputParser()
        )
    answer = chain.invoke(question)
    return answer