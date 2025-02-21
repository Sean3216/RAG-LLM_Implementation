from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def define_model(hugging_api):
    """
    When necessary, define the model hyperparameters in the HuggingFaceEndpoint
    model_kwargs = {
        "max_length": 64,
        "max_new_tokens": 512,
        "temperature": 0.7
    """
    model = HuggingFaceEndpoint(
        repo_id = 'meta-llama/Llama-3.2-3B',
        task = 'text-generation',
        huggingfacehub_api_token=hugging_api,
        top_k = 10,
        top_p = 0.85,
        temperature = 0.7,
        max_new_tokens = 512,
        seed = 42
    )
    return model

def ask_question(question, llm_mod, retriever, use_rag = True):
    if use_rag:
        prompt = """
        You are a customer service representative of a bank. Your answer should refer to the context provided. If you are not sure, please ask for the customer to contact a human representative.
        Context: {context}
        Question: {question}
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