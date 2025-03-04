import argparse
from func.prompt_func import *
from func.rag_func import *
import os

#parse the config dir file
parser = argparse.ArgumentParser(
    description='A script to demonstrate the RAG-LLM pipeline'
)
parser.add_argument(
    '--rag_data_path',
    type = str,
    default = None,
    help = 'The path to the RAG dataset'
)
parser.add_argument(
    '--hugging_face_key',
    type = str,
    default = None,
    help = 'Your personal hugging face API key'
)
parser.add_argument(
    '--splitter_type',
    type = str,
    default = 'Semantic',
    help = 'The type of splitter to use. Options are "Semantic" or "Recursive"'
)
parser.add_argument(
    '--show_similar_docs',
    action='store_true',
    help='Whether to show similar documents found in the RAG dataset'
)
parser.add_argument(
    '--use_rag',
    action='store_true',
    help='Whether to use the RAG pipeline or not'
)
parser.add_argument(
    '--num_k',
    type = int,
    default = 2,
    help = 'The number of similar documents to retrieve'
)

args = parser.parse_args()

if args.rag_data_path is None:
    print("Please provide the path to the RAG dataset")
    exit(1)
if args.hugging_face_key is None:
    print("Please provide your hugging face API key")
    exit(1) 

def main():
    ### preparing the data
    data = load_data_from_csv(args.rag_data_path)
    docs_chunks, embedding_model = split_documents(data, args.splitter_type)
    retriever, vector_store = create_retriever(docs_chunks, embedding_model, num_k=args.num_k)

    ### prompt
    #while the user wants to continue, continue to prompt
    llm = define_model(args.hugging_face_key)
    #clear the screen
    os.system('cls' if os.name == 'nt' else 'clear')
    while True:
        question = input("Enter your question: ")
        if question.lower() == 'exit':
            break

        try:
            answer = ask_question(question, llm, retriever, use_rag = args.use_rag)
            #Display text after Answer:
            answer = answer.split("Answer:")[1]
        except Exception as e:
            print(f"An error occurred: {e}")

        if args.show_similar_docs:
            print("Similar documents found in the RAG dataset:")
            sim_docs = vector_store.similarity_search(question)
            for doc in sim_docs:
                print(doc)
            
        print("="*100)
        print(answer)

if __name__ == '__main__':
    main()