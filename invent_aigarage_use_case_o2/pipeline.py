from fastapi import FastAPI
from openai import AsyncAzureOpenAI
from utils.retriever import retriever_model_inference
from loguru import logger
import faiss

import numpy as np
import pandas as pd

import os
import dotenv
dotenv.load_dotenv()

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

def vector_similarity(search_vector, index, top_k: int): 

    _vector = np.array(search_vector)

    faiss.normalize_L2(_vector)
    index.nprobe = 10  # increase the number of nearby cells to search too with `nprobe`
    k = min(top_k, index.ntotal)

    distances, ann = index.search(_vector, k=k)

    return distances, ann



def retrieve_context(search_vector, df_condensed, index_path: str="a4_db_data.index", top_k: int=3, distance_threshold=0.3) -> pd.DataFrame:
    """
    function to retrieve relevant documents from an index based on a user query
    - documents are used as context for the LLM

    :param query: user question
    :param top_k: top k results found in the index based on the user query

    :return: found context information based on the user query
    """

    # use FAISS index to search for context

    index = faiss.read_index(index_path)
    distances, ann = vector_similarity(search_vector, index, top_k)


    results = pd.DataFrame({"distances": distances[0], "ann": ann[0]})

    merge = pd.merge(results, df_condensed, left_on="ann", right_index=True)
    print(f"these are the similar documents:")
    print(merge.sort_values(by="distances", ascending=False).head())
    print("-----------------------------------------------------")

    #merge_filtered = merge[merge["distances"] >= distance_threshold]
    
    return merge

def retrieve(query: str, top_k: int) -> list:

    path_to_df_full = "a4_db_data.csv"
    df_condensed = pd.read_csv(path_to_df_full)

    context = retrieve_context(query, df_condensed, top_k=top_k)

    return context

def get_embedding(text_data: str) -> list[float]:
    """Retrieve a single embedding synchrone"""
    try:
        response = retriever_model_inference([text_data])
        return response
    except Exception as e:
        logger.error(f"An error occurred requesting an embedding: {e}")


@app.post("/query")
async def query(query, top_k):
    # embedding
    embedding = get_embedding(query)
    # context definition / retrieval
    context = retrieve(embedding, int(top_k))
 
    logger.debug(f"Got context {context} for query: {query}")
 
    context_str = "\n".join(i for i in context.sort_values(by="distances", ascending=False).text_chunks.to_list())

    document_meta = context[["patent_id", "inventor", "distances"]].T.to_dict()

    # Add history

    prompt = generate_prompt(query, context_str)

    response = await execute_prompt_with_openai(prompt)

    print(response)

    logger.debug(f"OA response: {response}")

    if document_meta:
        context_answer = (
            "\n\nI found the following documents where you can find more details w.r.t. your "
            "question: "
        )
        response = response + context_answer

        for _, v in document_meta.items():
            context_document = f"\n- Filename: {v['patent_id']} (Inventor: {str(v['inventor'])}); Distance: {round(v['distances'], 2)}"
            response += context_document

    return {"response": response}

 
def generate_prompt(input_text: str, context, conversation_history: str="") -> str:
    prompt = [
            {
                "role": "system",
                "content": "You are an expert Q&A system that supports the employees of the consulting company Capgemini to search the internal knowledge database.\n"
            },
            {
                "role": "user",
                "content":
                    f"{conversation_history}\n"
                    "---------------------\n"
                    "We have provided context documents below:\n"
                    "---------------------\n"
                    f"{context}\n"
                    "---------------------\n"
                    f"Based on your prior knowledge as well as the reference information, please answer the question: '{input_text}'.\n"
                    "---------------------\n"
                    "Make sure to answer the query to the best of your ability. Your first priority should be to use information from the context documents, but otherwise, you are strongly encouraged to fall back on your prior knowledge.\n"
                    "---------------------\n"
                    "Make it clear to the user in case you relied on prior knowledge.\n"
            }
        ]

    return prompt
 

 
async def execute_prompt_with_openai(prompt: str) -> str:
    client = AsyncAzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
        api_version="2024-03-01-preview",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )

    
    deployment_name = "chat-model"
    response = await client.chat.completions.create(
        model=deployment_name,
        messages=prompt,
        temperature=0.0,
        max_tokens=200,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )

    return response.choices[0].message.content
