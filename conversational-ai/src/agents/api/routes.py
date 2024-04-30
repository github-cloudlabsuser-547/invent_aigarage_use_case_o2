from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

import pandas as pd
import numpy as np
import json
import faiss
from openai import AzureOpenAI
import os 
import dotenv
from numpy import dot
from numpy.linalg import norm
dotenv.load_dotenv()

import agents.api.schemas
import agents.crud
import agents.models
from agents.database import SessionLocal, engine
from agents.processing import (craft_agent_chat_context,
                               craft_agent_chat_first_message,
                               craft_agent_chat_instructions)
from agentsfwrk import integrations, logger

log = logger.get_logger(__name__)

agents.models.Base.metadata.create_all(bind = engine)

# Router basic information
router = APIRouter(
    prefix = "/agents",
    tags = ["Chat"],
    responses = {404: {"description": "Not found"}}
)

# Dependency: Used to get the database in our endpoints.
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Root endpoint for the router.
@router.get("/")
async def agents_root():
    return {"message": "Hello there conversational ai!"}

@router.get("/get-agents", response_model = List[agents.api.schemas.Agent])
async def get_agents(db: Session = Depends(get_db)):
    """
    Get all agents endpoint.
    """
    log.info("Getting all agents")
    db_agents = agents.crud.get_agents(db)
    log.info(f"Agents: {db_agents}")

    return db_agents

@router.post("/create-agent", response_model = agents.api.schemas.Agent)
async def create_agent(agent: agents.api.schemas.AgentCreate, db: Session = Depends(get_db)):
    """
    Create an agent endpoint.
    """
    log.info(f"Creating agent: {agent.json()}")
    db_agent = agents.crud.create_agent(db, agent)
    log.info(f"Agent created with id: {db_agent.id}")

    return db_agent

@router.get("/get-conversations", response_model = List[agents.api.schemas.Conversation])
async def get_conversations(agent_id: str, db: Session = Depends(get_db)):
    """
    Get all conversations for an agent endpoint.
    """
    log.info(f"Getting all conversations for agent id: {agent_id}")
    db_conversations = agents.crud.get_conversations(db, agent_id)
    log.info(f"Conversations: {db_conversations}")

    return db_conversations

@router.post("/create-conversation", response_model = agents.api.schemas.Conversation)
async def create_conversation(conversation: agents.api.schemas.ConversationCreate, db: Session = Depends(get_db)):
    """
    Create a conversation linked to an agent
    """
    log.info(f"Creating conversation assigned to agent id: {conversation.agent_id}")
    db_conversation = agents.crud.create_conversation(db, conversation)
    log.info(f"Conversation created with id: {db_conversation.id}")

    return db_conversation

@router.get("/get-messages", response_model = List[agents.api.schemas.Message])
async def get_messages(conversation_id: str, db: Session = Depends(get_db)):
    """
    Get all messages for a conversation endpoint.
    """
    log.info(f"Getting all messages for conversation id: {conversation_id}")
    db_messages = agents.crud.get_messages(db, conversation_id)
    log.info(f"Messages: {db_messages}")

    return db_messages

@router.post("/chat-agent")
async def chat_completion(message: agents.api.schemas.UserMessage, db: Session = Depends(get_db)):
    """
    Get a response from the GPT model given a message from the client using the chat
    completion endpoint.

    The response is a json object with the following structure:
    ```
    {
    "api_response": {
        "conversation_id": "string",
        "response": "string"
    },
    "relevant_history": [
        {
        "role": "string",
        "content": "string"
        },
        {
        "role": "string",
        "content": "string"
        },
        ...
    ],
    "metadata": [
        {
           'rank': "int",
            'title': 'string',
            'priority_date': "YYYY-MM-DD",
            'filing_date': "YYYY-MM-DD",
            'grant_date': "YYYY-MM-DD",
            'publication_date': "YYYY-MM-DD",
            'inventor': "string",
            'assignee': "string",
            'publication_number': "string",
            'language': "string",
            'thumbnail': "link",
            'pdf': "link",
            'page': "float",
            'entities' : "list[dict]"
        },
        {
           'rank': "int",
            'title': 'string',
            'priority_date': "YYYY-MM-DD",
            'filing_date': "YYYY-MM-DD",
            'grant_date': "YYYY-MM-DD",
            'publication_date': "YYYY-MM-DD",
            'inventor': "string",
            'assignee': "string",
            'publication_number': "string",
            'language': "string",
            'thumbnail': "link",
            'pdf': "link",
            'page': "float",
            'entities' : "list[dict]"
        },
        ...
        ]
    }
    ```
    """
    log.info(f"User conversation id: {message.conversation_id}")
    log.info(f"User message: {message.message}")

    conversation = agents.crud.get_conversation(db, message.conversation_id)

    if not conversation:
        # If there are no conversations, we can choose to create one on the fly OR raise an exception.
        # Which ever you choose, make sure to uncomment when necessary.

        # Option 1:
        # conversation = agents.crud.create_conversation(db, message.conversation_id)

        # Option 2:
        return HTTPException(
            status_code = 404,
            detail = "Conversation not found. Please create conversation first."
        )

    log.info(f"Conversation id: {conversation.id}")

    # NOTE: We are crafting the context first and passing the chat messages in a list
    # appending the first message (the approach from the agent) to it.
    context = craft_agent_chat_context(conversation.agent.context)
    chat_messages = [craft_agent_chat_first_message(conversation.agent.first_message)]

    # NOTE: Append to the conversation all messages until the last interaction from the agent
    # If there are no messages, then this has no effect.
    # Otherwise, we append each in order by timestamp (which makes logical sense).
    hist_messages = conversation.messages
    hist_messages.sort(key = lambda x: x.timestamp, reverse = False)
    if len(hist_messages) > 0:
        for mes in hist_messages:
            chat_messages.append(
                {
                    "role": "user",
                    "content": mes.user_message
                }
            )
            chat_messages.append(
                {
                    "role": "assistant",
                    "content": mes.agent_message
                }
            )
    # NOTE: We could control the conversation by simply adding
    # rules to the length of the history.
    # if len(hist_messages) > 10:
    #     # Finish the conversation gracefully.
    #     log.info("Conversation history is too long, finishing conversation.")
    #     api_response = agents.api.schemas.ChatAgentResponse(
    #         conversation_id = message.conversation_id,
    #         response        = "This conversation is over, good bye."
    #     )
    #     return api_response

    # Send the message to the AI agent and get the response
    service = integrations.OpenAIIntegrationService(
        context = context,
        instruction = craft_agent_chat_instructions(
            conversation.agent.instructions,
            conversation.agent.response_shape
        )
    )
    service.add_chat_history(messages = chat_messages)
    
    embedding = get_embedding(message.message)

    broad_context, df = retrieve(embedding, top_k = 30)

    # Reranker
    import requests

    url = f"https://api.jina.ai/v1/rerank"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer jina_a928d4390faa4b7ba9dc8c748ff57e39VIWwKH6c0_qgw8zyWnwg_tLMvV74"
    }

    data = {
        "model": "jina-reranker-v1-base-en",
        "query": message.message,
        "documents": broad_context.text_chunks.to_list(),
        "top_n": 3
    }

    response = requests.post(url, headers=headers, json=data)

    context = broad_context.iloc[[int(i['index']) for i in response.json()['results']], :].reset_index()

    # Theoretically attributed new relevance we could/ should use
    context_str = "\n".join(i['document']['text'] for i in response.json()['results'])

    # Get related NER docs
    context_ner = total_context_ner(context['entities'])["technologies"]
    context_ner_embed = embed(context_ner)


    df_excluded = broad_context.loc[(~broad_context["publication_number"].isin(context["publication_number"].values) & (broad_context['entities'].notnull()))]

    # sort by publication data and rank
    df_excluded.loc[:, "publication_date"] = pd.to_datetime(df_excluded['publication_date'], format="%Y-%m-%d")
    df_excluded = df_excluded.sort_values(by=['rank', 'publication_date'], ascending=[True, True])


    df_excluded.loc[:, "relevant_ner"] = False
    i = 0
    for _, row in df_excluded.iterrows():
        comparison_list = json.loads(row["entities"])["technologies"]
        comparison_embed = embed(comparison_list)

        # Check for overlap
        threshold = 0.95  # Adjust according to your needs
        overlap = check_overlap(context_ner_embed.data, comparison_embed.data, threshold)
        row["relevant_ner"] = overlap
        if overlap == True: 
            i += 1
        if i == 3:
            break

    ner_recommendations = df_excluded.loc[df_excluded["relevant_ner"] == True].to_json(orient='records')
    

    meta_json_data = context[[
        'rank', 'title', 'priority_date', 'filing_date', 'grant_date', 'publication_date',
        'inventor', 'assignee', 'publication_number', 'language', 'thumbnail', 'pdf', 'page', 'entities'
        ]].to_json(orient='records')

    context_prompt =  generate_prompt(message.message, context_str)
    

    response = service.answer_to_prompt(
        # We can test different OpenAI models.
        model               = "chat-model",
        prompt              = context_prompt,
        # We can test different parameters too.
        temperature         = 0.5,
        max_tokens          = 1000,
        frequency_penalty   = 0.5,
        presence_penalty    = 0
    )

    service.messages.append(
            {
                'role': 'user',
                'content': message.message
            }
        )
    
    service.messages.append(
            {
                'role': 'assistant',
                'content': response
            }
        )
    

    log.info(f"Agent response: {response}")

    # Prepare response to the user
    api_response = agents.api.schemas.ChatAgentResponse(
        conversation_id = message.conversation_id,
        response        = response.get('answer')
    )

    # Save interaction to database
    db_message = agents.crud.create_conversation_message(
        db = db,
        conversation_id = conversation.id,
        message = agents.api.schemas.MessageCreate(
            user_message = message.message,
            agent_message = response.get('answer'),
        ),
    )
    log.info(f"Conversation message id {db_message.id} saved to database")

    return {"api_response": api_response, "relevant_history": service.messages[:4], "metadata": meta_json_data, "ner_recommendations": ner_recommendations}

#### NICOLAS ADDED: MOVE APPROPRIATELY

# Define function to compute cosine similarity between two vectors
def compute_similarity(vector1, vector2):
    # Reshape vectors if necessary
    vector1 = np.array(vector1.embedding)
    vector2 = np.array(vector2.embedding)
    # Compute cosine similarity
    cos_sim = dot(vector1, vector2)/(norm(vector1)*norm(vector2))
    return cos_sim


# Define function to check overlap between two lists
def check_overlap(list1, list2, threshold):
    for item1 in list1:
        for item2 in list2:
            similarity_score = compute_similarity(item1, item2)
            if similarity_score > threshold:
                return True
    return False

def embed(list):
    client = AzureOpenAI(
        api_key = os.getenv("AZURE_OPENAI_KEY"),
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version="2024-03-01-preview"
        )


    embeddings = client.embeddings.create(
        input=list,
        model="text-embedding-ada-002"
    )
    return embeddings


def total_context_ner(data: np.array): 
    # Initialize dictionaries to hold aggregated values
    aggregated_data = {
        "technologies": [],
        "places": [],
        "people": [],
        "organizations": []
    }

    # Iterate over each JSON string in the array
    for item in data:
        if isinstance(item, str) and item != 'nan':  # Check if the item is a valid JSON string
            json_data = json.loads(item)
            for key in aggregated_data.keys():
                # Add unique values from the current JSON to the aggregated data
                aggregated_data[key] += [value for value in json_data.get(key, []) if value not in aggregated_data[key]]

    # Convert lists to sets to remove duplicates, then back to lists
    for key in aggregated_data.keys():
        aggregated_data[key] = list(set(aggregated_data[key]))

    return aggregated_data


def generate_prompt(input_text: str, context) -> list[str]:
    prompt = [
            {
                "role": "user",
                "content":
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
    #merge_filtered = merge[merge["distances"] >= distance_threshold]
    
    return merge


def retriever_model_inference(text_input: list[str]):

    client = AzureOpenAI(
        api_key = os.getenv("AZURE_OPENAI_KEY"),
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version="2024-03-01-preview"
        )

    embeddings = client.embeddings.create(
        input=text_input,
        model="text-embedding-ada-002"
    )

    vectors = [np.array(i.embedding, dtype='f') for i in embeddings.data]


    return vectors


def vector_similarity(search_vector, index, top_k: int): 

    _vector = np.array(search_vector)

    faiss.normalize_L2(_vector)
    index.nprobe = 10  # increase the number of nearby cells to search too with `nprobe`
    k = min(top_k, index.ntotal)

    distances, ann = index.search(_vector, k=k)

    return distances, ann


def retrieve(query: str, top_k: int) -> list:

    path_to_df_full = "a4_db_data.csv"
    df_condensed = pd.read_csv(path_to_df_full)

    context = retrieve_context(query, df_condensed, top_k=top_k)

    return context, df_condensed


def get_embedding(text_data: str) -> list[float]:
    """Retrieve a single embedding synchrone"""
    try:
        response = retriever_model_inference([text_data])
        return response
    except Exception as e:
        log.error(f"An error occurred requesting an embedding: {e}")

