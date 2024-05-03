from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

import pandas as pd
import numpy as np
import json
import faiss
from openai import AzureOpenAI
import os 
import requests
import dotenv
from numpy import dot
from numpy.linalg import norm
dotenv.load_dotenv()
from sqlalchemy import create_engine, MetaData
import agents.api.schemas
import agents.crud
import agents.models
from agents.database import SessionLocal, engine
from agents.processing import (craft_agent_chat_context,
                               craft_agent_chat_first_message,
                               craft_agent_chat_instructions)
from agentsfwrk import integrations, logger

log = logger.get_logger(__name__)


# Router basic information
router = APIRouter(
    prefix = "/agents",
    tags = ["Chat"],
    responses = {404: {"description": "Not found"}}
)

# Dependency: Used to get the database in our endpoints.
def get_db():
    agents.models.Base.metadata.create_all(bind = engine)
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Root endpoint for the router.
@router.get("/")
async def agents_root():
    return {"message": "Hello there conversational ai!"}

# Root endpoint for the router.
@router.get("/reset-db")
def reset_db(db: Session = Depends(get_db)):
    
    # Close existing sessions
    SessionLocal.close_all()

    # Reflect existing tables and drop them
    metadata = MetaData()
    metadata.reflect(bind=engine)
    metadata.drop_all(bind=engine)

    return {"message": "DB reset"}

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

@router.post("/save-chat")
async def save_chat(message: agents.api.schemas.SaveChat, db: Session = Depends(get_db)):
    log.info(f"User conversation id: {message.conversation_id}")

    conversation = agents.crud.get_conversation(db, message.conversation_id)

    if not conversation:
        return HTTPException(
            status_code = 404,
            detail = "Conversation not found. Please create conversation first."
        )

    chat_messages = []

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

    # Initialize an empty string to store the chat
    chat_string = ""
    for msg in chat_messages:
        chat_string += f"{msg['role']}: {msg['content']}\n"

    prompt = [
        {
            "role": "system",
            "content": "Summarize the user chat input, focusing on capturing user knowledge."
        },
        {
            "role": "user",
            "content": chat_string
        }
    ]

    client = AzureOpenAI(
        api_key = os.getenv("AZURE_OPENAI_KEY"),
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version="2024-03-01-preview"
        )

    summary = client.chat.completions.create(
                    model       = "chat-model",
                    messages    = prompt,
                ).choices[0].message.content

    # tagging
    timestamp = hist_messages[-1].timestamp

    # Prepare response to the user
    api_response = agents.api.schemas.ChatAgentResponse(
        conversation_id = message.conversation_id,
        timestamp       = timestamp,
        response        = summary
    )

    # Save interaction to csv
    # TODO: df groupby conversation id get latest record based on timestamp
    csv_file = "saved_conversations.csv"
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
    else:
        df = pd.DataFrame(columns=['conversation_id', 'timestamp', 'summary'])
    
    new_row = {'conversation_id': message.conversation_id, 'timestamp': timestamp, 'summary': summary}
    df.loc[len(df)] = new_row
    df.to_csv(csv_file, index=False)
    
    return {"api_response": api_response}

    # save to db
    # delete conversation

@router.post("/chat-agent")
async def chat_completion(message: agents.api.schemas.UserMessage, db: Session = Depends(get_db)):
    """
    Get a response from the GPT model given a message from the client using the chat
    completion endpoint.
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

    message.message = message.message.lower().replace("Quaero, ", "")

    chat_messages = []

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


    ### USER INTENT CLASSIFICATION
    intent = get_user_intent(message.message, chat_messages[-1:]) # What if not adminstration or technical
    log.info(f"User input classified as: {intent}")

    if ("1" == intent) or ("organisation" in message.message) or ("admin" in message.message) or ("company" in message.message):
        # Adminstration
        tag = "procedures"
        path_to_df = "data/a4_db_onboarding.csv"
        path_to_index = "data/a4_db_onboarding.index"
        meta_columns = ['rank', 'title', 'publication_date', 'author', 'department', 'publication_number', 'language', 
            'thumbnail', 'pdf', 'page', 'entities']
        system_prompt = "You are an organizational agent, helping users with their questions on the organization and adminstrative procedures."

    elif ("2" == intent) or ("manual" in message.message) or ("patent" in message.message) or ("tech" in message.message): 
        # Technical 
        tag = "technologies"
        path_to_df = "data/a4_db_data.csv"
        path_to_index = "data/a4_db_data.index"
        meta_columns = ['rank', 'title', 'publication_date', 'inventor', 'assignee', 'publication_number', 'language', 
            'thumbnail', 'pdf', 'page', 'entities']
        system_prompt = "You are an organizational agent, helping users with their technical questions related to the automotive industry."
        
    else: 
        system_prompt = "You are an organizational agent supposed to help employees of the company with any question they might have."
        context_prompt = [{"role": "user", "content": message.message}]
        meta_json_data = []
        ner_recommendations = []
           

    context = craft_agent_chat_context(system_prompt)
    # Send the message to the AI agent and get the response
    service = integrations.OpenAIIntegrationService(
        context = context,
        instruction = craft_agent_chat_instructions(
            conversation.agent.instructions,
            conversation.agent.response_shape
        )
    )

    service.add_chat_history(messages = chat_messages)


    if (intent == "1") or (intent == "2"):
        embedding = get_embedding(message.message)

        broad_context = retrieve(
            embedding,
            top_k           = 25,
            path_to_df      = path_to_df, 
            path_to_index   = path_to_index
            )
        
        # Reranker
        response = jina_reranker(message.message, broad_context, top_n=5)

        context = broad_context.iloc[[int(i['index']) for i in response['results']], :].reset_index()
        context_str = "\n".join(i['document']['text'] for i in response['results'])
        context_prompt =  generate_prompt(message.message, context_str)

        ner_recommendations = get_recommendations(context, broad_context, tag=tag)
        meta_json_data = context[meta_columns].to_json(orient='records')

        
    # fuzzy mathching
    
    chat_history = service.messages[:1] + service.messages[1:][-4:]
    response = service.answer_to_prompt(
        # We can test different OpenAI models.
        model               = "chat-model",
        prompt              = context_prompt,
        chat_history        = chat_history,
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
    

    log.info(f"Agent response: {response.get("answer")}")

    # Prepare response to the user
    api_response = agents.api.schemas.ConversationSaveResponse(
        conversation_id = message.conversation_id,
        response        = response.get("answer")
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

    return {"api_response": api_response, "relevant_history": chat_history, "metadata": meta_json_data, "ner_recommendations": ner_recommendations}

#### NICOLAS ADDED: MOVE APPROPRIATELY

def get_recommendations(context, broad_context, tag):
    # Get related NER docs
    context_ner = total_context_ner(context['entities'])[tag]

    df_excluded = broad_context.loc[(~broad_context["publication_number"].isin(context["publication_number"].values) & (broad_context['entities'].notnull()))]
    # sort by publication data and rank
    df_excluded.loc[:, "publication_date"] = pd.to_datetime(df_excluded['publication_date'], format="%Y-%m-%d")
    df_excluded = df_excluded.sort_values(by=['rank', 'publication_date'], ascending=[True, True])
    df_excluded.loc[:, "relevant_ner"] = np.nan
    
    if context_ner:
        context_ner_embed = embed(context_ner)
        i = 0
        for _, row in df_excluded.iterrows():
            comparison_list = json.loads(row["entities"])[tag]
            if comparison_list: # TODO:  fuzzy search // muss nur für test fall funktionieren
                comparison_embed = embed(comparison_list)

                # Check for overlap
                threshold = 0.50 
                overlap = check_overlap(context_ner_embed.data, comparison_embed.data, threshold)
                row["relevant_ner"] = overlap
                if overlap == True: 
                    i += 1
                if i == 3:
                    break

    ner_recommendations = df_excluded.loc[df_excluded["relevant_ner"] == True].to_json(orient='records')
    
    return ner_recommendations

def jina_reranker(msg, context, top_n):
    url = f"https://api.jina.ai/v1/rerank"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer " + os.getenv("JINA_KEY")
    }

    data = {
        "model": "jina-reranker-v1-base-en",
        "query": msg,
        "documents": context.text_chunks.to_list(),
        "top_n": top_n
    }

    # TODO: Manuals synthetic

    response = requests.post(url, headers=headers, json=data).json()
    return response


def get_user_intent(query: str, chat_history):
    prompt = [
        {
            "role": "system",
            "content": 
                "Classify the user input into one of two categories (1/ 2):\n\n"
                "(1) Administration and Organizational Information: This category includes queries related to administrative tasks, organizational procedures, and general information about the company's operations.\n"
                "(2) Patent and Manual Information for Engineering Questions: This category covers inquiries regarding patents, engineering manuals, technical documentation, and guidance related to engineering concepts and practices.\n\n"
                "Return only the number '1' or '2'. If neither is applicaable return '0'.\n\n"
                "Examples:\n"
                "Who won yesterday football match? > returns: 0\n"
                "How do I request time off from work? > returns: 1\n"
                "What are the steps to onboard a new employee? > returns: 1\n"
                "What information do you have on kitchen furniture? > returns: 0\n"
                "Where can I find the company's holiday schedule? > returns: 1\n"
                "I need information about the filing process for a new patent. > returns: 2\n"
                "How do I troubleshoot a malfunctioning machine according to the engineering manual? > returns: 2\n"
                "Are strawberries a fruit? > returns: 0\n"
                "What is the recommended maintenance schedule for our equipment? > returns: 2 > returns: 2\n"
                "Where can I locate the technical specifications for product X? > returns: 2\n"
                "Can you explain the process for obtaining approval for a new project? > returns: 1\n"
                "What safety protocols should be followed when operating heavy machinery? > returns: 1\n"
                "I'm looking for information on the latest software updates for our systems. > returns: 1\n"
        }] + chat_history + [{
            "role": "user",
            "content": query + " Return only the number 1 or 2. Take into consideration the previous messages."
        }]
        
    client = AzureOpenAI(
                    api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
                    api_version="2024-03-01-preview",
                    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
                )
    response = client.chat.completions.create(
        model       = "chat-model",
        messages    = prompt
    ).choices[0].message.content
    return response


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
        "organizations": [],

        "procedures": [],
        "departments": [],
        "projects": []
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


def retrieve_context(search_vector, df_condensed, index, top_k: int=25, distance_threshold=0.3) -> pd.DataFrame:
    """
    function to retrieve relevant documents from an index based on a user query
    - documents are used as context for the LLM

    :param query: user question
    :param top_k: top k results found in the index based on the user query

    :return: found context information based on the user query
    """

    # use FAISS index to search for context
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
    index.nprobe = 50  # increase the number of nearby cells to search too with `nprobe`
    k = min(top_k, index.ntotal)

    distances, ann = index.search(_vector, k=k)

    return distances, ann


def retrieve(search_vector: list[float], top_k: int, path_to_df, path_to_index) -> list:

    df_condensed = pd.read_csv(path_to_df)
    index = faiss.read_index(path_to_index)

    context = retrieve_context(search_vector, df_condensed, index, top_k=top_k)

    return context


def get_embedding(text_data: str) -> list[float]:
    """Retrieve a single embedding synchrone"""
    try:
        response = retriever_model_inference([text_data])
        return response
    except Exception as e:
        log.error(f"An error occurred requesting an embedding: {e}")

