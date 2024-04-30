

from openai import AzureOpenAI
import os
import dotenv
import numpy as np
dotenv.load_dotenv()


def retriever_model_inference(text_input: list[str]):

    print(True)

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

