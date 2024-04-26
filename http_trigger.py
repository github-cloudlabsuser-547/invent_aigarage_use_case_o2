import azure.functions as func
import logging
import os
from openai import AzureOpenAI
import openai
import json

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)


def establish_client():
    client = AzureOpenAI(
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
            api_version="2024-03-01-preview"
            )
    return client


@app.function_name(name="HttpTrigger")
@app.route(route="http_trigger", auth_level=func.AuthLevel.ANONYMOUS)
def http_trigger(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    try:
        req_body = req.get_json()
        query = req_body.get('query')

        if not query:
                return func.HttpResponse(
                    "Please pass a prompt in the request body.",
                    status_code=400
                )

        deployment_name = "gpt-35-turbo-instruct"
        client = establish_client()
        response = client.completions.create(model=deployment_name, prompt=query, max_tokens=50)
        reponse_clean = response.choices[0].text.strip()
        json_output =  str({'response': reponse_clean})


        return func.HttpResponse(
             json_output,
             status_code=200
             )
    

    except Exception as e:
        return func.HttpResponse(
            str(e),
            status_code=500
        )