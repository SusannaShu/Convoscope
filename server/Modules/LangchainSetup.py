from langchain_community.chat_models import ChatOpenAI
from langchain_community.chat_models.azureml_endpoint import *
from langchain.chat_models.azure_openai import AzureChatOpenAI
from langchain.chat_models.azureml_endpoint import AzureMLChatOnlineEndpoint
from server_config import use_azure_openai, openai_api_key, azure_openai_api_key, azure_openai_api_base, azure_openai_api_gpt4_deployment, azure_openai_api_gpt35_deployment, azure_openai_api_gpt35_16k_deployment, azure_api_llama_13b_chat_endpoint, azure_api_llama_13b_chat_api_key

from langchain.chat_models.azure_openai import *
from constants import GPT_4_MODEL, GPT_4_MAX_TOKENS, GPT_TEMPERATURE, GPT_35_MODEL, GPT_35_MAX_TOKENS, GPT_35_16K_MODEL, GPT_35_16K_MAX_TOKENS

from langchain_community.callbacks import openai_info

def get_langchain_tiny():
    formatter = LlamaChatContentFormatter()

    return AzureMLChatOnlineEndpoint(
        endpoint_url=azure_api_llama_13b_chat_endpoint,
        endpoint_api_type='serverless',
        endpoint_api_key=azure_api_llama_13b_chat_api_key,
        content_formatter=formatter,
        model_kwargs={"temperature": 0.8, "max_new_tokens": 400}
    )

def get_langchain_gpt4(temperature=GPT_TEMPERATURE, model=GPT_4_MODEL, max_tokens=GPT_4_MAX_TOKENS):
    if use_azure_openai:
        return AzureChatOpenAI(openai_api_key=azure_openai_api_key, 
                            openai_api_base=azure_openai_api_base,
                            openai_api_version = "2024-02-15-preview",
                            deployment_name=azure_openai_api_gpt4_deployment,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            )
    else:
        return ChatOpenAI(temperature=GPT_TEMPERATURE, openai_api_key=openai_api_key, model=model, max_tokens=GPT_4_MAX_TOKENS)
    
def get_langchain_gpt35(temperature=GPT_TEMPERATURE, model=GPT_35_MODEL, max_tokens=GPT_35_MAX_TOKENS):
    if use_azure_openai:
        return AzureChatOpenAI(openai_api_key=azure_openai_api_key, 
                            openai_api_base=azure_openai_api_base,
                            openai_api_version = "2024-02-15-preview",
                            deployment_name=azure_openai_api_gpt35_deployment,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            )
    else:
        return ChatOpenAI(temperature=GPT_TEMPERATURE, openai_api_key=openai_api_key, model=model, max_tokens=GPT_35_MAX_TOKENS)

def get_langchain_gpt35_16k(temperature=GPT_TEMPERATURE, model=GPT_35_16K_MODEL, max_tokens=GPT_35_16K_MAX_TOKENS):
    if use_azure_openai:
        return AzureChatOpenAI(openai_api_key=azure_openai_api_key, 
                            openai_api_base=azure_openai_api_base,
                            openai_api_version = "2024-02-15-preview",
                            deployment_name=azure_openai_api_gpt35_16k_deployment,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            )
    else:
        return ChatOpenAI(temperature=GPT_TEMPERATURE, openai_api_key=openai_api_key, model=model, max_tokens=GPT_35_16K_MAX_TOKENS)
