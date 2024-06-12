
import os
import json
from typing import List
import pymongo
from dotenv import load_dotenv
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.vectorstores.azure_cosmos_db import AzureCosmosDBVectorSearch
from langchain.schema.document import Document
from langchain.agents import Tool
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain.tools import StructuredTool
from langchain_core.messages import SystemMessage

load_dotenv(".env")
DB_CONNECTION_STRING = os.environ.get("DB_CONNECTION_STRING")
AOAI_ENDPOINT = os.environ.get("AOAI_ENDPOINT")
AOAI_KEY = os.environ.get("AOAI_KEY")
AOAI_API_VERSION = "2023-09-01-preview"
COMPLETIONS_DEPLOYMENT = "hackathon-0"
EMBEDDINGS_DEPLOYMENT = "embedding"
db = pymongo.MongoClient(DB_CONNECTION_STRING).jobs

class AIAgent:
    
    def __init__(self, session_id: str):
        llm = AzureChatOpenAI(
            temperature = 0,
            openai_api_version = AOAI_API_VERSION,
            azure_endpoint = AOAI_ENDPOINT,
            openai_api_key = AOAI_KEY,
            azure_deployment = COMPLETIONS_DEPLOYMENT
        )
        self.embedding_model = AzureOpenAIEmbeddings(
            openai_api_version = AOAI_API_VERSION,
            azure_endpoint = AOAI_ENDPOINT,
            openai_api_key = AOAI_KEY,
            azure_deployment = EMBEDDINGS_DEPLOYMENT,
            chunk_size=10
        )
        system_message = SystemMessage(
            content = """
                You are a helpful, friendly job interviewer for a company. 
                You are designed to conduct interviews for jobs and also answer applicants question about the job role from the list.

                Only answer applicant questions related to the interview and job provided in the list of jobs below that are represented
                in JSON format.

                If you are asked a question that is not in the list, respond with "we have not decided yet, but once decided will get back to you with the info"

                List of jobs:
                """
        )
        self.agent_executor = create_conversational_retrieval_agent(
                llm,
                self.__create_agent_tools(),
                system_message = system_message,
                memory_key=session_id,
                verbose=True
        )

    def run(self, prompt: str) -> str:
        """
        Run the AI agent.
        """
        result = self.agent_executor({"input": prompt})
        return result["output"]

    def __create_interview_vector_store_retriever(
            self,
            collection_name: str,
            top_k: int = 3
        ):
        """
        Returns a vector store retriever for the given collection.
        """
        vector_store =  AzureCosmosDBVectorSearch.from_connection_string(
            connection_string = DB_CONNECTION_STRING,
            namespace = f"jobs.{collection_name}",
            embedding = self.embedding_model,
            index_name = "VectorSearchIndex",
            embedding_key = "contentVector",
            text_key = "_id"
        )
        return vector_store.as_retriever(search_kwargs={"k": top_k})

    def __create_agent_tools(self) -> List[Tool]:
        """
        Returns a list of agent tools.
        """
        jobs_retriever = self.__create_interview_vector_store_retriever("jobs")

        # create a chain on the retriever to format the documents as JSON
        jobs_retriever_chain = jobs_retriever | format_docs

        tools = [
            Tool(
                name = "vector_search_jobs",
                func = jobs_retriever_chain.invoke,
                description = """
                    Searches jobs database information for similar information 
                    on the question. Returns the job information in JSON format.
                    """
            ),
            StructuredTool.from_function(get_job_by_id)]
        return tools

def format_docs(docs:List[Document]) -> str:
    """
    Prepares the jobs list for the system prompt.
    """
    str_docs = []
    for doc in docs:
        # Build the job document without the contentVector
        doc_dict = {"_id": doc.page_content}
        doc_dict.update(doc.metadata)
        if "contentVector" in doc_dict:
            del doc_dict["contentVector"]
        str_docs.append(json.dumps(doc_dict, default=str))
    # Return a single string containing each job JSON representation
    # separated by two newlines
    return "\n\n".join(str_docs)

def get_job_by_id(job_id: str) -> str:
    """
    Retrieves a product by its ID.    
    """
    doc = db.jobs.find_one({"_id": job_id})
    if "contentVector" in doc:
        del doc["contentVector"]
    return json.dumps(doc)
