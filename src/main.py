import chromadb
import json

from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langgraph.graph import MessagesState
from langgraph.graph import START, StateGraph
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_community.document_loaders import TextLoader

import tools
import sys

def load_config():
    # Load Configuration
    with open("config/config.json") as json_data_file:
        config = json.load(json_data_file)
    
    return config


def init_vector_store(config):
    chroma_config = config["chroma"]
    chroma_client = chromadb.HttpClient(host=chroma_config["host"], port=chroma_config["port"])
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma(client=chroma_client,
           collection_name=chroma_config["collection"],
           embedding_function=embeddings,
           create_collection_if_not_exists=True)
    
    return vector_store


def add_documents(vector_store: Chroma):
    # Step 1: Load the text file
    file_path = "data/news_articles.txt"  # Replace with your file path
    loader = TextLoader(file_path)
    documents = loader.load()
    # Step 2: Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=25)
    texts = text_splitter.split_documents(documents)
    vector_store.add_documents(texts)

def reasoner(state: MessagesState):
   return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

if __name__ == '__main__':
    # Load Configuration
    config = load_config()
    # Initialize Chroma Vector Store
    vector_store = init_vector_store(config=config)
    # Add Knowledge to the store if enabled
    if config["chroma"]["add_knowledge"]:
        add_documents(vector_store)
    # Tools that are available for Bias Aware Agent
    available_tools = [tools.news_articles_retrieval_tool(vector_store), tools.bias_detector]
    # Choose the LLM to use
    llm = ChatOpenAI(model="gpt-4o",
                     temperature=0)
    # Bind tools with LLM
    llm_with_tools = llm.bind_tools(available_tools)
    # System message
    sys_msg = SystemMessage(content="You are a highly advanced bias detection system designed to analyze retrieved news articles for bias. Your task is to:"
                            "1. Answer the query based on the content of the article in a concise and factual manner."
                            "2. Analyze the retrieved content for bias by utilizing tools available"
                            "3. Provide a bias evaluation: If bias is output: This content contains bias. Include a brief explanation of why the content is biased, citing specific examples. If no bias is detected, output: This content appears unbiased")
    
    # Graph
    builder = StateGraph(MessagesState)
    # Add nodes
    builder.add_node("reasoner", reasoner)
    builder.add_node("tools", ToolNode(available_tools))
    # Add edges
    builder.add_edge(START, "reasoner")
    builder.add_conditional_edges(
        "reasoner",
        tools_condition,
    )
    builder.add_edge("tools", "reasoner")
    react_graph = builder.compile()
    # query = input("Please enter your query: ")
    query=sys.argv[1]
    messages = [HumanMessage(content=query)]
    messages = react_graph.invoke({"messages": messages})
    for m in messages['messages']:
        m.pretty_print()