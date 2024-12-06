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
from langchain_community.document_loaders.tsv import UnstructuredTSVLoader
from langchain_text_splitters import TokenTextSplitter


from src import tools


def load_config():
    # Load Configuration
    with open("config/config.json") as json_data_file:
        config = json.load(json_data_file)
    
    return config

def init_vector_store(config):
    chroma_config = config["chroma"]
    chroma_client = chromadb.HttpClient(host=chroma_config["host"], port=chroma_config["port"])
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = Chroma(client=chroma_client,
           collection_name=chroma_config["collection"],
           embedding_function=embeddings,
           create_collection_if_not_exists=True)
    
    return vector_store


def add_documents(vector_store: Chroma):
    text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=50)
    loader = UnstructuredTSVLoader(file_path="data/collection.tsv", mode="single")
    docs = loader.load_and_split([text_splitter])
    print("Documents loaded")
    vector_store.add_documents(docs)

def reasoner(state: MessagesState):
   return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

if __name__ == '__main__':
    # Load Configuration
    config = load_config()
    # Initialize Chroma Vector Store
    vector_store = init_vector_store(config=config)
    # Add Knowledge to the store if enabled
    if config["chroma"]["add_knowledge"]:
        add_documents(vector_store=vector_store)

    # Tools that is available for Gardening Agent
    available_tools = [tools.msmarco_passage_knowledge_tool(vector_store), tools.bias_detector]
    # Choose the LLM to use
    llm = ChatOpenAI(model="gpt-4o",
                     temperature=0)
    # Bind tools with LLM
    llm_with_tools = llm.bind_tools(available_tools)
    # System message
    sys_msg = SystemMessage(content="You are an expert detecting bias use tools only to detect whether the bias is present in the retrieved text")
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

    messages = [HumanMessage(content=f"how to get big thighs and hips")]
    messages = react_graph.invoke({"messages": messages})
    for m in messages['messages']:
        m.pretty_print()