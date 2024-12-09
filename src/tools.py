from langchain.tools.retriever import create_retriever_tool
from langchain.tools import tool
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
from transformers import AutoTokenizer


def news_articles_retrieval_tool(vector_store):
    retriever = vector_store.as_retriever()
    retriever_tool = create_retriever_tool(
        retriever,
        "NewsArticlesCorpus",
        "Search for news articles content")
    
    return retriever_tool

@tool
def bias_detector(passage: str):
    """
    Detects the bias in the retrieved content

    Args:
    passage: a string which is retrieved

    Returns:
    bias_classification, classification which tells whethere is bias is detected or not
    probability, how much bias is present
    """
    tokenizer = AutoTokenizer.from_pretrained("d4data/bias-detection-model")
    model = AutoModelForSequenceClassification.from_pretrained("d4data/bias-detection-model", from_tf=True)
    classifier = pipeline('text-classification', model=model, tokenizer=tokenizer)
    out = classifier(passage) 
    classi_out = out[0]['label']
    probability = out[0]['score']

    return classi_out, probability

