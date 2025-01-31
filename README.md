# Bias Aware Agent

## Table of Contents
- [About the Project](#about-the-project)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [License](#license)
- [Authors](#authors)

## About the Project

This repository contains the implementation of Bias-Aware Agent, a novel AI-driven framework designed to enhance fairness in Agentic systems by detecting and mitigating bias in retrieved content. The agent integrates bias detection tools within an agentic framework, leveraging ReAct-based reasoning to dynamically assess and flag bias in retrieved knowledge. Implemented using LangChain, LangGraph, ChromaDB, and OpenAI's GPT-4o.

## Getting Started

Instructions on how to get a copy of the project running on your local machine.

### Prerequisites

Ensure the following software and packages are installed:
```sh
- Python 3.x
- LangChain
- LangGraph
- ChromaDB
- OpenAI Python SDK
```

### Installation

Step-by-step guide on how to install and set up the project.
```sh
1. Clone the repo
   git clone https://github.com/SinghKaranbir/BiasAwareAgent.git

2. Navigate to the project directory
   cd BiasAwareAgent

3. (Optional but recommended) Create a virtual environment
   python -m venv venv

4. Activate the virtual environment
   # On Windows
   venv\Scripts\activate
   
   # On macOS and Linux
   source venv/bin/activate

5. Install dependencies
   pip install -r requirements.txt

6. Pull and run ChromaDB using Docker
   docker pull chromadb/chromadb
   docker run -p 8000:8000 -v PATH_TO_DIR:/chroma/chroma chromadb/chroma

7. Update the `config.json` file with the following ChromaDB configuration. When running for the first time, set `add_knowledge` to `true`:
```json
"chroma": {
    "host": "localhost",
    "port": 8000,
    "collection": "news_articles",
    "add_knowledge": true
}```
```

After the initial run, set `add_knowledge` to `false` for subsequent runs.

## Authors
- **Karanbir Singh** (Salesforce, ILM Cloud)
- **William Ngu** (Salesforce, ILM Cloud)

## Experimental Results
- **F1-score**: 0.795 (Weighted Average)
- **Bias Detection Accuracy**: ~79%
- **Confidence Level**: 0.821 on average

## Citation
If you use this work, please cite:
```bibtex
@inproceedings{singh2024biasaware,
  author = {Karanbir Singh and William Ngu},
  title = {Bias-Aware Agent: Enhancing Fairness in AI-Driven Knowledge Retrieval},
  booktitle = {Companion Proceedings of the Web Conference},
  year = {2025}
}
```

## License
Distributed under the MIT License.

## Acknowledgments

- [LangChain Documentation](https://python.langchain.com/)
- [OpenAI API](https://platform.openai.com/docs)
- [ChromaDB](https://www.chromadb.com)
- [LangGraph](https://github.com/langgraph/langgraph)
