# Bias Aware Agent

## Table of Contents
- [About the Project](#about-the-project)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## About the Project

BiasAwareAgent is an intelligent agent designed to assist users to retrieve least biased relevant information out of the corpus. Implemented using LangChain, LangGraph, ChromaDB, and OpenAI's GPT-4o, this project is based on the ReAct (Reasoning and Acting) Agent framework. 

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
    "collection": "gardening_store",
    "add_knowledge": true
}
```
After the initial run, set `add_knowledge` to `false` for subsequent runs.
```

The agent will guide you through setting up your crop plan for the current season and provide personalized tips based on your preferences.

## Features

- Intelligent gardening schedule planner
- Crop planning for the current season
- Personalized gardening tips using GPT-4
- Integration with ChromaDB for efficient data management
- Customizable user input and outputs

## Contributing

Contributions are what make the open-source community amazing! If you would like to contribute, please follow these steps:
1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

Karanbir Singh - [www.linkedin.com/in/singhkaranbir1](www.linkedin.com/in/singhkaranbir1)


## Acknowledgments

- [LangChain Documentation](https://python.langchain.com/)
- [OpenAI API](https://platform.openai.com/docs)
- [ChromaDB](https://www.chromadb.com)
- [LangGraph](https://github.com/langgraph/langgraph)