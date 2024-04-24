# Atlee: Your AI-powered Personalized Insight Generator from PDFs

Atlee stands as a cutting-edge Streamlit application, harnessing the prowess of Anthropics advanced Generative AI, particularly the Claude model. It's your gateway to extracting and analyzing text, images and tables from PDF documents, utilizing a sophisticated Retrieval-Augmented Generation (RAG) technique. This enables precise, contextually rich responses to user queries, all derived from the content of uploaded documents.

## Features

- **Instant Insights**: Extracts and analyses text, images and tables from uploaded PDF documents to provide instant insights.
- **Retrieval-Augmented Generation**: Utilizes Anthropics Generative AI model Claude for high-quality, contextually relevant answers.
- **Personalized Insights**: Tailor your Document experience to your preferences and interests by providing input on topics and areas of interest.

## Getting Started

### Prerequisites

- Anthropic API Key: Obtain a Claude API key to interact with Claude Generative AI models. Visit [Anthropic API Key Setup]([https://docs.anthropic.com/claude/docs/getting-access-to-claude]) to get your key.
- Streamlit: This application is built with Streamlit. Ensure you have Streamlit installed in your environment.

## How to Use

## Start the Application: Launch the Streamlit application by running the command:

streamlit run <path_to_script.py>
Replace <path_to_script.py> with the path to the script file.

**Enter Your Anthropic API Key:** Securely enter your Claude API key. This key enables the application to access Anthropics Generative AI models.

**Upload PDF Documents:** You can upload one PDF documents. The application will analyze the content of these documents to respond to queries.

**Ask Questions:** Once your documents are processed, you can ask any question related to the content of your uploaded documents.

## Technical Overview:

**PDF Processing:**
The system utilizes the partition_pdf function from the unstructured module to extract text, images, and tables from PDF documents.
It employs a strategy to extract content based on titles within the document, ensuring a structured approach to content extraction.

**Text Chunking:**
The extracted text from PDF documents is chunked using LangChain, a tool that segments large bodies of text into smaller, more manageable chunks.
Chunking helps in organizing and processing large volumes of text efficiently, enabling better analysis and understanding.

**Vector Store Creation:**
The system utilizes the Sentence Transformer pre-trained model, specifically the "sentence-transformers/all-mpnet-base-v2" model, for generating embeddings from the text chunks. These embeddings are then used to create a searchable vector store in Pinecone Vector DB.
Pinecone Vector DB provides a scalable and efficient solution for storing and querying high-dimensional vectors, making it suitable for similarity search tasks.

**Answer Generation:**
LangChain is used for generating answers to user queries based on the context provided by the uploaded documents.
The system employs a conversational approach, leveraging memory of past interactions and the context from the uploaded documents to generate relevant and personalized responses. It integrates with various AI models, such as ChatAnthropic to generate responses tailored to user queries.


### Installation

Clone this repository or download the source code to your local machine. Navigate to the application directory and install the required Python packages:
```bash
pip install -r requirements.txt
