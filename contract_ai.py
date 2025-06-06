import os
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# 1. Load parsed markdown file
markdown_path = "contract_markdown.md"  # Replace with your converted .md file
loader = TextLoader(markdown_path, autodetect_encoding=True)
docs = loader.load()

# 2. Split by markdown headers
splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
)

split_docs = []
for doc in docs:
    split_docs.extend(splitter.split_text(doc.page_content))


# 3. Embed & index
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
vectorstore = FAISS.from_documents(split_docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# 4. Define Agents (Workers) and Supervisor Logic

retrieval_prompt = """As a Retrieval agent, your task is to thoroughly investigate the uploaded knowledge base using semantic search.
Focus on identifying sections and clauses most relevant to the user’s question.
Do not include loosely related or redundant information. Avoid assumptions or extrapolation.
Output: Provide a concise collection of retrieved contract excerpts that closely match the user's query. This will be passed on to the Clause extractor for further examination.
User query: {query}
Retrieved Excerpts:
"""

clause_extractor_prompt = """As a Clause extractor, analyze the retrieved contract excerpts and extract specific clauses directly related to the user’s query.
Extract only those parts that contain direct legal language or clauses pertinent to the query (e.g., penalty clauses, SLAs, deliverables).
Do not summarize or reword. Do not include irrelevant content.
Output: Return a bullet-point list of directly extracted clauses. This output will be provided to the Summarizer for final interpretation.
User query: {query}
Retrieved Excerpts:
{retrieved_excerpts}
Extracted Clauses:
"""

summarizer_prompt = """As a Summarizer, synthesize the extracted legal clauses into a clear and concise answer to the user's query.
Read the extracted clauses carefully and interpret them in context of the original user query.
Compose a short, factual summary (2–4 sentences) for a legal reviewer or business executive.
Do not hallucinate, embellish, or introduce external information.
Output: 2–4 sentence summary addressing the user’s query. Where appropriate, include the clause title or section reference.
User query: {query}
Extracted Clauses:
{extracted_clauses}
Summary:
"""

# --- Chains for each worker agent ---
llm = ChatOpenAI(model="gpt-4o", temperature=0.5, streaming=True)

retrieval_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(input_variables=["query"], template=retrieval_prompt)
)
clause_extractor_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(input_variables=["query", "retrieved_excerpts"], template=clause_extractor_prompt)
)
summarizer_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(input_variables=["query", "extracted_clauses"], template=summarizer_prompt)
)

# 5. Supervisor Function
def contract_query_pipeline(user_query: str):
    # Step 1: Retrieval Agent
    docs = retriever.get_relevant_documents(user_query)
    retrieved_excerpts = "\n\n".join([doc.page_content for doc in docs])
    retrieval_output = retrieval_chain.run(query=user_query)
    
    # Step 2: Clause Extractor Agent
    clause_extractor_output = clause_extractor_chain.run(
        query=user_query,
        retrieved_excerpts=retrieved_excerpts
    )
    
    # Step 3: Summarizer Agent
    summary = summarizer_chain.run(
        query=user_query,
        extracted_clauses=clause_extractor_output
    )
    return summary

# 6. Run the pipeline
if __name__ == "__main__":
    user_query = input("Enter your contract/legal question: ")
    summary = contract_query_pipeline(user_query)
    print("\n--- Final Answer ---\n", summary)
