import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables (for OpenAI keys, etc.)
load_dotenv()

# --- LangChain & Workflow Imports ---
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# 1. App Title and Description
st.set_page_config(page_title="AI Contract Retrieval Assistant", layout="centered")
st.title("🤖 Contract Retrieval & Clause Summarization App")
st.markdown(
    "Ask any question about your uploaded contract, and get relevant clauses and a concise, business-friendly summary."
)

# 2. File Upload
uploaded_file = st.file_uploader(
    "Upload your contract as a markdown (.md) file", type=["md"]
)

if uploaded_file is not None:
    with open("uploaded_contract.md", "wb") as f:
        f.write(uploaded_file.read())

    st.success("Contract uploaded successfully!")

    # 3. Build LangChain Objects (on file upload)
    loader = TextLoader("uploaded_contract.md", autodetect_encoding=True)
    docs = loader.load()
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

    # 4. Embed & Index
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # 5. Define Prompts & Chains
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

    def contract_query_pipeline(user_query: str):
        # Retrieval
        docs = retriever.get_relevant_documents(user_query)
        retrieved_excerpts = "\n\n".join([doc.page_content for doc in docs])
        # Retrieval agent
        retrieval_output = retrieval_chain.run(query=user_query)
        # Clause extractor agent
        clause_extractor_output = clause_extractor_chain.run(
            query=user_query,
            retrieved_excerpts=retrieved_excerpts
        )
        # Summarizer agent
        summary = summarizer_chain.run(
            query=user_query,
            extracted_clauses=clause_extractor_output
        )
        return {
            "retrieved_excerpts": retrieval_output,
            "extracted_clauses": clause_extractor_output,
            "summary": summary
        }

    # 6. User Query Input
    st.header("Ask a Question About Your Contract")
    user_query = st.text_input("Enter your contract/legal question:", "")

    if st.button("Submit") and user_query:
        with st.spinner("Processing your request with AI agents..."):
            results = contract_query_pipeline(user_query)

        st.subheader("🔍 Retrieved Excerpts")
        st.write(results["retrieved_excerpts"])
        st.subheader("📑 Extracted Clauses")
        st.write(results["extracted_clauses"])
        st.subheader("📝 Final Summary")
        st.success(results["summary"])
else:
    st.info("Please upload a contract markdown file to get started.")

