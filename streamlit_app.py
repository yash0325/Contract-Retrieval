import streamlit as st
import os

# For secrets management (OpenAI key)
openai_api_key = st.secrets["OPENAI_API_KEY"]

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# UI setup
st.set_page_config(page_title="Contract Retrieval Debug", layout="centered")
st.title("🛠️ Contract Retrieval & Clause Summarization (with Debugging)")

uploaded_file = st.file_uploader("Upload your contract as a markdown (.md) file", type=["md"])

# Initialize results dictionary for debugging
debug_results = {}

if uploaded_file is not None:
    # Save uploaded file
    file_path = "uploaded_contract.md"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    st.success("Contract uploaded successfully!")

    # 1. Load document
    try:
        loader = TextLoader(file_path, autodetect_encoding=True)
        docs = loader.load()
        debug_results['loaded_docs'] = [doc.page_content[:300] for doc in docs]  # Preview first 300 chars
        st.info(f"✅ Document loaded. First 300 chars:\n{debug_results['loaded_docs'][0]}")
    except Exception as e:
        st.error(f"❌ Error loading document: {e}")

    # 2. Split by markdown headers
    try:
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
        debug_results['split_docs'] = [doc.page_content[:300] for doc in split_docs]  # First 300 chars of each chunk
        st.info(f"✅ Document split. Example split section:\n{debug_results['split_docs'][0]}")
        st.write("Total split sections:", len(split_docs))
    except Exception as e:
        st.error(f"❌ Error splitting document: {e}")

    # 3. Embed & index
    try:
        embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key=openai_api_key
        )
        vectorstore = FAISS.from_documents(split_docs, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        st.success("✅ Embeddings and vectorstore created successfully.")
    except Exception as e:
        st.error(f"❌ Error during embedding/indexing: {e}")

    # 4. Define agent prompts & chains
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

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.5,
        streaming=True,
        openai_api_key=openai_api_key
    )

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
        # Step 1: Retrieval agent
        docs = retriever.get_relevant_documents(user_query)
        retrieved_excerpts = "\n\n".join([doc.page_content for doc in docs])
        st.info(f"🔎 Retriever found {len(docs)} sections. Example excerpt:\n{retrieved_excerpts[:300]}")
        debug_results['retrieved_excerpts'] = retrieved_excerpts

        # Step 2: Retrieval chain (LLM)
        retrieval_output = retrieval_chain.run(query=user_query)
        st.info("Retrieval agent LLM output (first 500 chars):")
        st.code(retrieval_output[:500])
        debug_results['retrieval_output'] = retrieval_output

        # Step 3: Clause extractor agent (LLM)
        clause_extractor_output = clause_extractor_chain.run(
            query=user_query,
            retrieved_excerpts=retrieved_excerpts
        )
        st.info("Clause extractor LLM output (first 500 chars):")
        st.code(clause_extractor_output[:500])
        debug_results['clause_extractor_output'] = clause_extractor_output

        # Step 4: Summarizer agent (LLM)
        summary = summarizer_chain.run(
            query=user_query,
            extracted_clauses=clause_extractor_output
        )
        st.info("Summarizer LLM output (first 500 chars):")
        st.code(summary[:500])
        debug_results['summary'] = summary

        return {
            "retrieved_excerpts": retrieval_output,
            "extracted_clauses": clause_extractor_output,
            "summary": summary
        }

    # 5. User input and trigger pipeline
    st.header("Ask a Question About Your Contract")
    user_query = st.text_input("Enter your contract/legal question:", "")

    if st.button("Submit") and user_query:
        with st.spinner("Processing with AI agents..."):
            results = contract_query_pipeline(user_query)

        st.markdown("## 🔍 Retrieved Excerpts")
        st.write(results["retrieved_excerpts"])
        st.markdown("## 📑 Extracted Clauses")
        st.write(results["extracted_clauses"])
        st.markdown("## 📝 Final Summary")
        st.success(results["summary"])
else:
    st.info("Please upload a contract markdown file to get started.")

