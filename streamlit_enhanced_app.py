import streamlit as st
import os
import tempfile

# --- OpenAI API key from Streamlit secrets ---
openai_api_key = st.secrets["OPENAI_API_KEY"]

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# ---- UI Layout & Branding ----
st.set_page_config(page_title="Contract Retriever AI", layout="wide")
st.markdown(
    "<h1 style='text-align:center;'>🤖 Multi-Contract Retrieval Assistant</h1>"
    "<p style='text-align:center;'>Upload one or more contracts, ask questions, get clause-level AI answers.</p>",
    unsafe_allow_html=True
)

# ---- Query History in Sidebar ----
if 'history' not in st.session_state:
    st.session_state['history'] = []

with st.sidebar:
    st.header("🕑 Query History")
    if st.session_state['history']:
        for idx, item in enumerate(reversed(st.session_state['history'])):
            q = item['query']
            if st.button(f"{q[:40]}{'...' if len(q)>40 else ''}", key=f"hist_{idx}"):
                st.session_state['restore_query'] = q
    st.markdown("---")
    st.markdown("### 📄 Example Questions")
    ex_queries = [
        "What happens if the agreement is breached?",
        "When can the COMMISSION terminate this contract?",
        "What insurance coverage is required for the consultant?",
        "How are progress payments made?",
    ]
    for q in ex_queries:
        if st.button(q, key=f"exq_{q}"):
            st.session_state['restore_query'] = q
    st.markdown("---")
    st.markdown("""
        #### FAQ  
        - **What formats are supported?**  
          Upload any Markdown (.md) contract files.  
        - **Can I compare contracts?**  
          Yes! Upload multiple and select one or all for your query.
        - **Is my data safe?**  
          All files are processed only in this session, never saved.
        """)
    st.markdown("---")
    st.markdown("Built with ❤️ by your AI team.")

# ---- File Upload Section (Multi-file) ----
uploaded_files = st.file_uploader(
    "Upload one or more contract Markdown files (.md)", 
    type=["md"], 
    accept_multiple_files=True,
    help="You can upload several contracts for comparison. Max file size per file: 200MB"
)

if uploaded_files:
    # Save all uploaded files to temp directory
    temp_dir = tempfile.mkdtemp()
    contract_paths = []
    contract_names = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        contract_paths.append(file_path)
        contract_names.append(uploaded_file.name)

    st.success(f"{len(contract_paths)} contract(s) uploaded successfully!")

    # User selects one or more files for querying
    selected_files = st.multiselect(
        "Select contracts to include in your query:",
        options=contract_names,
        default=contract_names
    )

    # --- Build vectorstores per contract, cache for speed ---
    @st.cache_resource(show_spinner=False)
    def build_vectorstore(file_path):
        loader = TextLoader(file_path, autodetect_encoding=True)
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
        embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key=openai_api_key
        )
        vectorstore = FAISS.from_documents(split_docs, embeddings)
        return {
            "vectorstore": vectorstore,
            "sections": split_docs
        }

    contract_stores = {}
    for name, path in zip(contract_names, contract_paths):
        if name in selected_files:
            contract_stores[name] = build_vectorstore(path)

    # ---- Prompts and LLM chains (same as before) ----
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

    # ---- Main multi-contract query pipeline ----
    def multi_contract_pipeline(user_query: str, contract_stores):
        results = {}
        for name, store in contract_stores.items():
            retriever = store["vectorstore"].as_retriever(search_kwargs={"k": 4})
            docs = retriever.get_relevant_documents(user_query)
            retrieved_excerpts = "\n\n".join([doc.page_content for doc in docs])
            retrieval_output = retrieval_chain.run(query=user_query)
            clause_extractor_output = clause_extractor_chain.run(
                query=user_query,
                retrieved_excerpts=retrieved_excerpts
            )
            summary = summarizer_chain.run(
                query=user_query,
                extracted_clauses=clause_extractor_output
            )
            results[name] = {
                "retrieved_excerpts_raw": retrieved_excerpts,
                "retrieval_output": retrieval_output,
                "extracted_clauses": clause_extractor_output,
                "summary": summary
            }
        return results

    # ---- Main UI ----
    st.header("Ask a Question About Your Contract(s)")
    restored = st.session_state.get('restore_query', "")
    user_query = st.text_input(
        "Enter your contract/legal question:",
        value=restored,
        key="main_query"
    )
    st.session_state['restore_query'] = ""  # Reset after use

    if st.button("Submit", key="submit_query") and user_query:
        if not selected_files:
            st.error("Please select at least one contract file for analysis.")
            st.stop()

        with st.spinner("Processing your request with AI agents..."):
            results = multi_contract_pipeline(user_query, contract_stores)

        # Save to query history
        st.session_state['history'].append({
            "query": user_query,
            "selected_files": list(selected_files),
            "results": results
        })

        # Multi-file Results Display
        for name in selected_files:
            st.subheader(f"📄 Results for: {name}")
            contract_result = results[name]
            with st.expander("🔍 Retrieved Excerpts", expanded=True):
                st.write(contract_result["retrieved_excerpts_raw"])
            with st.expander("📑 Extracted Clauses", expanded=True):
                st.write(contract_result["extracted_clauses"])
            with st.expander("📝 Final Summary", expanded=True):
                st.success(contract_result["summary"])

            # Download for this file
            result_text = (
                f"Query: {user_query}\n\n"
                f"File: {name}\n\n"
                "----- Retrieved Excerpts -----\n"
                f"{contract_result['retrieved_excerpts_raw']}\n\n"
                "----- Extracted Clauses -----\n"
                f"{contract_result['extracted_clauses']}\n\n"
                "----- Final Summary -----\n"
                f"{contract_result['summary']}"
            )
            st.download_button(
                label=f"Download Results for {name}",
                data=result_text,
                file_name=f"{name}_query_results.txt",
                mime="text/plain",
                key=f"dl_{name}"
            )

else:
    st.info("Please upload at least one contract markdown file to get started.")

