import streamlit as st
import os
import tempfile

# OpenAI API key from Streamlit secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]

from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# ---- UI Layout & Branding ----
st.set_page_config(page_title="Contract Retriever AI", layout="wide")
st.markdown(
    "<h1 style='text-align:center;'>🤖 Multi-Format Contract Retrieval Assistant</h1>"
    "<p style='text-align:center;'>Upload Markdown, TXT, PDF, or DOCX contracts and query them with AI.</p>",
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
          Markdown (.md), Text (.txt), PDF (.pdf), and Word (.docx)
        - **Can I compare contracts?**  
          Yes! Upload multiple and select one or all for your query.
        - **Is my data safe?**  
          All files are processed only in this session, never saved.
        """)
    st.markdown("---")
    st.markdown("Built with ❤️ by your AI team.")

# ---- File Upload Section (Multi-file, Multi-format) ----
allowed_types = ["md", "txt", "pdf", "docx"]
uploaded_files = st.file_uploader(
    "Upload contract files (.md, .txt, .pdf, .docx)", 
    type=allowed_types, 
    accept_multiple_files=True,
    help="You can upload several contracts for comparison."
)

def load_and_split(file_path, ext):
    """
    Loads file with the appropriate loader, applies the best splitter.
    Returns list of Document objects.
    """
    if ext == "md":
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
        return split_docs

    elif ext == "txt":
        loader = TextLoader(file_path, autodetect_encoding=True)
        docs = loader.load()
        # Use a recursive character splitter for .txt
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        split_docs = []
        for doc in docs:
            split_docs.extend(splitter.create_documents([doc.page_content]))
        return split_docs

    elif ext == "pdf":
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        # Use recursive splitter, since PDF headings may be inconsistent
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        split_docs = []
        for doc in docs:
            split_docs.extend(splitter.create_documents([doc.page_content]))
        return split_docs

    elif ext == "docx":
        loader = UnstructuredWordDocumentLoader(file_path)
        docs = loader.load()
        # Use recursive splitter for .docx
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        split_docs = []
        for doc in docs:
            split_docs.extend(splitter.create_documents([doc.page_content]))
        return split_docs

    else:
        raise ValueError(f"Unsupported file extension: {ext}")

if uploaded_files:
    temp_dir = tempfile.mkdtemp()
    contract_paths = []
    contract_names = []
    contract_exts = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        contract_paths.append(file_path)
        contract_names.append(uploaded_file.name)
        contract_exts.append(uploaded_file.name.split(".")[-1].lower())

    st.success(f"{len(contract_paths)} contract(s) uploaded successfully!")

    # User selects one or more files for querying
    selected_files = st.multiselect(
        "Select contracts to include in your query:",
        options=contract_names,
        default=contract_names
    )

    # --- Build vectorstores per contract, cache for speed ---
    @st.cache_resource(show_spinner=False)
    def build_vectorstore(file_path, ext):
        split_docs = load_and_split(file_path, ext)
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
    for name, path, ext in zip(contract_names, contract_paths, contract_exts):
        if name in selected_files:
            try:
                contract_stores[name] = build_vectorstore(path, ext)
            except Exception as e:
                st.error(f"Failed to process {name}: {e}")

    # ---- Prompts and LLM chains ----
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

    judge_prompt = """
You are an impartial evaluator. Compare the following two answers to the user's contract/legal question. 
Answer A was generated by a multi-agent retrieval pipeline, and Answer B was generated by prompting the latest GPT-4o model directly.
Evaluate both for accuracy, completeness, relevance, and clarity.
Choose the better answer overall, and explain your reasoning.

User Query:
{query}

Answer A (multi-agent pipeline):
{answer_a}

Answer B (direct GPT-4o prompt):
{answer_b}

Your Evaluation:
1. Strengths/weaknesses of each answer
2. Which is better and why?
3. Final verdict: "Answer A is better", "Answer B is better", or "Both are equally good"
"""

    judge_chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate(
            input_variables=["query", "answer_a", "answer_b"],
            template=judge_prompt
        )
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

            # Generate a direct GPT-4o baseline answer using the same retrieved excerpts
            direct_prompt = f"""
Given the following contract excerpt, answer the user's question concisely and accurately. Do not hallucinate or guess.
Contract excerpt:
{retrieved_excerpts}

User question:
{user_query}
"""
            direct_gpt_output = llm(direct_prompt)

            # Run judge agent to compare outputs
            judge_output = judge_chain.run(
                query=user_query,
                answer_a=summary,
                answer_b=direct_gpt_output
            )

            results[name] = {
                "retrieved_excerpts_raw": retrieved_excerpts,
                "retrieval_output": retrieval_output,
                "extracted_clauses": clause_extractor_output,
                "summary": summary,
                "direct_gpt_output": direct_gpt_output,
                "judge_output": judge_output
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
            with st.expander("🆚 Direct GPT-4o Answer", expanded=False):
                st.write(contract_result["direct_gpt_output"])
            with st.expander("🤖 GPT-Based Judge Evaluation", expanded=True):
                st.write(contract_result["judge_output"])
            result_text = (
                f"Query: {user_query}\n\n"
                f"File: {name}\n\n"
                "----- Retrieved Excerpts -----\n"
                f"{contract_result['retrieved_excerpts_raw']}\n\n"
                "----- Extracted Clauses -----\n"
                f"{contract_result['extracted_clauses']}\n\n"
                "----- Final Summary -----\n"
                f"{contract_result['summary']}\n\n"
                "----- Direct GPT-4o Answer -----\n"
                f"{contract_result['direct_gpt_output']}\n\n"
                "----- GPT-Based Judge Evaluation -----\n"
                f"{contract_result['judge_output']}"
            )
            st.download_button(
                label=f"Download Results for {name}",
                data=result_text,
                file_name=f"{name}_query_results.txt",
                mime="text/plain",
                key=f"dl_{name}"
            )

else:
    st.info("Please upload at least one contract file (.md, .txt, .pdf, .docx) to get started.")
