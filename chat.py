import pandas as pd
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
# Upgraded to a powerful multilingual embedding pipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

def process_qa_excel(file):
    """
    Reads the specific QA structure from the Excel sheets,
    and indexes the questions into the FAISS vector database.
    """
    documents = []
    df = pd.read_excel(file)
    
    q_col, a_col = None, None
    for col in df.columns:
        if col in ['Questions', 'မေးခွန်းများ']:
            q_col = col
        if col in ['Answers', 'အဖြေများ']:
            a_col = col

    if not q_col or not a_col:
        raise ValueError(f"Could not find valid Question/Answer columns in {file.name}")

    for idx, row in df.iterrows():
        if pd.isna(row[q_col]) or pd.isna(row[a_col]):
            continue
            
        question = str(row[q_col]).strip()
        answer = str(row[a_col]).strip()
        
        doc = Document(
            page_content=question,
            metadata={
                "source": file.name,
                "row": idx + 1,
                "ground_truth_answer": answer
            }
        )
        documents.append(doc)
        
    return documents

def build_vector_store(uploaded_files, mistral_api_key):
    """Combine documents and build the FAISS index using a multilingual model."""
    all_documents = []
    
    for file in uploaded_files:
        if file.name.endswith(('.xlsx', '.xls')):
            docs = process_qa_excel(file)
            all_documents.extend(docs)
            
    if not all_documents:
        return None
        
    # FIX 1: Upgraded model to a specialized multilingual variant that natively handles Myanmar script
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    vector_store = FAISS.from_documents(all_documents, embeddings)
    return vector_store

def get_qa_chain(vector_store, mistral_api_key):
    """Configures Mistral to process the matched bilingual question/answer pairs cleanly."""
    # Use mistral-large-latest or mistral-small-latest depending on access tier
    llm = ChatMistralAI(mistral_api_key=mistral_api_key, model="mistral-small-latest", temperature=0.1)
    
    # Get the single absolute best match (k=1) to prevent prompt cluttering
    retriever = vector_store.as_retriever(search_kwargs={"k": 1})
    
    # FIX 2: Streamlined the instruction template completely to eliminate character encoding drops
    system_prompt = (
        "You are an organizational customer support assistant.\n"
        "You must answer the user's question using ONLY the Ground-Truth Answer provided in the context below.\n"
        "If the user asks in Myanmar, reply with the Myanmar answer. If they ask in English, reply with the English answer.\n"
        "Do not invent facts outside of the context.\n\n"
        "Context:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    document_prompt = ChatPromptTemplate.from_template(
        "Ground-Truth Answer: {ground_truth_answer}"
    )
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt, document_variable_name="context", document_prompt=document_prompt)
    retrieval_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return retrieval_chain
