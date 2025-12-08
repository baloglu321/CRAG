from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from functools import partial
import chromadb
from sentence_transformers import CrossEncoder
from croma_db_update import update_db_with_feedback
from langchain_core.documents import Document
from langgraph.graph import END, StateGraph
from typing import List, TypedDict, Literal
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from tavily import TavilyClient
import time
import os

tavily = TavilyClient(api_key="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")
CLOUDFLARE_TUNNEL_URL = (
    "https://arrangement-newsletter-feelings-impossible.trycloudflare.com/"
)
OLLAMA_MODEL_ID = "gemma3:27b"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-TinyBERT-L-2"
CHROMA_HOST = "localhost"  # Sadece ana bilgisayar adı
CHROMA_PORT = 8000
COLLECTION_NAME = "rag_test_data"
reranker = CrossEncoder(RERANKER_MODEL_NAME)

# 1. Veritabanını Güncelle ve VectorStore'u Al


TOP_K_RETRIEVAL = 20  # ChromaDB'den çekilecek toplam parça sayısı
TOP_K_RERANK = 5  # LLM'e gönderilecek nihai parça sayısı


# ==========================================
# 1. STATE (HAFIZA) TANIMI
# ==========================================
class GraphState(TypedDict):
    """Akış boyunca taşınacak veriler"""

    question: str
    generation: str
    web_search: str  # "Yes" veya "No"
    documents: List[Document]


def get_hybrid_reranked_docs(ensemble_retriever, query):
    """ChromaDB'den çok sayıda parça çeker ve bunları Cross-Encoder ile yeniden sıralar."""

    # 3. İlk Geniş Havuz (Keyword + Vector sonuçlarının karışımı)

    first_pass_docs = ensemble_retriever.invoke(query)

    # 2. Yeniden Sıralama için Veri Hazırlama

    # Cross-Encoder, (sorgu, metin) çiftleri listesi ister.

    sentences = [(query, doc.page_content) for doc in first_pass_docs]

    # 3. Puanlama (Score)

    # Model, her parça için 0-1 arasında bir alaka puanı hesaplar.

    scores = reranker.predict(sentences)

    # 4. Parçaları Puanlarla Birleştirme ve Sıralama

    doc_scores = sorted(zip(first_pass_docs, scores), key=lambda x: x[1], reverse=True)

    # 5. En iyi (TOP_K_RERANK) parçayı seçme

    final_docs = [doc_score[0] for doc_score in doc_scores[:TOP_K_RERANK]]

    return final_docs


def retrieve_node(state, my_retriever):
    print("---RETRIEVE---")
    question = state["question"]

    # Parametre olarak gelen my_retriever'ı kullanıyoruz
    documents = get_hybrid_reranked_docs(my_retriever, query=question)

    return {"documents": documents, "question": question}


def grade_documents_node(state):
    """
    DOKÜMANLARI PUANLAR VE FİLTRELER
    """

    print("---GRADE DOCUMENTS---")
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    web_search = "No"

    for d in documents:
        score = grader_chain.invoke({"question": question, "document": d.page_content})
        grade = score.binary_score

        if grade == "yes":
            print("  - Belge onaylandı.")
            filtered_docs.append(d)
        else:
            print("  - Belge reddedildi (Alakasız).")
            continue

    # Eğer elimizde hiç belge kalmadıysa Web Araması bayrağını kaldır
    if not filtered_docs:
        print("  - !!! HİÇ BELGE KALMADI -> WEB ARAMASI GEREKİYOR !!!")
        web_search = "Yes"

    return {"documents": filtered_docs, "web_search": web_search}


def web_search_node(state):
    print("---WEB SEARCH (Tavily)---")
    question = state["question"]

    try:
        # 1. Tavily ile arama yap
        # 'search_depth="advanced"' daha derin arama yapar
        response = tavily.search(query=question, max_results=3)

        # 2. Sonuçları LangChain Document formatına çevir
        web_results = []
        for result in response["results"]:
            content = result["content"]
            url = result["url"]
            # Source kısmına URL'i koyuyoruz ki LLM kaynak göstersin
            doc = Document(page_content=content, metadata={"source": url})
            web_results.append(doc)

        print(f"  -> {len(web_results)} adet web sonucu bulundu.")

    except Exception as e:
        print(f"  -> Web arama hatası: {e}")
        web_results = []

    # Mevcut dokümanlara ekle (veya sadece web sonuçlarını kullan)
    # Genelde mevcutlar yetersiz olduğu için buradayız, o yüzden eskileri korumak veya atmak tercihtir.
    # CRAG genelde eskileri atıp sadece yenilere güvenmeyi tercih eder ama biz birleştirelim:
    documents = state["documents"] + web_results

    return {"documents": documents}


def generate_node(state):
    """
    1. State'ten dokümanları alır.
    2. [Source: ...] formatına çevirir (String birleştirme).
    3. LLM'e gönderir.
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    llm = ChatOllama(
        model=OLLAMA_MODEL_ID,
        base_url=CLOUDFLARE_TUNNEL_URL,
        temperature=0,
    )

    # --- ADIM 1: CONTEXT FORMATLAMA (Senin istediğin kısım) ---
    context_list = []
    for doc in documents:
        # Metadata'dan kaynak ismini al
        full_path = doc.metadata.get("source", "Unknown source")
        filename = os.path.basename(full_path)

        # İçeriği formatla
        formatted_chunk = f"[Source: {filename}]\n{doc.page_content}"
        context_list.append(formatted_chunk)

    # Hepsini tek bir metin haline getir
    formatted_context = "\n\n---\n\n".join(context_list)

    # --- ADIM 2: PROMPT VE ZİNCİR ---

    template = """
    You are an expert Question-Answering system. 
    Use the following CONTEXT to answer the QUESTION.
    Always cite the source using the format [Source: filename].
    
    CONTEXT:
    {context}
    
    QUESTION: {question}
    
    Answer:
    """

    prompt = ChatPromptTemplate.from_template(template)
    rag_chain = prompt | llm | StrOutputParser()

    # --- ADIM 3: ÇALIŞTIR ---
    # Not: Buraya 'documents' listesini değil, hazırladığımız 'formatted_context' stringini veriyoruz!
    generation = rag_chain.invoke({"context": formatted_context, "question": question})

    return {"generation": generation}


def decide_to_generate(state):
    """
    Grader'ın kararına göre yolu çizer.
    """
    print("---DECIDE---")
    if state["web_search"] == "Yes":
        return "web_search_node"  # Rotayı web aramasına çevir
    else:
        return "generate"  # Rotayı cevap üretmeye çevir


# ==========================================
# GRAFİĞİ KURMA VE ÇALIŞTIRMA (APP)
# ==========================================

# Grafiği bir fonksiyon içinde değil, global olarak bir kere tanımla

# ==========================================
# 2. AYARLAR VE MODELLER (LLM & GRADER)
# ==========================================
llm = ChatOllama(
    model=OLLAMA_MODEL_ID,
    base_url=CLOUDFLARE_TUNNEL_URL,
    temperature=0,
)


# 1. Pydantic Yapısı (English)
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant if they contain keyword(s) or semantic meaning useful to answer the question. 'yes' if relevant, 'no' if not."
    )


# 2. Modeli Yapılandırılmış Çıktıya Zorlama
# Not: Gemma kullanırken 'json' modu bazen daha stabil çalışır,
# ama LangChain'in bu metodu arka planda bunu halletmeye çalışır.
structured_llm_grader = llm.with_structured_output(GradeDocuments)

# 3. System Prompt (English)
# Gemma gibi modeller için görevi biraz daha detaylandırdım (Chain of Thought etkisi yaratması için).
system_prompt = """You are a grader assessing relevance of a retrieved document to a user question. 

If the document contains keyword(s) or semantic meaning useful to answer the question, grade it as 'yes'. 
It does not need to be a stringent test. The goal is to filter out erroneous retrievals. 

Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "User Question: {question}\n\nRetrieved Document: {document}"),
    ]
)

grader_chain = grade_prompt | structured_llm_grader


ensemble_retriever = update_db_with_feedback(
    "./database/",
    client=chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT),
    collection_name="rag_test_data",
)
workflow = StateGraph(GraphState)
retrieve_node_with_param = partial(retrieve_node, my_retriever=ensemble_retriever)
workflow.add_node("retrieve", retrieve_node_with_param)
workflow.add_node("grade_documents", grade_documents_node)
workflow.add_node("web_search_node", web_search_node)
workflow.add_node("generate", generate_node)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {"web_search_node": "web_search_node", "generate": "generate"},
)
workflow.add_edge("web_search_node", "generate")
workflow.add_edge("generate", END)

# Uygulamayı derle
app = workflow.compile()


def test_reranked_rag_query(user_query):
    """Yeniden sıralama mantığını uygulayarak sorguyu çalıştırır ve sonuçları yazdırır."""
    start_time = time.time()
    print(f"\nSorgu: {user_query}")
    print("-" * 50)
    print(f"\nSoru Soruluyor: {user_query}")
    inputs = {"question": user_query}

    # Stream veya Invoke ile çalıştır
    result = app.invoke(inputs)
    stop_time = time.time()
    elapsed_time = stop_time - start_time
    print("\n--- NİHAİ CEVAP ---")
    print(result["generation"])
    print(f"⏱️  Cevap Süresi: {elapsed_time:.2f} saniye")


if __name__ == "__main__":

    test_reranked_rag_query(
        user_query="Although the Denver Broncos won Super Bowl 50, who is their current head coach as of the 2024 NFL season?",
    )

    test_reranked_rag_query(
        user_query="Kathmandu established its first international relationship with Eugene, Oregon in 1975; however, who is the current mayor of Eugene, Oregon today?",
    )

    test_reranked_rag_query(
        user_query="The kilopond is described as a non-SI unit of force, but exactly how was its definition impacted by the 2019 redefinition of the SI base units?",
    )

    test_reranked_rag_query(
        user_query="Normanların eski İskandinav dinini ve dilini bırakıp, yerel halkın dinini ve dilini benimsemesindeki temel kültürel adaptasyon süreci nasıldı?",
    )
    test_reranked_rag_query(
        user_query="Sürtünme gibi muhafazakar olmayan kuvvetler, neden aslında mikroskobik potansiyellerin sonuçları olarak kabul edilir?",
    )

    test_reranked_rag_query(
        user_query="Why were the traditional Roman numerals (L) not used for Super Bowl 50?",
    )
