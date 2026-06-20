import os
import time
import glob
import logging
from typing import Dict, Any, List

logger = logging.getLogger("app.services.rag")

CLINICAL_ROUTING_MAP = {
    "pregnancy": ["pregnancy"],
    "pregnant": ["pregnancy"],
    "expecting": ["pregnancy"],
    
    "child": ["pediatrics"],
    "infant": ["pediatrics"],
    "baby": ["pediatrics"],
    "pediatric": ["pediatrics"],
    "kid": ["pediatrics"],
    
    "chest pain": ["emergency", "cardiology", "mi", "pe"],
    "heart": ["emergency", "cardiology", "mi"],
    "mi": ["emergency", "cardiology", "mi"],
    
    "headache": ["chronic", "neurology", "stroke"],
    "stroke": ["emergency", "neurology", "stroke"],
    
    "cough": ["chronic", "emergency", "respiratory", "pneumonia"],
    "pneumonia": ["emergency", "respiratory", "pneumonia"],
    "asthma": ["chronic", "emergency", "respiratory"],
    
    "dysuria": ["miscellaneous", "infectious", "uti"],
    "urination": ["miscellaneous", "infectious", "uti"],
    "peeing": ["miscellaneous", "infectious", "uti"],
    "fever": ["emergency", "miscellaneous", "infectious", "sepsis"],
    "sepsis": ["emergency", "infectious", "sepsis"],
    
    "abdominal": ["emergency", "chronic", "gastro", "appendicitis"],
    "vomiting": ["emergency", "chronic", "gastro"],
    
    "flank": ["emergency", "renal", "stones"],
    "kidney": ["emergency", "renal", "stones"]
}

def calculate_keyword_score(text: str, doc_name: str, query: str) -> float:
    """
    Computes a keyword overlap score between query and document text/metadata.
    Returns a float between 0.0 and 1.0.
    """
    query_words = [w.lower() for w in query.split() if len(w) > 3]
    if not query_words:
        query_words = [query.lower()]
        
    text_lower = text.lower()
    doc_name_lower = doc_name.lower().replace("-", " ")
    
    matches = 0
    for word in query_words:
        if word in text_lower or word in doc_name_lower:
            matches += 1
            
    return float(matches) / len(query_words) if query_words else 0.0

class BaseRetriever:
    """
    Abstract Base Class for symptom-based retrieval services.
    Enables future hot-swaps to alternate vector databases (Chroma, Pinecone, Qdrant, etc.)
    without modifying LangGraph workflow nodes.
    """
    def retrieve(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        raise NotImplementedError

class FAISSRetriever(BaseRetriever):
    """
    Concrete retriever implementation utilizing a local FAISS index
    and HuggingFaceEmbeddings. Supports offline mock mode for testing/demoing.
    """
    def __init__(self):
        self.index = None
        self.embeddings = None
        self.store_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "resources", "vector_store"))
        self.docs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "resources", "medical_docs"))
        os.makedirs(self.store_dir, exist_ok=True)
        os.makedirs(self.docs_dir, exist_ok=True)
        
    def _is_index_persisted(self) -> bool:
        return (
            os.path.exists(os.path.join(self.store_dir, "index.faiss")) and
            os.path.exists(os.path.join(self.store_dir, "index.pkl"))
        )

    def initialize(self):
        """
        Initializes the retriever singleton.
        Loads the FAISS index if already persisted, otherwise scans PDFs recursively and builds it.
        Bypasses torch/transformers imports if the application is running in mock mode.
        """
        # Retrieve test mode or mock settings
        if os.getenv("GROQ_API_KEY") == "mock":
            logger.info("🛠️ FAISSRetriever initializing in MOCK mode. Pre-loading PDF chunks locally.")
            self._initialize_mock_chunks()
            return

        t0 = time.time()
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            from langchain_community.vectorstores import FAISS
            
            logger.info("🔌 Live Mode: Initializing HuggingFaceEmbeddings (all-MiniLM-L6-v2)...")
            self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            
            if self._is_index_persisted():
                logger.info(f"📂 Persisted FAISS index found. Loading from directory: {self.store_dir}")
                self.index = FAISS.load_local(self.store_dir, self.embeddings, allow_dangerous_deserialization=True)
            else:
                logger.info(f"📂 FAISS index not found. Building index from clinical documents recursively in {self.docs_dir}...")
                self._build_index()
            
            latency = int((time.time() - t0) * 1000)
            logger.info(f"✅ FAISSRetriever loaded successfully in {latency}ms.")
        except Exception as e:
            logger.error(f"❌ Failed to load live FAISSRetriever: {e}", exc_info=True)
            logger.warning("Falling back to offline MOCK retriever mode.")
            self._initialize_mock_chunks()

    def _build_index(self):
        """Loads all PDF files in the medical_docs directory recursively, chunks them, and builds a FAISS index."""
        from langchain_community.vectorstores import FAISS
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_community.document_loaders import PyPDFLoader
        
        pdf_files = glob.glob(os.path.join(self.docs_dir, "**", "*.pdf"), recursive=True)
        if not pdf_files:
            logger.warning(f"⚠️ No PDF documents found in clinical reference folder: {self.docs_dir}")
            # Create a fallback placeholder chunk in the index
            from langchain_core.documents import Document
            dummy_docs = [Document(
                page_content="General Clinical Symptoms: Monitor patient vitals, ensure hydration, and seek physician evaluation for severe cases.", 
                metadata={
                    "source": "none.pdf", 
                    "page": 1,
                    "document_type": "miscellaneous"
                }
            )]
            self.index = FAISS.from_documents(dummy_docs, self.embeddings)
            self.index.save_local(self.store_dir)
            return

        logger.info(f"Found {len(pdf_files)} PDF clinical guides. Loading documents recursively...")
        
        guideline_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
        case_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
        
        chunks = []
        for pdf_file in pdf_files:
            try:
                loader = PyPDFLoader(pdf_file)
                pages = loader.load()
                doc_name = os.path.basename(pdf_file)
                folder_name = os.path.basename(os.path.dirname(pdf_file))
                
                # Determine doc type and corresponding splitter
                if doc_name == "api_cases.pdf" or folder_name in ["cases", "case"]:
                    doc_type = "case"
                    splitter = case_splitter
                else:
                    doc_type = folder_name
                    splitter = guideline_splitter
                
                # Assign custom metadata to pages before splitting
                for idx, page in enumerate(pages):
                    page.metadata = {
                        "source": doc_name,
                        "page": idx + 1,
                        "document_type": doc_type
                    }
                
                doc_chunks = splitter.split_documents(pages)
                chunks.extend(doc_chunks)
            except Exception as e:
                logger.error(f"Error parsing PDF document {pdf_file}: {e}")
                
        if not chunks:
            raise ValueError("Failed to load clinical text contents from PDF documents.")

        logger.info(f"Vectorizing {len(chunks)} chunks and saving FAISS index locally...")
        self.index = FAISS.from_documents(chunks, self.embeddings)
        self.index.save_local(self.store_dir)
        logger.info(f"Persisted FAISS index successfully to: {self.store_dir}")

    def _initialize_mock_chunks(self):
        """Loads and splits PDF files locally without loading heavy PyTorch/Transformers dependencies."""
        self.mock_chunks = []
        pdf_files = glob.glob(os.path.join(self.docs_dir, "**", "*.pdf"), recursive=True)
        if not pdf_files:
            self.mock_chunks = [{
                "text": "Fallback clinical guidance for triage. Monitor vitals and consult a physician.",
                "doc": "fallback_guidelines.pdf",
                "page": 1,
                "document_type": "miscellaneous"
            }]
            return
            
        try:
            import pypdf
            for pdf_file in pdf_files:
                doc_name = os.path.basename(pdf_file)
                folder_name = os.path.basename(os.path.dirname(pdf_file))
                
                if doc_name == "api_cases.pdf" or folder_name in ["cases", "case"]:
                    doc_type = "case"
                    chunk_size = 400
                    chunk_overlap = 50
                else:
                    doc_type = folder_name
                    chunk_size = 1200
                    chunk_overlap = 200

                reader = pypdf.PdfReader(pdf_file)
                for page_num, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if not text:
                        continue
                    
                    i = 0
                    while i < len(text):
                        chunk_text = text[i:i+chunk_size]
                        self.mock_chunks.append({
                            "text": chunk_text,
                            "doc": doc_name,
                            "page": page_num + 1,
                            "document_type": doc_type
                        })
                        i += chunk_size - chunk_overlap
            logger.info(f"Mock index initialized. Splitted {len(self.mock_chunks)} chunks from {len(pdf_files)} PDF(s).")
        except Exception as e:
            logger.error(f"Failed to load mock chunks: {e}")
            self.mock_chunks = [{
                "text": "Fallback clinical guidance for triage. Monitor vitals and consult a physician.",
                "doc": "fallback_guidelines.pdf",
                "page": 1,
                "document_type": "miscellaneous"
            }]

    def retrieve(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Queries the retriever for relevant clinical guidelines.
        Returns context content, sources list (with score/page/document_type), and latency metrics.
        """
        t0 = time.time()
        if not query:
            return {
                "context": "",
                "sources": [],
                "retrieval_latency_ms": 0
            }

        # Mock Mode Search
        if os.getenv("GROQ_API_KEY") == "mock" or self.index is None:
            results = self._retrieve_mock(query, top_k)
            latency = int((time.time() - t0) * 1000)
            results["retrieval_latency_ms"] = max(latency, 1)
            
            # Diagnostics logging block
            diagnostics_lines = [
                "[RAG DIAGNOSTICS]",
                f"Query: {query}",
                f"Retrieved Chunks: {len(results['sources'])}",
                f"Latency: {results['retrieval_latency_ms']} ms",
                "Top Sources:"
            ]
            for src in results["sources"]:
                diagnostics_lines.append(f"- {src['source']} ({src['score']})")
            logger.info("\n".join(diagnostics_lines))
            return results

        # Live Mode Search
        try:
            INITIAL_RETRIEVAL_K = 8
            FINAL_TOP_K = top_k
            MIN_RELEVANCE_SCORE = 0.25
            RELAXED_MIN_RELEVANCE = 0.20
            
            docs_and_scores = self.index.similarity_search_with_relevance_scores(query, k=INITIAL_RETRIEVAL_K)
            
            candidates = []
            for doc, score in docs_and_scores:
                raw_score = float(score)
                page_num = doc.metadata.get("page", 1)
                doc_type = doc.metadata.get("document_type")
                if doc_type is None:
                    doc_type = "miscellaneous"
                    if isinstance(page_num, int):
                        page_num = page_num + 1
                
                pdf_name = os.path.basename(doc.metadata.get("source", "unknown_doc.pdf"))
                if doc_type in ["cases", "case"] or pdf_name == "api_cases.pdf":
                    doc_type = "case"
 
                # Calculate explicit hybrid score (Fix 4)
                kw_score = calculate_keyword_score(doc.page_content, pdf_name, query)
                hybrid_score = 0.7 * raw_score + 0.3 * kw_score

                adjusted_score = hybrid_score
                if doc_type in ["emergency", "chronic", "pregnancy", "pediatrics", "miscellaneous"] or doc_type != "case":
                    adjusted_score *= 1.15
                elif doc_type == "case":
                    adjusted_score *= 0.90
                
                # Clinical Corpus-Aware Boost (Task 6)
                corpus_boost = 1.0
                query_lower = query.lower()
                for keyword, target_types in CLINICAL_ROUTING_MAP.items():
                    if keyword in query_lower:
                        if doc_type in target_types or any(t in pdf_name.lower() for t in target_types):
                            corpus_boost = 1.20
                            break
                adjusted_score *= corpus_boost
                
                # Keyword reranking bonus (Task 3)
                symptom_phrases = ["back pain", "stroke", "uti", "asthma", "pregnancy", "fever", "headache"]
                query_lower = query.lower()
                bonus = 0.0
                for phrase in symptom_phrases:
                    if phrase in query_lower:
                        chunk_text_lower = doc.page_content.lower().replace("-", " ")
                        filename_lower = pdf_name.lower().replace("-", " ")
                        if phrase in filename_lower:
                            bonus = 0.15
                            break
                        elif phrase in chunk_text_lower:
                            bonus = 0.10
                            break
                adjusted_score += bonus
                adjusted_score = min(adjusted_score, 1.0)
                
                adjusted_score_rounded = round(adjusted_score, 2)
                
                candidates.append({
                    "context": doc.page_content,
                    "pdf_name": pdf_name,
                    "page_num": page_num,
                    "document_type": doc_type,
                    "score": adjusted_score_rounded
                })
            
            # Apply threshold filtering (first 0.25, fallback to 0.15 if 0 remain)
            filtered_candidates = [c for c in candidates if c["score"] >= MIN_RELEVANCE_SCORE]
            if not filtered_candidates:
                filtered_candidates = [c for c in candidates if c["score"] >= RELAXED_MIN_RELEVANCE]
            
            # Sort by adjusted score descending first to prioritize higher scoring duplicates
            filtered_candidates.sort(key=lambda x: x["score"], reverse=True)
            
            # Deduplicate by (source, page) (Task 1)
            seen = set()
            deduped_candidates = []
            for candidate in filtered_candidates:
                key = (candidate["pdf_name"], candidate["page_num"])
                if key not in seen:
                    seen.add(key)
                    deduped_candidates.append(candidate)
            
            # Cap to FINAL_TOP_K
            final_candidates = deduped_candidates[:FINAL_TOP_K]
            
            context_parts = [c["context"] for c in final_candidates]
            sources = [
                {
                    "source": c["pdf_name"],
                    "page": c["page_num"],
                    "document_type": c["document_type"],
                    "score": c["score"]
                }
                for c in final_candidates
            ]
            
            latency = int((time.time() - t0) * 1000)
            
            # Diagnostics logging block
            diagnostics_lines = [
                "[RAG DIAGNOSTICS]",
                f"Query: {query}",
                f"Retrieved Chunks: {len(sources)}",
                f"Latency: {latency} ms",
                "Top Sources:"
            ]
            for src in sources:
                diagnostics_lines.append(f"- {src['source']} ({src['score']})")
            logger.info("\n".join(diagnostics_lines))
            
            return {
                "context": "\n\n---\n\n".join(context_parts),
                "sources": sources,
                "retrieval_latency_ms": latency
            }
        except Exception as e:
            logger.error(f"❌ RAG live retrieval failed: {e}", exc_info=True)
            # Fallback to mock search
            results = self._retrieve_mock(query, top_k)
            latency = int((time.time() - t0) * 1000)
            results["retrieval_latency_ms"] = latency
            
            # Diagnostics logging block for fallback
            diagnostics_lines = [
                "[RAG DIAGNOSTICS]",
                f"Query: {query}",
                f"Retrieved Chunks: {len(results['sources'])}",
                f"Latency: {latency} ms",
                "Top Sources:"
            ]
            for src in results["sources"]:
                diagnostics_lines.append(f"- {src['source']} ({src['score']})")
            logger.info("\n".join(diagnostics_lines))
            
            return results

    def _retrieve_mock(self, query: str, top_k: int) -> Dict[str, Any]:
        """Performs a hybrid keyword + simulated vector overlap query over the local chunk cache."""
        scored_chunks = []
        for chunk in getattr(self, "mock_chunks", []):
            kw_score = calculate_keyword_score(chunk["text"], chunk["doc"], query)
            if kw_score > 0.0:
                # Simulate raw vector relevance score
                raw_score = 0.70
                hybrid_score = 0.7 * raw_score + 0.3 * kw_score
                scored_chunks.append((chunk, hybrid_score))
                
        # Sort by match score descending
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        INITIAL_RETRIEVAL_K = 8
        FINAL_TOP_K = top_k
        MIN_RELEVANCE_SCORE = 0.25
        RELAXED_MIN_RELEVANCE = 0.20

        top_chunks = scored_chunks[:INITIAL_RETRIEVAL_K]
        
        # If no keywords match, yield the first available chunks with baseline dummy scores
        if not top_chunks and getattr(self, "mock_chunks", None):
            top_chunks = [(c, 0.50) for c in self.mock_chunks[:INITIAL_RETRIEVAL_K]]
            
        candidates = []
        for chunk, score in top_chunks:
            doc_type = chunk.get("document_type", "miscellaneous")
            doc_name = chunk["doc"]
            if doc_name == "api_cases.pdf" or doc_type in ["cases", "case"]:
                doc_type = "case"
                
            adjusted_score = score
            if doc_type in ["emergency", "chronic", "pregnancy", "pediatrics", "miscellaneous"] or doc_type != "case":
                adjusted_score *= 1.15
            elif doc_type == "case":
                adjusted_score *= 0.90
            
            # Clinical Corpus-Aware Boost (Task 6)
            corpus_boost = 1.0
            query_lower = query.lower()
            for keyword, target_types in CLINICAL_ROUTING_MAP.items():
                if keyword in query_lower:
                    if doc_type in target_types or any(t in doc_name.lower() for t in target_types):
                        corpus_boost = 1.20
                        break
            adjusted_score *= corpus_boost
            
            # Keyword reranking bonus (Task 3)
            symptom_phrases = ["back pain", "stroke", "uti", "asthma", "pregnancy", "fever", "headache"]
            query_lower = query.lower()
            bonus = 0.0
            for phrase in symptom_phrases:
                if phrase in query_lower:
                    chunk_text_lower = chunk["text"].lower().replace("-", " ")
                    filename_lower = doc_name.lower().replace("-", " ")
                    if phrase in filename_lower:
                        bonus = 0.15
                        break
                    elif phrase in chunk_text_lower:
                        bonus = 0.10
                        break
            adjusted_score += bonus
            adjusted_score = min(adjusted_score, 1.0)
            
            adjusted_score_rounded = round(adjusted_score, 2)
            
            candidates.append({
                "text": chunk["text"],
                "doc": doc_name,
                "page": chunk["page"],
                "document_type": doc_type,
                "score": adjusted_score_rounded
            })
            
        # Apply threshold filtering (first 0.25, fallback to 0.15 if 0 remain)
        filtered_candidates = [c for c in candidates if c["score"] >= MIN_RELEVANCE_SCORE]
        if not filtered_candidates:
            filtered_candidates = [c for c in candidates if c["score"] >= RELAXED_MIN_RELEVANCE]
            
        # Sort by adjusted score descending first to prioritize higher scoring duplicates
        filtered_candidates.sort(key=lambda x: x["score"], reverse=True)
        
        # Deduplicate by (source, page) (Task 1)
        seen = set()
        deduped_candidates = []
        for candidate in filtered_candidates:
            key = (candidate["doc"], candidate["page"])
            if key not in seen:
                seen.add(key)
                deduped_candidates.append(candidate)
        
        # Cap to FINAL_TOP_K
        final_candidates = deduped_candidates[:FINAL_TOP_K]
        
        context_parts = [c["text"] for c in final_candidates]
        sources = [
            {
                "source": c["doc"],
                "page": c["page"],
                "document_type": c["document_type"],
                "score": c["score"]
            }
            for c in final_candidates
        ]
            
        return {
            "context": "\n\n---\n\n".join(context_parts),
            "sources": sources
        }

_retriever_instance = None

def get_retriever() -> FAISSRetriever:
    """Singleton getter for the FAISSRetriever application state."""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = FAISSRetriever()
        _retriever_instance.initialize()
    return _retriever_instance
