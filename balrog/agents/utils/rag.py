import xml.etree.ElementTree as ET
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re
import torch
import logging
import os
import pickle
from tqdm import tqdm
import time
import hashlib

logger = logging.getLogger(__name__)

def clean_wiki_markup(text):
    """Clean wiki markup from text."""
    if not text:
        return ""
        
    # Remove [[File:]] tags
    text = re.sub(r'\[\[File:.*?\]\]', '', text)
    
    # Remove other wiki markup
    text = re.sub(r'\{\{.*?\}\}', '', text)
    text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', text)
    text = re.sub(r"'{2,}", '', text)
    text = re.sub(r'={2,}.*?={2,}', '', text)
    text = re.sub(r'<.*?>', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove multiple spaces and newlines
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def parse_xml(file_path):
    """Parse an XML file into meaningful chunks, specifically handling NetHack wiki format."""
    logger.info(f"Starting to parse XML file: {file_path}")
    
    # First check if we have cached chunks
    cache_path = f"{file_path}.chunks.pkl"
    if os.path.exists(cache_path):
        logger.info("Loading cached chunks...")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        chunks = []
        namespace = {'mw': 'http://www.mediawiki.org/xml/export-0.10/'}
        
        for page in tqdm(root.findall('.//mw:page', namespace), desc="Parsing pages"):
            title = page.find('mw:title', namespace)
            text = page.find('.//mw:text', namespace)
            
            if title is not None and text is not None:
                title_text = title.text
                content = text.text
                
                if content and title_text and not title_text.startswith('Talk:') and not title_text.startswith('User:'):
                    content = clean_wiki_markup(content)
                    paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
                    
                    for para in paragraphs:
                        if len(para) > 50:
                            chunk = f"Title: {title_text}\n{para}"
                            chunks.append(chunk)

        logger.info(f"Generated {len(chunks)} chunks")
        
        # Cache the chunks
        with open(cache_path, 'wb') as f:
            pickle.dump(chunks, f)
        
        return chunks
    
    except Exception as e:
        logger.error(f"Error parsing XML: {str(e)}")
        raise

class RAG:
    def __init__(self, config):
        """Initialize RAG with automatic device selection."""
        self.config = config
        self.device = None
        self.model = None
        self.res = None
        self.index = None
        self.faiss_index = None
        self.cache_dir = config.rag.cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _ensure_initialized(self):
        """Lazy initialization of CUDA resources."""
        if self.device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() and self.config.rag.device == 'cuda' else 'cpu')
            logger.info(f"Initializing RAG on {self.device}")
            self.model = SentenceTransformer(self.config.rag.model_name).to(self.device)
            
            if self.device.type == 'cuda':
                self.res = faiss.StandardGpuResources()

    def build_index(self, passages):
        """Build search index efficiently on GPU or CPU."""
        start_time = time.time()
        logger.info(f"Building index for {len(passages)} passages on {self.device}")
        self._ensure_initialized()
        # Setup cache paths
        model_dim = str(self.model.get_sentence_embedding_dimension())
        model_hash = hashlib.md5(model_dim.encode('utf-8')).hexdigest()[:8]

        content = ''.join(sorted([p[:100] for p in passages]))
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()[:8]
        
        device_str = self.device.type
        cache_base = os.path.join(self.cache_dir, f'{model_hash}_{content_hash}_{device_str}')

        logger.info(f"Using cache path: {cache_base}")
        
        # Try loading from cache
        if all(os.path.exists(f"{cache_base}.{ext}") for ext in ['npy', 'faiss', 'pkl']):
            try:
                logger.info("Loading from cache...")
                print("Loading from cache...")
                self.passage_embeddings = np.load(f"{cache_base}.npy")
                self.faiss_index = faiss.read_index(f"{cache_base}.faiss")
                with open(f"{cache_base}.pkl", 'rb') as f:
                    self.index = pickle.load(f)
                
                if self.device.type == 'cuda':
                    logger.info("Moving index to GPU...")
                    self.faiss_index = faiss.index_cpu_to_gpu(self.res, 0, self.faiss_index)
                
                logger.info("Successfully loaded from cache")
                return
            except Exception as e:
                logger.warning(f"Cache load failed: {e}")
        
        # Filter passages
        logger.info("Filtering passages...")
        filtered_passages = [p.strip() for p in passages if len(p.strip()) > 50]
        self.index = {i: p for i, p in enumerate(filtered_passages)}
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        batch_size = 512 if self.device.type == 'cuda' else 32
        embeddings_list = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(filtered_passages), batch_size), desc="Embedding"):
                batch = filtered_passages[i:i + batch_size]
                embeddings = self.model.encode(
                    batch,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    device=self.device,
                    batch_size=len(batch)
                )
                embeddings_list.append(embeddings)
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        self.passage_embeddings = np.vstack(embeddings_list)
        dim = self.passage_embeddings.shape[1]
        
        # Create FAISS index
        logger.info("Creating FAISS index...")
        
        # Always create index on CPU first
        quantizer = faiss.IndexFlatL2(dim)
        self.faiss_index = faiss.IndexIVFFlat(quantizer, dim, 100, faiss.METRIC_L2)
        
        logger.info("Training index...")
        self.faiss_index.train(self.passage_embeddings)
        
        logger.info("Adding vectors...")
        self.faiss_index.add(self.passage_embeddings)
        
        # Move to GPU if needed
        if self.device.type == 'cuda':
            logger.info("Moving index to GPU...")
            self.faiss_index = faiss.index_cpu_to_gpu(self.res, 0, self.faiss_index)
        
        # Save to cache
        logger.info("Saving to cache...")
        if self.device.type == 'cuda':
            cpu_index = faiss.index_gpu_to_cpu(self.faiss_index)
            faiss.write_index(cpu_index, f"{cache_base}.faiss")
        else:
            faiss.write_index(self.faiss_index, f"{cache_base}.faiss")
            
        np.save(f"{cache_base}.npy", self.passage_embeddings)
        with open(f"{cache_base}.pkl", 'wb') as f:
            pickle.dump(self.index, f)
        
        total_time = time.time() - start_time
        logger.info(f"Index built in {total_time:.2f}s")

    def search(self, query, top_k=5):
        """Search for relevant passages."""
        self._ensure_initialized()
        
        if not query.strip():
            return []
            
        try:
            logger.info(f"Generating embedding for query: {query[:100]}...")
            query_embedding = self.model.encode(
                query,
                convert_to_numpy=True,
                show_progress_bar=False,
                device=self.device
            ).reshape(1, -1)
            logger.info("Searching FAISS index...")
            D, I = self.faiss_index.search(query_embedding, top_k)
            results = [(self.index[idx], float(score)) for idx, score in zip(I[0], D[0])]
            logger.info(f"Found {len(results)} results")
            logger.info("Top scores: " + ", ".join(f"{score:.4f}" for _, score in results))
        
            return results
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []