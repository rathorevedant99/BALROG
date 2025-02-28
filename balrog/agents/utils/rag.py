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
import json
logger = logging.getLogger(__name__)

def clean_wiki_markup(text):
    """Clean wiki markup from text."""
    if not text:
        return ""
    # Remove [[File:]] tags and other wiki markup
    text = re.sub(r'\[\[File:.*?\]\]', '', text)
    text = re.sub(r'\{\{.*?\}\}', '', text)
    text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', text)
    text = re.sub(r"'{2,}", '', text)
    text = re.sub(r'={2,}.*?={2,}', '', text)
    text = re.sub(r'<.*?>', '', text)
    # Remove URLs and multiple spaces/newlines
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def load_cached_chunks(cache_path):
    """Load cached chunks if available."""
    if os.path.exists(cache_path):
        logger.info(f"Loading cached chunks from {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    return None

def parse_xml(file_path):
    """Parse an XML file into meaningful chunks, specifically handling NetHack wiki format."""
    logger.info(f"Starting to parse XML file: {file_path}")
    cache_path = f"{file_path}.chunks.pkl"
    cached_chunks = load_cached_chunks(cache_path)
    if cached_chunks:
        return cached_chunks

    try:
        # Set environment variable to disable tokenizers parallelism before any tokenizer is loaded
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        # Use iterparse to reduce memory usage
        chunks = []
        namespace = {'mw': 'http://www.mediawiki.org/xml/export-0.10/'}
        
        # Count pages first for progress tracking
        page_count = 0
        for _, elem in ET.iterparse(file_path, events=('end',)):
            if elem.tag.endswith('page'):
                page_count += 1
            elem.clear()
        
        logger.info(f"Found {page_count} pages to process")
        
        # Process pages with iterparse to save memory
        context = ET.iterparse(file_path, events=('end',))
        current_page = 0
        title_text = None
        
        for event, elem in tqdm(context, desc="Processing pages", total=page_count*2):  # Rough estimate
            if elem.tag.endswith('title'):
                title_text = elem.text
            elif elem.tag.endswith('text') and title_text and not title_text.startswith(('Talk:', 'User:')):
                content = elem.text
                if content:
                    content = clean_wiki_markup(content)
                    paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
                    
                    for para in paragraphs:
                        if len(para) > 50:
                            chunk = f"Title: {title_text}\nContent: {para}"
                            chunks.append(chunk)
            
            # Clear element to save memory
            if elem.tag.endswith('page'):
                current_page += 1
                title_text = None
            elem.clear()
            
            # Clear root periodically to save memory
            if current_page % 100 == 0:
                context.root.clear()
        
        logger.info(f"Generated {len(chunks)} chunks")
        
        # Cache the chunks
        with open(cache_path, 'wb') as f:
            pickle.dump(chunks, f)
        
        return chunks
    
    except Exception as e:
        logger.error(f"Error parsing XML: {str(e)}")
        raise

def parse_json(file_path):
    """Parse a JSON file into meaningful chunks."""
    logger.info(f"Starting to parse JSON file: {file_path}")
    cache_path = f"{file_path}.chunks.pkl"
    cached_chunks = load_cached_chunks(cache_path)
    if cached_chunks:
        return cached_chunks
    
    try:
        # Process JSON in a streaming fashion for large files
        chunks = []
        
        # Check file size to determine approach
        file_size = os.path.getsize(file_path)
        if file_size > 100 * 1024 * 1024:  # If file is larger than 100MB
            logger.info(f"Large JSON file detected ({file_size/1024/1024:.2f}MB), using streaming approach")
            import ijson  # Import here to avoid dependency if not needed
            
            with open(file_path, 'rb') as f:
                # Determine if it's a list or object at root
                for prefix, event, _ in ijson.parse(f):
                    if prefix == '' and event in ('start_array', 'start_map'):
                        is_array = event == 'start_array'
                        break
                
                f.seek(0)  # Reset file pointer
                
                if is_array:
                    # Process as array of items
                    item_index = 0
                    for item in ijson.items(f, 'item'):
                        path = f"Item {item_index}"
                        chunks.extend(_process_json_item(item, path))
                        item_index += 1
                        
                        # Log progress periodically
                        if item_index % 100 == 0:
                            logger.info(f"Processed {item_index} JSON items")
                else:
                    # Process as single object
                    for prefix, event, value in ijson.parse(f):
                        if event in ('string', 'number', 'boolean') and len(str(value)) >= 50:
                            chunks.append(f"Path: {prefix}\n{value}")
        else:
            # For smaller files, load the entire JSON
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                # Handle list of objects
                for i, item in enumerate(tqdm(data, desc="Processing JSON items")):
                    chunks.extend(_process_json_item(item, f"Item {i}"))
            elif isinstance(data, dict):
                # Handle dictionary
                chunks.extend(_process_json_item(data, "Root"))
        
        logger.info(f"Generated {len(chunks)} chunks from JSON")
        
        # Cache the chunks in batches if very large
        if len(chunks) > 100000:
            logger.info(f"Large number of chunks ({len(chunks)}), caching in batches")
            batch_size = 50000
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i+batch_size]
                batch_cache_path = f"{cache_path}.{i//batch_size}"
                with open(batch_cache_path, 'wb') as f:
                    pickle.dump(batch, f)
            
            # Create index file
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'type': 'batched',
                    'count': len(chunks),
                    'batch_size': batch_size,
                    'batches': (len(chunks) + batch_size - 1) // batch_size
                }, f)
        else:
            # Cache normally
            with open(cache_path, 'wb') as f:
                pickle.dump(chunks, f)
        
        return chunks
    
    except Exception as e:
        logger.error(f"Error parsing JSON: {str(e)}")
        raise

def _process_json_item(item, path="", min_length=50):
    """Process a JSON item recursively to extract text chunks."""
    chunks = []
    
    if isinstance(item, dict):
        # Process each key-value pair
        for key, value in item.items():
            current_path = f"{path}.{key}" if path else key
            
            # If value is a string, add it as a chunk
            if isinstance(value, str) and len(value) >= min_length:
                chunks.append(f"Path: {current_path}\n{value}")
            
            # Recursively process nested objects and arrays
            elif isinstance(value, (dict, list)):
                chunks.extend(_process_json_item(value, current_path))
    
    elif isinstance(item, list):
        # Process each item in the list
        for i, value in enumerate(item):
            current_path = f"{path}[{i}]"
            
            if isinstance(value, str) and len(value) >= min_length:
                chunks.append(f"Path: {current_path}\n{value}")
            
            elif isinstance(value, (dict, list)):
                chunks.extend(_process_json_item(value, current_path))
    
    return chunks

class RAG:
    def __init__(self, config):
        """Initialize RAG with configuration."""
        self.config = config
        self.model = None
        self.index = None
        self.faiss_index = None
        self.passage_embeddings = None
        self.cache_dir = config.rag.cache_dir
        self.top_k = config.rag.top_k
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Determine device once at initialization
        self.device = 'cuda' if torch.cuda.is_available() and self.config.rag.device == 'cuda' else 'cpu'
        logger.info(f"RAG will use device: {self.device}")
    
    def _ensure_initialized(self):
        """Lazy initialization of model."""
        if self.model is None:
            logger.info(f"Initializing SentenceTransformer model on {self.device}")
            self.model = SentenceTransformer(self.config.rag.model_name)
            if self.device == 'cuda':
                self.model = self.model.to(self.device)

    def _load_from_cache(self, cache_base):
        """Attempt to load index and embeddings from cache."""
        if all(os.path.exists(f"{cache_base}.{ext}") for ext in ['npy', 'faiss', 'pkl']):
            try:
                logger.info("Loading from cache...")
                self.passage_embeddings = np.load(f"{cache_base}.npy")
                self.faiss_index = faiss.read_index(f"{cache_base}.faiss")
                with open(f"{cache_base}.pkl", 'rb') as f:
                    self.index = pickle.load(f)
                
                logger.info("Successfully loaded from cache")
                return True
            except Exception as e:
                logger.warning(f"Cache load failed: {e}")
        return False

    def _save_to_cache(self, cache_base):
        """Save index and embeddings to cache."""
        try:
            logger.info("Saving to cache...")
            faiss.write_index(self.faiss_index, f"{cache_base}.faiss")
            np.save(f"{cache_base}.npy", self.passage_embeddings)
            with open(f"{cache_base}.pkl", 'wb') as f:
                pickle.dump(self.index, f, protocol=4)
            logger.info("Cache saved successfully")
        except Exception as e:
            logger.error(f"Error saving cache: {e}")

    def build_index(self, passages):
        """Build search index for passages."""
        start_time = time.time()
        logger.info(f"Building index for {len(passages)} passages")
        self._ensure_initialized()
        
        # Create cache identifier
        model_hash = hashlib.md5(self.config.rag.model_name.encode('utf-8')).hexdigest()[:8]
        content_sample = ''.join(sorted([p[:50] for p in passages[:100]]))
        content_hash = hashlib.md5(content_sample.encode('utf-8')).hexdigest()[:8]
        cache_base = os.path.join(self.cache_dir, f'{model_hash}_{content_hash}')
        logger.info(f"Using cache path: {cache_base}")
        
        if self._load_from_cache(cache_base):
            return
        
        # Filter and prepare passages
        logger.info("Filtering passages...")
        filtered_passages = [p.strip() for p in passages if len(p.strip()) > 50]
        self.index = {i: p for i, p in enumerate(filtered_passages)}
        
        # Generate embeddings with memory-efficient batching
        logger.info("Generating embeddings...")
        # Adjust batch size based on available memory and dataset size
        if self.device == 'cuda':
            # Smaller batches for GPU to prevent OOM
            if len(filtered_passages) > 10000:
                batch_size = 32  # Reduced from 64
            else:
                batch_size = 64  # Reduced from 128
        else:
            # CPU can handle smaller batches but more consistently
            batch_size = 16  # Reduced from 32
            
        logger.info(f"Using batch size: {batch_size}")
        
        # Instead of storing all embeddings in memory, save them to disk in chunks
        temp_embeddings_dir = os.path.join(self.cache_dir, f"temp_embeddings_{content_hash}")
        os.makedirs(temp_embeddings_dir, exist_ok=True)
        
        total_embeddings = 0
        chunk_size = 10000  # Number of embeddings per file
        current_chunk = []
        chunk_files = []
        
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
                
                current_chunk.append(embeddings)
                total_embeddings += len(embeddings)
                
                # If we've accumulated enough embeddings, save to disk
                if sum(len(e) for e in current_chunk) >= chunk_size:
                    chunk_data = np.vstack(current_chunk)
                    chunk_file = os.path.join(temp_embeddings_dir, f"chunk_{len(chunk_files)}.npy")
                    np.save(chunk_file, chunk_data)
                    chunk_files.append(chunk_file)
                    current_chunk = []
                
                # Clear cache after each batch if using CUDA
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
        
        # Save any remaining embeddings
        if current_chunk:
            chunk_data = np.vstack(current_chunk)
            chunk_file = os.path.join(temp_embeddings_dir, f"chunk_{len(chunk_files)}.npy")
            np.save(chunk_file, chunk_data)
            chunk_files.append(chunk_file)
        
        logger.info(f"Generated {total_embeddings} embeddings in {len(chunk_files)} chunks")
        
        # Create and train FAISS index
        logger.info("Creating FAISS index...")
        dim = np.load(chunk_files[0]).shape[1]  # Get dimension from first chunk
        
        # Use a simpler index for smaller datasets
        if len(filtered_passages) < 10000:
            self.faiss_index = faiss.IndexFlatL2(dim)
            logger.info("Using FlatL2 index for small dataset")
        else:
            # For larger datasets, use IVF index with appropriate number of clusters
            nlist = min(4096, max(1, int(np.sqrt(len(filtered_passages)))))
            quantizer = faiss.IndexFlatL2(dim)
            self.faiss_index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
            logger.info(f"Training IVF index with {nlist} clusters...")
            
            # For very large datasets, use a subset for training
            if len(filtered_passages) > 100000:
                train_size = min(100000, len(filtered_passages))
                logger.info(f"Using {train_size} samples for training the index")
                
                # Load a subset of vectors for training
                train_vectors = []
                vectors_needed = train_size
                for chunk_file in chunk_files:
                    chunk_data = np.load(chunk_file)
                    if len(chunk_data) <= vectors_needed:
                        train_vectors.append(chunk_data)
                        vectors_needed -= len(chunk_data)
                    else:
                        # Take only what we need from this chunk
                        train_vectors.append(chunk_data[:vectors_needed])
                        vectors_needed = 0
                    
                    if vectors_needed <= 0:
                        break
                
                train_vectors = np.vstack(train_vectors)
                self.faiss_index.train(train_vectors)
                del train_vectors  # Free memory
            else:
                # Load all vectors for training
                train_vectors = []
                for chunk_file in chunk_files:
                    train_vectors.append(np.load(chunk_file))
                train_vectors = np.vstack(train_vectors)
                self.faiss_index.train(train_vectors)
                del train_vectors  # Free memory
        
        # Add vectors to index in chunks
        logger.info("Adding vectors to index...")
        for i, chunk_file in enumerate(tqdm(chunk_files, desc="Adding to index")):
            chunk_data = np.load(chunk_file)
            self.faiss_index.add(chunk_data)
            # Explicitly delete to free memory
            del chunk_data
            if self.device == 'cuda':
                torch.cuda.empty_cache()
        
        # Now load all embeddings for saving to cache
        logger.info("Loading all embeddings for cache...")
        embeddings_list = []
        for chunk_file in tqdm(chunk_files, desc="Loading for cache"):
            embeddings_list.append(np.load(chunk_file))
        self.passage_embeddings = np.vstack(embeddings_list)
        
        # Save to cache
        self._save_to_cache(cache_base)
        
        # Clean up temporary files
        logger.info("Cleaning up temporary files...")
        for chunk_file in chunk_files:
            try:
                os.remove(chunk_file)
            except Exception as e:
                logger.warning(f"Failed to remove temporary file {chunk_file}: {e}")
        try:
            os.rmdir(temp_embeddings_dir)
        except Exception as e:
            logger.warning(f"Failed to remove temporary directory {temp_embeddings_dir}: {e}")
        
        total_time = time.time() - start_time
        logger.info(f"Index built in {total_time:.2f}s")

    def search(self, query):
        """Search for relevant passages."""
        # self._ensure_initialized()
        
        if not query.strip():
            return []
            
        try:
            logger.info(f"Searching for: {query[:100]}...")
            query_embedding = self.model.encode(
                query,
                convert_to_numpy=True,
                show_progress_bar=False,
                device=self.device
            ).reshape(1, -1)
            faiss.normalize_L2(query_embedding)
            
            D, I = self.faiss_index.search(query_embedding, self.top_k)
            results = [(self.index[idx], float(score)) for idx, score in zip(I[0], D[0]) if idx in self.index]
            
            logger.info(f"Found {len(results)} results")
            if results:
                logger.info("Top scores: " + ", ".join(f"{score:.4f}" for _, score in results[:3]))
            
            return results
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
        
        # """Search FAISS index for similar documents and return titles + content."""
        # if self.index is None or self.doc_store is None:
        #     print("Index not loaded. Load or build it first.")
        #     return []

        # query_embedding = self.model.encode([query], convert_to_numpy=True)
        # faiss.normalize_L2(query_embedding)  # Normalize query

        # distances, indices = self.index.search(query_embedding, self.top_k)

        # return [(self.doc_store[idx]["title"], self.doc_store[idx]["content"]) for idx in indices[0]]