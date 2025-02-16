import xml.etree.ElementTree as ET
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def parse_txt(file_path):
    with open(file_path, 'r') as f:
        return f.readlines()

def parse_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    return [child.text for child in root]

class RAG:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.faiss_index = None

    def build_index(self, passages):
        self.index = {i: passage for i, passage in enumerate(passages)}
        self.faiss_index = faiss.IndexFlatL2(self.model.get_sentence_embedding_dimension())
        embeddings = self.model.encode(passages)
        self.faiss_index.add(np.array(embeddings))

    def search(self, query, top_k=5):
        query_embedding = self.model.encode(query)
        D, I = self.faiss_index.search(np.array([query_embedding]), top_k)
        return [(self.index[i], float(D[0][j])) for j, i in enumerate(I[0])]

    def search_xml(self, query, top_k=5):
        results = self.search(query, top_k)
        root = ET.Element('search_results')
        for passage, score in results:
            result = ET.SubElement(root, 'result')
            ET.SubElement(result, 'passage').text = passage
            ET.SubElement(result, 'score').text = str(score)
        return ET.tostring(root).decode('utf-8')

    def search_json(self, query, top_k=5):
        results = self.search(query, top_k)
        return [{'passage': passage, 'score': score} for passage, score in results]

    def search_csv(self, query, top_k=5):
        results = self.search(query, top_k)
        return '\n'.join([f'{passage},{score}' for passage, score in results])

    def search_tsv(self, query, top_k=5):
        results = self.search(query, top_k)
        return '\n'.join([f'{passage}\t{score}' for passage, score in results])
    
    def search_text(self, query, top_k=5):
        results = self.search(query, top_k)
        return '\n'.join([f'{passage}\nScore: {score}\n' for passage, score in results])