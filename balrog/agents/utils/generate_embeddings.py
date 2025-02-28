from balrog.agents.utils.rag import RAG, clean_wiki_markup, load_cached_chunks, parse_xml, parse_json
from hydra import compose, core, initialize
import omegaconf
import os
import logging

logger = logging.getLogger(__name__)

def parse_documents(config):
    if config.rag.documents_path.endswith('.xml'):
        return parse_xml(config.rag.documents_path)
    elif config.rag.documents_path.endswith('.json'):
        return parse_json(config.rag.documents_path)
    else:
        raise ValueError(f"Unsupported document format: {config.rag.documents_path}")

def generate_embeddings(config):
    rag = RAG(config)
    documents = parse_documents(config)
    rag.build_index(documents)


if __name__ == "__main__":
    with open("balrog/config/config.yaml", "r") as f:
        config = omegaconf.OmegaConf.load(f)
    generate_embeddings(config)
    print("Embeddings generated and saved")