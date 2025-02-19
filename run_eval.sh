CUDA_VISIBLE_DEVICES=0 python eval.py \
  agent.type=naive_rag_vedant \
  eval.num_workers=64 \
  client.client_name=vllm \
  client.model_id=meta-llama/Llama-3.2-1B-Instruct \
  client.base_url=http://0.0.0.0:8081/v1 \
  rag.enabled=True \
  rag.model_name=sentence-transformers/all-mpnet-base-v2 \
  rag.documents_path=local/data/nethackwiki_current.xml \
  rag.device=cuda

# CUDA_VISIBLE_DEVICES=0 python eval.py \
#   agent.type=naive_rag_vedant \
#   eval.num_workers=64 \
#   client.client_name=vllm \
#   client.model_id=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
#   client.base_url=http://0.0.0.0:8081/v1 \
#   rag.enabled=True \
#   rag.model_name=sentence-transformers/all-mpnet-base-v2 \
#   rag.documents_path=local/data/nethackwiki_current.xml \
#   rag.device=cuda


# vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --port 8081 --gpu-memory-utilization 0.8

# CUDA_VISIBLE_DEVICES=0 python eval.py \
#   agent.type=naive_rag \
#   eval.num_workers=64 \
#   client.client_name=openai \
#   client.model_id=gpt-4o-mini-2024-07-18 \
#   rag.enabled=True \
#   rag.model_name=sentence-transformers/all-mpnet-base-v2 \
#   rag.documents_path=local/data/nethackwiki_current.xml \
#   rag.device=cuda