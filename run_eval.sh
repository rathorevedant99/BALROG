# python eval.py \
#   agent.type=naive_rag \
#   eval.num_workers=64 \
#   client.client_name=vllm \
#   client.model_id=meta-llama/Llama-3.2-1B-Instruct \
#   client.base_url=http://0.0.0.0:8081/v1 \
#   rag.enabled=True \
#   rag.model_name=sentence-transformers/all-mpnet-base-v2 \
#   rag.documents_path=local/data/nethackwiki_current.xml \
#   rag.device=cuda

# python eval.py \
#   agent.type=robust_cot_rag \
#   eval.num_workers=4 \
#   eval.num_episodes.nle=1 \
#   client.client_name=vllm \
#   client.model_id=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
#   client.base_url=http://0.0.0.0:8081/v1 \
#   rag.enabled=True \
#   rag.model_name=sentence-transformers/all-mpnet-base-v2 \
#   rag.documents_path=local/data/nethackwiki_current.xml \
#   rag.device=cuda \
#   agent.remember_cot=True

python eval.py \
  agent.type=robust_cot_rag \
  eval.num_workers=9 \
  eval.num_episodes.nle=5 \
  client.client_name=openai \
  client.model_id=gpt-4o \
  rag.enabled=True \
  rag.model_name=sentence-transformers/all-mpnet-base-v2 \
  rag.documents_path=local/data/nethackwiki_current.xml \
  rag.device=cuda \
  agent.remember_cot=True


# vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --port 8081 --gpu-memory-utilization 0.3

# python eval.py \
#   agent.type=naive_rag \
#   eval.num_workers=8 \
#   eval.num_episodes.nle=1 \
#   client.client_name=openai \
#   client.model_id=gpt-4o \
#   rag.enabled=True \
#   rag.model_name=sentence-transformers/all-mpnet-base-v2 \
#   rag.documents_path=local/data/nethackwiki_current.xml \
#   rag.device=cuda

# python eval.py \
#   agent.type=naive \
#   eval.num_workers=8 \
#   client.client_name=gemini \
#   client.model_id=gemini-2.0-flash \
#   rag.enabled=True \
#   rag.model_name=sentence-transformers/all-mpnet-base-v2 \
#   rag.documents_path=local/data/nethackwiki_current.xml \
#   rag.device=cuda