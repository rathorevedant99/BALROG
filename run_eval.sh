# vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --port 8081 --gpu-memory-utilization 0.3


python eval.py \
  agent.type=robust_cot_rag \
  agent.remember_cot=True \
  agent.max_cot_history=4 \
  eval.num_workers=48 \
  client.client_name=openai \
  client.model_id=gpt-4o-mini \
  rag.enabled=True \
  rag.device=cuda
  
  # python eval.py \
  # agent.type=robust_cot_rag \
  # agent.remember_cot=True \
  # agent.max_cot_history=2 \
  # eval.num_workers=7 \
  # client.client_name=openrouter \
  # client.model_id=deepseek/deepseek-r1:free \
  # rag.enabled=True \
  # rag.device=cpu
