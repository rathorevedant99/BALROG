python eval.py \
  agent.type=naive_rag \
  eval.num_workers=64 \
  client.client_name=openai \
  client.model_id=gpt-4o-mini-2024-07-18 \
  eval.document_path=local/data/nethackwiki_current.xml