Custom knowledge based query assistant.
- install modules
- upload your data (e.g. pdf / doc) to the data folder (delete dummy data)
- change the API key in config.json
- c:/poppler/library/bin
- start main.py

All program requirements will be created during startup, based on you data folder.
This will take time

Root/
├── main.py
├── splash_widget.py
├── config/
│   └── config.json
├── gui/
│   ├── __init__.py
│   ├── main_window.py
│   ├── chat/
│   │   ├── __init__.py
│   │   ├── chat_tab.py
│   │   └── llm_worker.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_tab.py
│   │   └── import_utils.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── config_tab.py    
│   ├── status/
│   │   └── status_tab.py
│   └── common/
│       ├── __init__.py
│       ├── query_text_edit.py
│       └── ui_utils.py
├── scripts/
│   ├── llm/
│   │   ├── llm_interface.py
│   │   └── mcp_client.py
│   ├── apps_logs/
│   │   └── scraper.log
│   ├── retrieval/
│   │   └── retrieval_core.py
│   ├── indexing/
│   │   ├── embedding_utils.py
│   │   └── index_manager.py
│   └── ingest
│   │   ├── data_loader.py
│   │   └── scrape_pdfs.py
├── cache/
│   ├── query_cache.json
│   └── corrections.json
├── app_logs/
│   ├── knowledge_llm.log
│   └── ...
├── data/
│   └── [uploaded/processed PDFs]
└── docker-compose.yml




//"api_key": "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",

If your dataset is small (<10K vectors), use IndexFlatL2.
If your dataset is large (50K+ vectors), use IndexIVFFlat with nlist: sqrt(# vectors).
Keep chunk_size: 128 and chunk_overlap: 50 for best results.

support openai, gpt4all and ollama (ollama incl streaming)
docker desktop for windows



Top-k Parameterization: In VectorStoreConnector.search(), consider allowing top_k to be controlled globally via config.

Token Budget Handling: MemoryContextManager can use token length estimates to trim context size toward max_tokens_per_context.

Source ID Robustness: In get_document_source(), validate if the index exists before accessing .iloc[doc_id].

Unit Tests: Add coverage for hybrid search fusion logic and memory updates.

Optional: Log context chunks retrieved per query if debug=True in config for traceability.