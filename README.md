Datavizion
Version 0.3.0
<!-- VERSION_PLACEHOLDER -->

Contact me for collab on this repo
=========

Overview
--------
Datavizion is a beginner‑friendly desktop RAG tool that lets you scrape, index, query, and manage documents with just a few clicks. Built on PyQt6, Qdrant, and your choice of LLM backends, it provides a seamless drag‑and‑drop workflow for turning websites and PDFs into an intelligent, updatable knowledge base.

Key Features
------------
• **Five intuitive tabs** for configuration, data management, API control, live status, and chat/query.  
• **Drag & Drop** support for files and folders (note: doesn’t work inside IDEs—launch from your OS file explorer).  
• **Website scraper** that downloads text and PDFs, then builds a semantic index.  
• **Vector database integration** via Docker-powered Qdrant, with advanced quantization options.  
• **Internal API server** which you can turn on/off from the GUI.  
• **Real-time log viewer** and health summary to monitor threads, index size, and last operation time.  
• **Customizable LLM settings** with templates, temperature, and response format.  
• **Built-in “intense” indexing profile** for deeper, overlapping chunks.  

Screenshots
-----------
![Alt text](/screenshots/config_tab.png?raw=true "Config Tab")  
![Alt text](/screenshots/data_tab.png?raw=true "Data Tab")  
![Alt text](/screenshots/api_tab.png?raw=true "API Tab Tab")  
![Alt text](/screenshots/status_tab.png?raw=true "Status Tab")  
![Alt text](/screenshots/chat_tab.png?raw=true "Chat Tab")  

Installation (Easy for Beginners)
---------------------------------
1. **Install Docker**  
   Download and install Docker Desktop (Windows/macOS/Linux):  
   https://www.docker.com/get-started  
2. **Clone this repo**  
   ```bash
   git clone https://github.com/zoner72/Datavizion-RAG
   cd DATAVIZION-RAG
Create a virtual environment & install Python deps


python3 -m venv .venv
source .venv/bin/activate    # or `.venv\Scripts\activate` on Windows
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
Start Qdrant with Docker Compose

docker-compose up -d
Run the app

python main.py

Known Issues & Tips
Drag & Drop does not work when you run inside an IDE window; please launch via your system’s file explorer or terminal.
Embeddings Dir, MCP Server and CLient are currently not in use, maybe future releases if the interest is there.

Developed with Python 3.11 and Docker Desktop.

For very large scrapes, set scraping_timeout: null in your config.json to allow unlimited runtime.
You can find many more settings in the config.json to tinker with (see config_models.py for details)

Support & Coffee
If Datavizion saves you time and effort, please consider buying me a coffee!
https://www.buymeacoffee.com/zoner

License
This project is released under the MIT License. See LICENSE.txt for the full text.
