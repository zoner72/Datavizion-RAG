Datavizion
=========

Overview
--------
Datavizion is a beginner-friendly desktop RAG implementation that lets you scrape, index, query, and manage documents with a few clicks. Powered by PyQt6, Qdrant, and your choice of LLM backends (LM studio, Ollama etc), it delivers a smooth, drag-and-drop workflow for turning websites and PDFs into an intelligent knowledge base.

Key Features
------------
• **Five intuitive tabs** for configuration, data management, API control, live status, and chat/query.  
• **Drag & Drop** support for files and folders.  
• **Website scraper** that downloads text and PDFs from websites, then builds a semantic index.  
• **Vector database integration** via Docker-powered Qdrant, with advanced quantization options.   
• **Internal API server** which you can use to port to your own website to for setting up your own assistant.  
• **Real-time log viewer** and health summary to monitor threads, index size, and last operation time.  
• **Customizable LLM settings** with templates, temperature, and response format for major LLM software, or just use openai.  
• **Built-in “intense” indexing profile** for deeper, overlapping chunks.  
• **Not geting the answer you expect, add your own answer which can be used later for model training. 

Screenshots
-----------
1. Config Tab ­– screenshots/config_tab.png  
2. Data Tab ­– screenshots/data_tab.png  
3. API Tab ­– screenshots/api_tab.png  
4. Status Tab ­– screenshots/status_tab.png  
5. Chat Tab ­– screenshots/chat_tab.png  

Installation (Easy for Beginners)
---------------------------------
1. **Install Docker**  
   Download and install Docker Desktop (Windows/macOS/Linux):  
   https://www.docker.com/get-started  
2. **Clone this repo**  
   ```bash
   git clone https://github.com/zoner72/Datavizion-RAG.git
   cd DATAVIZION-RAG
Create a virtual environment & install Python deps


python3 -m venv .venv
source .venv/bin/activate    # or `.venv\Scripts\activate` on Windows
pip install -r requirements.txt
Start Qdrant with Docker Compose

docker-compose up -d
Run the app

python main.py

Known Issues & Tips
Drag & Drop does not work when you run inside an IDE window; please launch via your system’s file explorer or terminal.
Embeddings Dir, MCP Server and CLient are currently not in use, maybe future releases if the interest is there.
I had great results with phi-4 mini, but feel free to try some other LLMs

Developed with Python 3.11 and Docker Desktop.

For very large scrapes, set scraping_timeout: null in your config.json to allow unlimited runtime.
You can find many more settings in the config.json to tinker with (see config_models.py for details)

Support & Coffee
If Datavizion saves you time and effort, please consider buying me a coffee!
https://www.buymeacoffee.com/zoner

License
This project is released under the MIT License. See LICENSE.txt for the full text.
