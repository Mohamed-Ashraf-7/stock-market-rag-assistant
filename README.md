# 📊 Stock Market RAG Assistant

An AI-powered **RAG (Retrieval-Augmented Generation) assistant** that lets you ask questions about a **Stock Market PDF** using **Google Generative AI (Gemini)**, **LangChain**, and **LangGraph**.

You can interact with it in two ways:
1. **CLI Mode (Terminal)** → Ask questions directly in the console.  
2. **GUI Mode (Web)** → Run with Streamlit for a chat-style interface.

---

## 🚀 Features
- Load and process a Stock Market PDF
- Embed & store chunks in **ChromaDB**
- Retrieval-Augmented QA using **Gemini 2.5 Flash**
- Tool-driven agent built with **LangGraph**
- Chat-style web interface with **Streamlit**

---

## 📂 Project Structure
stock-market-rag-assistant/
├── gui.py # Streamlit chat UI
├── rag_agent_trial.py # RAG agent logic (LangGraph + Gemini)
├── keys.py # Stores your Google API key (⚠️ ignored in Git)
├── requirements.txt # Dependencies
├── README.md # Documentation
└── data/
└── Stock_Market_Performance_2024.pdf

yaml
Copy code

---

## 🛠️ Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/stock-market-rag-assistant.git
   cd stock-market-rag-assistant
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Add your Google API Key in keys.py:

python
Copy code
GOOGLE_API_KEY = "your_api_key_here"
💻 Usage
▶️ CLI Mode
Run directly in the terminal:

bash
Copy code
python rag_agent_trial.py
Type your questions and get answers from the PDF.
Exit anytime with:

bash
Copy code
exit
🌐 GUI Mode (Streamlit)
Run the Streamlit app:

bash
Copy code
streamlit run gui.py
When Streamlit starts, type exit in the terminal where the CLI loop is running.

This stops the console chat.

The Streamlit web app stays running.

Open the link shown in the terminal (usually http://localhost:8501) → now you have a chat interface like ChatGPT.

📝 Notes
keys.py is ignored with .gitignore → your API key is safe.

Replace Stock_Market_Performance_2024.pdf in /data with your own documents if you want.

If you restart the app, run the same steps again.

📌 Example (GUI)
yaml
Copy code
👤 You: What are the key stock market trends?
🤖 Assistant: The document highlights strong growth in technology and renewable energy sectors...
pgsql
Copy code

---

✅ Copy this into your `README.md` and your repo will look professional and easy to use.  

Do you want me to also **edit your Python code** so you don’t need to type `exit` when running Streamlit (auto-detects GUI vs CLI)?




