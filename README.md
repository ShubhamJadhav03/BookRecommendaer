# 📚 Semantic Book Recommender using LLMs

An intelligent book recommendation system powered by Large Language Models (LLMs), semantic search, and emotion-aware filtering. Users can input natural language queries like _"a suspenseful story with emotional twists"_ and receive personalized recommendations with options to filter by **genre** and **tone**.

![image](https://github.com/user-attachments/assets/97d16837-0c49-4687-8104-8c2a4f6ef7fe)



---

## 🚀 Features

✅ Natural language book search  
✅ Emotion-based filtering (Joy, Sadness, Suspense, etc.)  
✅ Genre classification (Fiction/Non-fiction) using Zero-shot LLMs  
✅ Semantic similarity search via vector embeddings  
✅ Interactive UI built with Gradio (dark theme)  
✅ All models are **free and open-source**

---

## 🧠 Tech Stack

- **Python 3.11**
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/index)
- [LangChain](https://www.langchain.com/)
- [ChromaDB](https://www.trychroma.com/)
- [Gradio](https://gradio.app/)
- `pandas`, `numpy`, `dotenv`, `tqdm`, and more

---

## 📦 Installation

```bash
git clone https://github.com/ShubhamJadhav03/BookRecommendaer.git
cd BookRecommendaer

python -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate

pip install -r requirements.txt
