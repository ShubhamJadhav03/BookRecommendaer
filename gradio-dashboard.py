import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

import gradio as gr

load_dotenv()

#books Cover
books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)


from langchain_core.documents import Document

def load_txt_utf8(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    return [Document(page_content=content)]

raw_documents = load_txt_utf8("tagged_description.txt")

text_splitter = CharacterTextSplitter(separator="\n", chunk_size=0, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db_books = Chroma.from_documents(
    documents,
    embedding=embedding
)

def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:

    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return book_recs


def recommend_books(
        query: str,
        category: str,
        tone: str
):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))
    return results

categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

custom_theme = gr.themes.Base(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
).set(
    body_background_fill="linear-gradient(to right, #0f2027, #203a43, #2c5364)",
    body_text_color="#ffffff",
    input_background_fill="#1e293b",
    input_border_color="#334155",
    block_background_fill="#111827",
    block_shadow="0px 2px 10px rgba(0,0,0,0.5)"
)


with gr.Blocks(theme=custom_theme, css="body { font-family: 'Segoe UI', sans-serif; }") as dashboard:
    gr.Markdown("<h1 style='text-align: center; color: #93c5fd;'>📚 Semantic Book Recommender</h1>")

    with gr.Row(equal_height=True):
        user_query = gr.Textbox(
            label="💬 Describe your ideal book:",
            placeholder="e.g., A suspenseful mystery with emotional depth",
            lines=2,
        )
        category_dropdown = gr.Dropdown(
            choices=categories,
            label="📂 Category",
            value="All"
        )
        tone_dropdown = gr.Dropdown(
            choices=tones,
            label="🎭 Emotional Tone",
            value="All"
        )
        submit_button = gr.Button("🔍 Find Books", scale=1)

    gr.Markdown("---")
    gr.Markdown("## ✨ Top Recommendations")

    output = gr.Gallery(
        label="📖 Books you may love",
        columns=4,
        rows=2,
        object_fit="contain",
        height="auto",
        show_label=False,
    )

    submit_button.click(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=output,
    )


if __name__ == "__main__":
    dashboard.launch()