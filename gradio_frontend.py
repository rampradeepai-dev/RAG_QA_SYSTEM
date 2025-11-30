import gradio as gr
import requests
import os

# ----------------------------
# Backend API URL
# ----------------------------
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

UPLOAD_ENDPOINT = f"{API_URL}/documents"
QUERY_ENDPOINT = f"{API_URL}/query"


# ----------------------------
# Upload PDF → receive document_id
# ----------------------------
def upload_pdf(pdf_file):
    if pdf_file is None:
        return "❌ No file selected", None

    try:
        with open(pdf_file.name, "rb") as f:
            files = {"file": (pdf_file.name, f, "application/pdf")}
            response = requests.post(UPLOAD_ENDPOINT, files=files)

        if response.status_code == 200:
            data = response.json()
            doc_id = data["document_id"]
            return f"✅ Uploaded successfully! Document ID:\n{doc_id}", doc_id
        else:
            return f"❌ Upload failed:\n{response.text}", None
    except Exception as e:
        return f"❌ Error:\n{str(e)}", None


# ----------------------------
# Ask a question → get answer
# ----------------------------
def ask_question(question, document_id, top_k):
    if not document_id:
        return "❌ Upload a document first."

    payload = {
        "question": question,
        "document_id": document_id,
        "top_k": int(top_k),
    }

    try:
        response = requests.post(QUERY_ENDPOINT, json=payload)
        if response.status_code == 200:
            return response.json()["answer"]
        else:
            return f"❌ Query failed:\n{response.text}"
    except Exception as e:
        return f"❌ Error:\n{str(e)}"


# ----------------------------
# Gradio UI Layout
# ----------------------------
with gr.Blocks(title="RAG QA System") as demo:
    gr.HTML("""
        <style>
            footer, .footer, #footer {
                display: none !important;
            }
        </style>
    """)
    gr.Markdown("# 📄 RAG QA System")
    gr.Markdown(
        "Upload a PDF → Embed & Store → Ask Questions\n"
    )

    with gr.Tab("📤 Upload Document"):
        pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
        upload_button = gr.Button("Upload & Process")
        upload_status = gr.Textbox(label="Status", lines=2)
        document_id_box = gr.Textbox(label="Document ID (auto-filled)", interactive=False)

        upload_button.click(
            upload_pdf,
            inputs=[pdf_input],
            outputs=[upload_status, document_id_box]
        )

    with gr.Tab("❓ Ask Questions"):
        question_box = gr.Textbox(label="Your Question")
        top_k_slider = gr.Slider(1, 10, value=4, label="Top-K Context Chunks")
        ask_button = gr.Button("Get Answer")
        answer_box = gr.Textbox(label="Answer", lines=6)

        ask_button.click(
            ask_question,
            inputs=[question_box, document_id_box, top_k_slider],
            outputs=[answer_box]
        )

demo.launch(server_name="127.0.0.1", server_port=7860)
