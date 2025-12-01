from gradio import Blocks, HTML, Markdown, Files, Button, Textbox, Slider, Dropdown, State
import gradio as gr
import requests
import os

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
UPLOAD_ENDPOINT = f"{API_URL}/documents"
QUERY_ENDPOINT = f"{API_URL}/query"

def upload_pdfs(pdf_files, existing_ids):
    if not pdf_files:
        return "❌ No files selected", existing_ids, gr.update(choices=existing_ids)

    messages = []
    updated_ids = existing_ids.copy()

    for pdf_file in pdf_files:
        try:
            filename = os.path.basename(pdf_file.name)
            # Open as real binary to send proper PDF bytes
            with open(pdf_file.name, "rb") as f:
                files = {"file": (filename, f, "application/pdf")}
                response = requests.post(UPLOAD_ENDPOINT, files=files)

            if response.status_code == 200:
                data = response.json()
                doc_id = data["document_id"]
                # Store (label + value)
                updated_ids.append({
                    "label": f"{filename} ({doc_id})",
                    "value": doc_id
                })
                messages.append(f"✅ {filename} → Document ID: {doc_id}")
            else:
                messages.append(f"❌ {filename} → Upload failed: {response.text}")
        except Exception as e:
            messages.append(f"❌ {filename} → Error: {str(e)}")

    status_text = "\n".join(messages)

    # Always keep placeholder at the top
    dropdown_choices = ["-- select --"] + [item["label"] for item in updated_ids]

    dropdown_update = gr.update(
        choices=dropdown_choices,
        value="-- select --",   # ALWAYS default to placeholder
    )

    return status_text, updated_ids, dropdown_update, gr.update(value="")


def ask_question(question, dropdown_label, top_k, id_state):
    if dropdown_label == "-- select --" or not dropdown_label:
        return "❌ Please select a document."

    # Convert UI label → doc_id
    match = next((item for item in id_state if item["label"] == dropdown_label), None)

    if not match:
        return "❌ Invalid document selection."

    document_id = match["value"]  # the real uuid
    payload = {
        "question": question,
        "document_id": document_id,
        "top_k": int(top_k)
    }

    try:
        response = requests.post(QUERY_ENDPOINT, json=payload)
        if response.status_code == 200:
            return response.json()["answer"]
        else:
            return f"❌ Query failed:\n{response.text}"
    except Exception as e:
        return f"❌ Error:\n{str(e)}"

def fetch_existing_docs():
    try:
        resp = requests.get(f"{API_URL}/documents/index")
        if resp.status_code != 200:
            return [], gr.update(choices=["-- select --"], value="-- select --")

        items = resp.json()  # list of {document_id, filename}
        state = [
            {
                "label": f"{item['filename']} ({item['document_id']})",
                "value": item["document_id"],
            }
            for item in items
        ]
        if not state:
            return [], gr.update(
                choices=["-- select --"],
                value="-- select --"
            )
        choices = ["-- select --"] + [s["label"] for s in state]
        return state, gr.update(choices=choices, value="-- select --")
    except Exception:
        return [], gr.update(choices=["-- select --"], value="-- select --")


with Blocks(title="RAG QA System") as demo:
    HTML("""
        <style>
            footer, .footer, #footer {
                display: none !important;
            }
        </style>
    """)
    Markdown("# 📄 RAG QA System")
    Markdown("Upload PDFs → Build your vector store → Ask questions")

    doc_ids_state = State([])

    with gr.Tab("📤 Upload Documents"):
        pdf_input = Files(label="Upload PDF(s)", file_types=[".pdf"])
        upload_button = Button("Upload & Process")
        upload_status = Textbox(label="Status", lines=8)
        
    with gr.Tab("❓ Ask Questions"):
        doc_id_dropdown = gr.Dropdown(
            label="Select Document",
            choices=["-- select --"],
            value="-- select --",
            interactive=True,
        )

        question_box = Textbox(label="Your Question")
        top_k_slider = Slider(1, 10, value=4, label="Top-K Chunks")
        ask_button = Button("Get Answer")
        answer_box = Textbox(label="Answer", lines=6)

        ask_button.click(
            ask_question,
            inputs=[question_box, doc_id_dropdown, top_k_slider, doc_ids_state],
            outputs=[answer_box]
        )

    # Load existing docs whenever the page loads / refreshes
    demo.load(
        fetch_existing_docs,
        inputs=None,
        outputs=[doc_ids_state, doc_id_dropdown],
    )

    upload_results = upload_button.click(
        upload_pdfs,
        inputs=[pdf_input, doc_ids_state],
        outputs=[upload_status, doc_ids_state, doc_id_dropdown, question_box],
    )

demo.launch(server_name="127.0.0.1", server_port=7860)
