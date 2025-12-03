from gradio import (
    Blocks,
    HTML,
    Markdown,
    Files,
    Button,
    Textbox,
    Slider,
    State,
)
import gradio as gr
import requests
import os

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
UPLOAD_ENDPOINT = f"{API_URL}/documents"
QUERY_ENDPOINT = f"{API_URL}/query"
FETCH_ENDPOINT = f"{API_URL}/documents/index"

# 🔹 How many documents per page in the dropdown
DOCS_PER_PAGE = 50


def upload_pdfs(
    pdf_files,
    existing_ids,
    current_total_pages,
    current_page,
):
    if not pdf_files:
        # Do not change dropdown or pagination if nothing uploaded
        return (
            "❌ No files selected",
            existing_ids,
            gr.update(),              # leave dropdown as-is
            gr.update(value=""),      # clear question box
            current_total_pages,
            current_page,
        )

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
                messages.append(
                    f"❌ {filename} → Upload failed: "
                    f"{response.text}"
                )
        except Exception as e:
            messages.append(f"❌ {filename} → Error: {str(e)}")

    status_text = "\n".join(messages)

    # 🔹 Recompute total pages based on updated_ids
    total_pages = max(
        1,
        (
            len(updated_ids)
            + DOCS_PER_PAGE
            - 1
        ) // DOCS_PER_PAGE,
    )

    # 🔹 Normalize current_page within bounds
    try:
        page = int(current_page) if current_page is not None else 1
    except Exception:
        page = 1

    if page < 1:
        page = 1
    if page > total_pages:
        page = total_pages

    # Use pagination to get dropdown update and corrected page
    # Ignore search_clear here
    (
        dropdown_update,
        fixed_page,
        _,
    ) = paginate_docs(
        page,
        updated_ids,
        total_pages,
    )

    # Clear question box after upload
    question_reset = gr.update(value="")

    return (
        status_text,
        updated_ids,
        dropdown_update,
        question_reset,
        total_pages,
        fixed_page,
    )


def ask_question(question, dropdown_label, top_k, id_state):
    if dropdown_label == "-- select --" or not dropdown_label:
        return "❌ Please select a document."

    # Convert UI label → doc_id
    match = next(
        (
            item
            for item in id_state
            if item["label"] == dropdown_label
        ),
        None,
    )

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
        resp = requests.get(FETCH_ENDPOINT)
        if resp.status_code != 200:
            return (
                [],
                gr.update(
                    choices=["-- select --"],
                    value="-- select --",
                ),
                1,
            )

        items = resp.json()
        state = [
            {
                "label": f"{item['filename']} ({item['document_id']})",
                "value": item["document_id"],
            }
            for item in items
        ]

        if not state:
            return (
                [],
                gr.update(
                    choices=["-- select --"],
                    value="-- select --",
                ),
                1,
            )

        total_pages = max(1, (len(state) + DOCS_PER_PAGE - 1) // DOCS_PER_PAGE)

        # Initialize with page 1
        dropdown, _, _ = paginate_docs(1, state, total_pages)

        return state, dropdown, total_pages

    except Exception:
        return [], gr.update(choices=["-- select --"], value="-- select --"), 1


def filter_docs(search_text, id_state):
    """
    Filters all documents (id_state) by search text.
    Returns dropdown update.
    """
    if not id_state:
        return gr.update(choices=["-- select --"], value="-- select --")

    query = (search_text or "").strip().lower()

    if not query:
        # No search → show only placeholder
        return gr.update(choices=["-- select --"], value="-- select --")

    # Filter by label substring match
    results = [
        item["label"] for item in id_state
        if query in item["label"].lower()
    ]

    # Limit results for UI performance
    results = results[:30]

    if not results:
        return gr.update(choices=["-- select --"], value="-- select --")

    choices = ["-- select --"] + results
    return gr.update(choices=choices, value="-- select --")


def paginate_docs(page, id_state, total_pages):
    """
    Paginate safely. If page > total_pages or < 1, fix it and
    return:
      - dropdown update
      - corrected page
      - cleared search box
    """
    # No documents at all
    if not id_state:
        return (
            gr.update(choices=["-- select --"], value="-- select --"),
            1,                      # corrected page
            gr.update(value=""),    # 🔹 clear Search Documents
        )

    # convert safely
    try:
        page = int(page) if page is not None else 1
    except Exception:
        page = 1

    # clamp within valid bounds
    if total_pages is None:
        total_pages = max(
            1,
            (
                len(id_state)
                + DOCS_PER_PAGE
                - 1
            ) // DOCS_PER_PAGE,
        )

    try:
        total_pages = int(total_pages)
    except Exception:
        total_pages = max(
            1,
            (
                len(id_state)
                + DOCS_PER_PAGE
                - 1
            ) // DOCS_PER_PAGE,
        )

    if page < 1:
        page = 1
    if page > total_pages:
        page = total_pages

    start = (page - 1) * DOCS_PER_PAGE
    end = start + DOCS_PER_PAGE
    page_items = id_state[start:end]

    if not page_items:
        return (
            gr.update(choices=["-- select --"], value="-- select --"),
            page,
            gr.update(value=""),  # 🔹 clear Search Documents
        )

    choices = ["-- select --"] + [item["label"] for item in page_items]

    return (
        gr.update(choices=choices, value="-- select --"),
        page,                   # corrected page
        gr.update(value=""),    # 🔹 clear Search Documents
    )


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
        # 🔹 Pagination controls
        total_pages = gr.Number(
            label="Total Pages",
            value=1,
            interactive=False,
            precision=0,
        )
        page_number = gr.Number(label="Page", value=1, precision=0)
        search_box = Textbox(
            label="Search Documents",
            placeholder="Type keyword…",
        )

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
            inputs=[
                question_box,
                doc_id_dropdown,
                top_k_slider,
                doc_ids_state,
            ],
            outputs=[answer_box]
        )

        # When the page changes, re-paginate
        # and correct the page value if needed.
        page_number.change(
            fn=paginate_docs,
            inputs=[page_number, doc_ids_state, total_pages],
            outputs=[doc_id_dropdown, page_number, search_box],
        )

        # Search update (overrides pagination)
        search_box.change(
            fn=filter_docs,
            inputs=[search_box, doc_ids_state],
            outputs=[doc_id_dropdown],
        )

    # Load existing docs whenever the page loads / refreshes
    demo.load(
        fetch_existing_docs,
        inputs=None,
        outputs=[doc_ids_state, doc_id_dropdown, total_pages],
    )

    # 🔹 Upload now also updates pagination info
    upload_results = upload_button.click(
        upload_pdfs,
        inputs=[pdf_input, doc_ids_state, total_pages, page_number],
        outputs=[
            upload_status,
            doc_ids_state,
            doc_id_dropdown,
            question_box,
            total_pages,
            page_number,
        ],
    )


demo.launch(server_name="127.0.0.1", server_port=7860)
