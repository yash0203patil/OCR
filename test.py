# %%writefile app.py
import streamlit as st
import base64
import pandas as pd
import json
from pathlib import Path
from mistralai import Mistral, ImageURLChunk, TextChunk
from pydantic import BaseModel
from PyPDF2 import PdfReader


# ✅ Initialize Mistral client with API key
api_key = "XpbDVszRJPoE41i9T0gUjyhq3Tmhdy1T"  # Replace with your API key
client = Mistral(api_key=api_key)

# ✅ Define the response structure for OCR
class StructuredOCR(BaseModel):
    file_name: str
    topics: list[str]
    languages: str
    ocr_contents: dict


def structured_ocr(image_path: str) -> StructuredOCR:
    """
    Process an image using OCR and extract structured data.

    Args:
        image_path: Path to the image file to process

    Returns:
        StructuredOCR object containing the extracted data
    """
    # ✅ Validate input file
    image_file = Path(image_path)
    assert image_file.is_file(), "The provided image path does not exist."

    # ✅ Read and encode the image file
    encoded_image = base64.b64encode(image_file.read_bytes()).decode()
    base64_data_url = f"data:image/jpeg;base64,{encoded_image}"

    # ✅ Process the image using OCR
    image_response = client.ocr.process(
        document=ImageURLChunk(image_url=base64_data_url),
        model="mistral-ocr-latest"
    )
    image_ocr_markdown = image_response.pages[0].markdown
    print(image_ocr_markdown)

    # ✅ Parse the OCR result into a structured JSON response
    chat_response = client.chat.parse(
        model="pixtral-12b-latest",
        messages=[
            {
                "role": "user",
                "content": [
                    ImageURLChunk(image_url=base64_data_url),
                    TextChunk(text=(
                    f"This is the image's OCR in markdown:\n{image_ocr_markdown}\n\n"
                    "Convert this into a structured JSON response with the OCR contents in a sensible dictionary. "
                    "If an ID like 'WZU1004977' is present in the markdown, include it in the response. "
                    "If the ID is not found, ignore it."
                ))
                ]
            }
        ],
        response_format=StructuredOCR,
        temperature=0
    )

    return chat_response.choices[0].message.parsed


def process_pdf(pdf_path: str) -> StructuredOCR:
    """
    Process a PDF file using OCR and extract structured data from all pages.
    
    Args:
        pdf_path: Path to the PDF file to process.

    Returns:
        StructuredOCR object containing the extracted data.
    """
    pdf_reader = PdfReader(pdf_path)
    all_ocr_contents = {}

    for i, page in enumerate(pdf_reader.pages):
        text = page.extract_text()
        if not text:
            continue

        # ✅ Process the page using Mistral OCR
        page_response = client.chat.parse(
            model="pixtral-12b-latest",
            messages=[
                {
                    "role": "user",
                    "content": [
                        TextChunk(text=(
                            f"This is page {i+1} of the PDF:\n{text}\n"
                            "Convert this into a structured JSON response "
                            "with the OCR contents in a sensible dictionary."
                        ))
                    ]
                }
            ],
            response_format=StructuredOCR,
            temperature=0
        )

        # ✅ Merge all pages into a single dict
        if page_response.choices:
            page_data = page_response.choices[0].message.parsed.ocr_contents
            all_ocr_contents.update({f"page_{i+1}": page_data})

    # ✅ Return a consolidated response
    return StructuredOCR(
        file_name=Path(pdf_path).name,
        topics=["PDF document"],
        languages="English",
        ocr_contents=all_ocr_contents
    )

# ✅ Encode PDF to base64 for preview
def get_pdf_base64(pdf_path):
    """Convert PDF to base64 for displaying in iframe."""
    with open(pdf_path, "rb") as pdf_file:
        pdf_base64 = base64.b64encode(pdf_file.read()).decode('utf-8')
    return f"data:application/pdf;base64,{pdf_base64}"


# ✅ Flatten nested JSON to single-level dictionary
def flatten_json(nested_data, parent_key="", separator="."):
    flat_dict = {}

    for key, value in nested_data.items():
        new_key = f"{parent_key}{separator}{key}" if parent_key else key

        if isinstance(value, dict):
            flat_dict.update(flatten_json(value, new_key, separator=separator))
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    flat_dict.update(flatten_json(item, f"{new_key}[{i}]", separator=separator))
                else:
                    flat_dict[f"{new_key}[{i}]"] = item
        else:
            flat_dict[new_key] = value

    return flat_dict


# ✅ Streamlit App Interface
# st.set_page_config(page_title="OCR Key-Value Extraction")
st.sidebar.image("OB Logo.png", use_container_width=False)
# st.sidebar.markdown("----")
# st.sidebar.title("📄 OCR TOOL")


# ✅ File uploader for image and PDF files
uploaded_file = st.file_uploader("📂 Upload an image or PDF", type=["jpg", "jpeg", "png", "pdf"])

if uploaded_file is not None:
    temp_path = "temp_file"
    if uploaded_file.type == "application/pdf":
        temp_path += ".pdf"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # ✅ Show PDF preview in sidebar
        pdf_base64_url = get_pdf_base64(temp_path)
        pdf_iframe = f"""
        <iframe
            src="{pdf_base64_url}"
            width="100%"
            height="500"
            style="border: none;"
        ></iframe>
        """
        st.sidebar.markdown("📄 **PDF Preview:**")
        st.sidebar.markdown(pdf_iframe, unsafe_allow_html=True)
    else:
        temp_path += ".jpg"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())
        
        # ✅ Show image preview in sidebar
        st.sidebar.image(uploaded_file, caption="Uploaded Document", use_container_width=True)

# ✅ Initialize session state to store results
if "structured_response" not in st.session_state:
    st.session_state.structured_response = None


# ✅ Button to analyze the document
if st.button("🔍 Analyze Document") and uploaded_file is not None:
    with st.spinner("⏳ Processing document..."):
        # Save uploaded file temporarily
        temp_path = "temp_file"
        
        if uploaded_file.type == "application/pdf":
            temp_path += ".pdf"
        else:
            temp_path += ".jpg"

        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        # ✅ Check if it's a PDF or an image
        try:
            if uploaded_file.type == "application/pdf":
                st.session_state.structured_response = process_pdf(temp_path)
            else:
                st.session_state.structured_response = structured_ocr(temp_path)

            st.success("✅ OCR completed successfully!")
        except Exception as e:
            st.error(f"❌ Error processing document: {str(e)}")


# ✅ Display results if available
if st.session_state.structured_response:
    structured_response = st.session_state.structured_response

    # ✅ Show metadata in the sidebar
#     st.sidebar.markdown("---")
    st.sidebar.write(f"📄 **File Name:** `{structured_response.file_name}`")
    st.sidebar.write(f"🗣️ **Detected Language(s):** `{structured_response.languages}`")
    st.sidebar.write(f"📚 **Topics Identified:** {', '.join(structured_response.topics)}")
#     st.sidebar.markdown("---")

    # ✅ Flattened JSON to single-level key-value pairs
#     st.subheader("📊 Extracted Key-Value Pairs (Flattened)")

    # Flatten JSON before displaying
    flattened_data = flatten_json(structured_response.ocr_contents)

    # ✅ Convert flattened data to Pandas DataFrame for displaying
    flattened_df = pd.DataFrame(flattened_data.items(), columns=["Key", "Value"])

    # ✅ Display extracted values in a non-editable table
    st.dataframe(flattened_df)
#     st.dataframe(flattened_df.set_index(flattened_df.columns[0]))

    # ✅ Download as CSV
    csv_data = flattened_df.to_csv(index=False).encode("utf-8")
    st.sidebar.download_button(
        label="📥 Download CSV",
        data=csv_data,
        file_name="ocr_results.csv",
        mime="text/csv",
    )

    # ✅ Download as JSON
    json_data = json.dumps(flattened_data, indent=4, ensure_ascii=False)
    st.sidebar.download_button(
        label="📥 Download JSON",
        data=json_data.encode("utf-8"),
        file_name="ocr_results.json",
        mime="application/json",
    )

# ✅ Footer
st.markdown("---")

