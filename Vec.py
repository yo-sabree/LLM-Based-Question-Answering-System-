import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
#streamlit run Vec.py --server.enableXsrfProtection false
# Set up the Streamlit app
st.set_page_config(page_title="RAG APP")
st.header("Question your pdf")

# Sidebar for file upload
with st.sidebar:
    st.subheader("Post your document below")
    pdfs = st.file_uploader("Click to upload", accept_multiple_files=True)
    if st.button("Submit"):
        if pdfs:
            with st.spinner("Processing"):
                texts = ""
                for pdf in pdfs:
                    read_pdf = PdfReader(pdf)
                    for page in read_pdf.pages:
                        texts += page.extract_text()

                # Split the texts into chunks
                split = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                text_chunks = split.split_text(texts)

                # Initialize the embedding function and Chroma database
                embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
                db = Chroma.from_texts(text_chunks, embedding_function)

                st.success("PDFs processed and stored successfully in the database. You can now enter your query.")
                st.session_state['db'] = db
        else:
            st.write("Please upload at least one PDF document.")

# Input field for the user's question
if 'db' in st.session_state:
    inp = st.text_input("Enter your question")
    if inp:
        # Perform similarity search
        docs = st.session_state['db'].similarity_search(inp, k=5)  # Retrieve top 5 relevant documents

        # Display the most relevant document content
        if docs:
            st.write("Answers based on the most relevant documents:")
            for i, doc in enumerate(docs):
                with st.expander(f"Document {i+1} Content"):
                    st.markdown(doc.page_content)
        else:
            st.write("No relevant documents found.")
