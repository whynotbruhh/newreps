import PyPDF2
import tiktoken
from typing import List, Dict
import os
import google.generativeai as genai
from pymongo import MongoClient

# --- Configuration ---
PDF_PATH = r"C:\Hackathon\BAJHLIP23020V012223.pdf"
MAX_TOKENS_PER_CHUNK = 500
OVERLAP_TOKENS = 50

# --- Gemini API Configuration ---
# WARNING: Hardcoding API keys is NOT recommended for production.
# Use environment variables or a secure configuration system instead.
GEMINI_API_KEY = "AIzaSyCtXETyNddsCV8YSZI05LUzAYuyfgdvlwM" # YOUR API KEY
genai.configure(api_key=GEMINI_API_KEY)

# --- MongoDB Atlas Configuration ---
# WARNING: Hardcoding passwords is NOT recommended for production.
# Replace <password> with your actual user password.
# Make sure your MongoDB Atlas connection string is correct for your cluster!
MONGO_PASSWORD = "abhiram" # YOUR MONGODB PASSWORD
MONGO_CONNECTION_STRING = f"mongodb+srv://abhiramchowdary2006:abhiram@cluster0.mjrc75v.mongodb.net/?retryWrites=true&w=majority" # Constructed string

DB_NAME = "chatbot_db" # Name of your database
COLLECTION_NAME = "pdf_chunks" # Name of the collection to store PDF data

# --- PDF Extraction and Chunking Functions ---
def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts all text from a PDF file.
    """
    text = ""
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at '{pdf_path}'")
        return ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""
    return text

def chunk_text_by_tokens(text: str, model_name: str = "cl100k_base", max_tokens: int = 500, overlap: int = 50) -> List[Dict]:
    """
    Splits text into chunks based on a token limit, with optional overlap.
    Returns a list of dictionaries, where each dict has 'chunk_id', 'text', and 'tokens'.
    """
    try:
        encoding = tiktoken.get_encoding(model_name)
    except Exception:
        print(f"Warning: tiktoken encoding '{model_name}' not found. Using 'p50k_base' instead.")
        encoding = tiktoken.get_encoding("p50k_base")

    tokens = encoding.encode(text)
    chunks = []
    chunk_id_counter = 0

    i = 0
    while i < len(tokens):
        chunk_tokens = tokens[i : i + max_tokens]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append({
            "chunk_id": chunk_id_counter,
            "text": chunk_text,
            "tokens": len(chunk_tokens)
        })
        chunk_id_counter += 1

        if i + max_tokens >= len(tokens):
            break
        else:
            i += (max_tokens - overlap)
            if i < 0:
                i = 0
    return chunks

# --- New Functions for Embeddings and DB Storage ---
def get_embedding(text: str) -> List[float]:
    """
    Generates an embedding for a given text using the Gemini embedding model.
    """
    try:
        # Ensure the text is not too long for the embedding model if there are issues
        # Or handle potential API rate limits / errors gracefully
        response = genai.embed_content(model="models/embedding-001", content=text)
        return response['embedding']
    except Exception as e:
        print(f"Error generating embedding for chunk: {e}")
        return [] # Return empty list on error

def upload_chunks_to_mongodb(chunks: List[Dict], connection_string: str, db_name: str, collection_name: str):
    """
    Connects to MongoDB Atlas and uploads text chunks with their embeddings.
    """
    client = None
    try:
        client = MongoClient(connection_string)
        db = client[db_name]
        collection = db[collection_name]

        # Add embeddings to each chunk dictionary
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            print(f"Generating embedding for chunk {i+1}/{len(chunks)}...")
            embedding = get_embedding(chunk['text'])
            if embedding: # Only add if embedding generation was successful
                chunk['embedding'] = embedding
                chunk['source_file'] = os.path.basename(PDF_PATH)
                processed_chunks.append(chunk)
            else:
                print(f"Skipping chunk {i} due to embedding error.")

        # Insert chunks into MongoDB
        if processed_chunks:
            # Optional: Clear existing collection if you're re-uploading
            # collection.delete_many({})
            # print(f"Cleared existing data in '{collection_name}' collection.")

            collection.insert_many(processed_chunks)
            print(f"Successfully uploaded {len(processed_chunks)} chunks to '{collection_name}' collection in '{db_name}' database.")
        else:
            print("No valid chunks with embeddings to upload.")

    except Exception as e:
        print(f"Error connecting to or uploading to MongoDB Atlas: {e}")
    finally:
        if client:
            client.close() # Close the connection when done

if __name__ == "__main__":
    print("Starting PDF processing and MongoDB upload...")

    # 1. Extract and Chunk PDF
    full_text = extract_text_from_pdf(PDF_PATH)

    if full_text:
        print(f"Extracted {len(full_text)} characters from PDF.")
        text_chunks = chunk_text_by_tokens(full_text, max_tokens=MAX_TOKENS_PER_CHUNK, overlap=OVERLAP_TOKENS)
        print(f"Generated {len(text_chunks)} chunks.")

        # 2. Upload Chunks with Embeddings to MongoDB
        if text_chunks:
            print("\nProceeding to generate embeddings and upload to MongoDB Atlas...")
            upload_chunks_to_mongodb(text_chunks, MONGO_CONNECTION_STRING, DB_NAME, COLLECTION_NAME)
        else:
            print("No chunks generated for embedding and upload.")
    else:
        print("PDF processing failed. Please review errors above and ensure PDF is valid and readable.")

    print("\nProcess complete.")