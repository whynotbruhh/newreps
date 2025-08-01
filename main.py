from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field # Import Field for more detailed schema
import os
from dotenv import load_dotenv
import google.generativeai as genai
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from typing import List, Dict, Optional # Import Optional for fields that might be null
import requests
import json

# Load environment variables from .env file
load_dotenv(override=True)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Insurance Clause Evaluator API", # Updated Title
    description="API for evaluating insurance queries against document clauses using Gemini (for embeddings), OpenRouter (for generation), and MongoDB Atlas.", # Updated Description
    version="1.0.0",
    docs_url="/"
)

# --- CORS Configuration ---
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:8000",
    "*" # WARNING: Restrict this in production.
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configuration from Environment Variables ---
GOOGLE_GEMINI_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

MONGO_USER = os.getenv("MONGO_USER")
MONGO_PASSWORD = os.getenv("MONGO_PASSWORD")
MONGO_HOST = os.getenv("MONGO_HOST")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")
MONGO_COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME")

# Validate environment variables
if not GOOGLE_GEMINI_API_KEY:
    raise ValueError("GOOGLE_GEMINI_API_KEY environment variable not set.")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable not set.")
if not all([MONGO_USER, MONGO_PASSWORD, MONGO_HOST, MONGO_DB_NAME, MONGO_COLLECTION_NAME]):
    raise ValueError("One or more MongoDB environment variables are not set.")

# --- Gemini API Configuration (for Embeddings) ---
genai.configure(api_key=GOOGLE_GEMINI_API_KEY)
GEMINI_EMBEDDING_MODEL = "models/embedding-001"

# --- OpenRouter API Configuration (for Generation) ---
OPENROUTER_GENERATION_MODEL = "google/gemini-pro-1.5" # Keep this, or try "mistralai/mistral-7b-instruct" if you prefer
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1/chat/completions"

# --- MongoDB Atlas Connection ---
mongo_client: MongoClient = None
mongo_collection = None

@app.on_event("startup")
async def startup_db_client():
    """Connects to MongoDB Atlas on FastAPI startup."""
    global mongo_client, mongo_collection
    try:
        mongo_connection_string = f"mongodb+srv://{MONGO_USER}:{MONGO_PASSWORD}@{MONGO_HOST}/?retryWrites=true&w=majority"
        mongo_client = MongoClient(mongo_connection_string)
        mongo_client.admin.command('ping') # Test connection
        mongo_collection = mongo_client[MONGO_DB_NAME][MONGO_COLLECTION_NAME]
        print("Connected to MongoDB Atlas successfully!")
    except ConnectionFailure as e:
        print(f"MongoDB Atlas connection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Database connection error: {e}")

@app.on_event("shutdown")
async def shutdown_db_client():
    """Closes MongoDB Atlas connection on FastAPI shutdown."""
    if mongo_client:
        mongo_client.close()
        print("MongoDB Atlas connection closed.")

# --- Helper Functions ---
async def get_embedding(text: str) -> List[float]:
    """Generates an embedding for the given text using the Google Gemini embedding model."""
    try:
        response = genai.embed_content(model=GEMINI_EMBEDDING_MODEL, content=text)
        return response['embedding']
    except Exception as e:
        print(f"Error generating embedding for text: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate embedding.")

async def semantic_search(query_embedding: List[float], limit: int = 5) -> List[Dict]: # Increased limit for more context
    """
    Performs a vector search in MongoDB Atlas to find the most relevant chunks.
    Requires a vector search index named 'vector_index' on the 'embedding' field.
    """
    if mongo_collection is None:
        raise HTTPException(status_code=500, detail="MongoDB collection not initialized.")

    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": 100, # Can be adjusted
                "limit": limit
            }
        },
        {"$project": {"_id": 0, "text": 1, "source_file": 1, "score": {"$meta": "vectorSearchScore"}}}
    ]

    results = []
    try:
        results = list(mongo_collection.aggregate(pipeline))
        print(f"Found {len(results)} relevant chunks.")
        return results
    except Exception as e:
        print(f"Error during MongoDB vector search: {e}")
        raise HTTPException(status_code=500, detail=f"Vector search failed: {e}")


async def generate_openrouter_response(user_query: str, context_chunks: List[Dict]) -> Dict: # Return Dict for structured response
    """
    Generates a structured response using OpenRouter based on the user query and retrieved context,
    evaluating against insurance clauses.
    """
    if not context_chunks:
        # Provide a default "rejected" decision if no context is found.
        return {
            "decision": "Rejected",
            "amount": None,
            "justification": "No relevant clauses or information found in the provided documents to evaluate the request.",
            "clauses_used": []
        }

    context_text = "\n\n".join([f"Source: {chunk.get('source_file', 'N/A')}\nClause/Text: {chunk['text']}" for chunk in context_chunks])

    # Enhanced System Prompt
    system_prompt = """
    You are an AI-powered insurance clause evaluator. Your task is to analyze a user's insurance claim query,
    extract key details (age, procedure, location, policy duration), and evaluate it strictly against the
    provided 'Document Context' which contains insurance clauses and rules.

    Your output MUST be a JSON object with the following structure:
    {
      "decision": "Approved" | "Rejected" | "Requires More Information",
      "amount": number | null, // If applicable, e.g., payout amount. Use null if not applicable.
      "justification": "string", // A clear explanation of the decision, referencing specific rules/clauses.
      "clauses_used": ["string"] // A list of direct quotes of the specific clauses from the Document Context that were directly used for the decision.
    }

    Follow these steps:
    1.  **Understand the User Query:** Identify details like:
        -   **Age:** e.g., 46 years
        -   **Procedure/Condition:** e.g., knee surgery, dental treatment
        -   **Location:** e.g., Pune, outside India
        -   **Policy Duration/Waiting Period:** e.g., 3-month-old policy, 2-year waiting period
        -   **Other relevant details** (e.g., type of illness, accident, inpatient/outpatient).
    2.  **Evaluate against Context:** Carefully read the 'Document Context'.
        -   Look for clauses related to waiting periods, specific procedure coverage, exclusions, geographical limits, age limits, etc.
        -   If a clause is relevant, apply its logic to the user's query details.
    3.  **Formulate Decision:**
        -   If all conditions for approval are met based *only* on the provided context, the decision is "Approved".
        -   If any exclusion or unmet condition (like a waiting period) is found in the context, the decision is "Rejected".
        -   If the context doesn't contain enough information to make a clear decision based on the query details, the decision is "Requires More Information".
    4.  **Provide Justification:** Explain *why* the decision was made, explicitly quoting or paraphrasing the relevant clauses from the 'Document Context'.
    5.  **List Clauses Used:** Extract the exact text of the specific clauses from the 'Document Context' that directly led to your decision.
    6.  **Output JSON:** Ensure your response is *only* the JSON object, nothing else before or after.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Document Context:\n---\n{context_text}\n---\n\nUser Claim Query: {user_query}\n\nStrictly provide your evaluation as a JSON object:"}
    ]

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000", # Replace with your actual domain for production
        "X-Title": "My Insurance Evaluator Bot" # Replace with a title for your app
    }

    payload = {
        "model": OPENROUTER_GENERATION_MODEL,
        "messages": messages,
        "max_tokens": 1000, # Increased tokens for more detailed responses
        "temperature": 0.1, # Keep temperature low for more deterministic, factual responses
        "response_format": {"type": "json_object"} # Crucial for JSON output
    }

    try:
        response = requests.post(OPENROUTER_API_BASE, headers=headers, json=payload)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        response_data = response.json()
        
        if response_data and response_data.get('choices'):
            ai_response_content = response_data['choices'][0]['message']['content']
            
            # Attempt to parse the content as JSON
            try:
                parsed_response = json.loads(ai_response_content)
                # Validate against expected keys
                if all(k in parsed_response for k in ["decision", "justification", "clauses_used"]):
                    return parsed_response
                else:
                    print(f"Parsed JSON missing required keys: {parsed_response}")
                    return {
                        "decision": "Error",
                        "amount": None,
                        "justification": "AI generated incomplete JSON response.",
                        "clauses_used": []
                    }
            except json.JSONDecodeError as e:
                print(f"AI response was not valid JSON: {ai_response_content}. Error: {e}")
                return {
                    "decision": "Error",
                    "amount": None,
                    "justification": "AI generated non-JSON response. Check API configuration or prompt.",
                    "clauses_used": []
                }
        else:
            print(f"Unexpected OpenRouter response structure: {response_data}")
            return {
                "decision": "Error",
                "amount": None,
                "justification": "Failed to get a valid response structure from the AI model.",
                "clauses_used": []
            }

    except requests.exceptions.RequestException as e:
        print(f"Error during OpenRouter API call: {e}")
        if e.response:
            print(f"OpenRouter API Error Response: {e.response.text}")
        raise HTTPException(status_code=500, detail=f"Failed to generate AI response from OpenRouter: {e}. Check OpenRouter API key, model name, and rate limits.")
    except Exception as e:
        print(f"An unexpected error occurred during OpenRouter response generation: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred during AI response generation.")


# --- API Endpoint ---
class ChatRequest(BaseModel):
    query: str

class ClauseDetail(BaseModel):
    text: str
    source_file: str
    score: float

class EvaluationResponse(BaseModel): # Renamed to better reflect purpose
    decision: str = Field(..., description="The decision made: Approved, Rejected, or Requires More Information.")
    amount: Optional[float] = Field(None, description="The payout amount, if applicable. Null otherwise.")
    justification: str = Field(..., description="A detailed explanation for the decision, referencing relevant clauses.")
    clauses_used: List[str] = Field([], description="Direct quotes of the specific clauses from the document used for the decision.")
    source_chunks_retrieved: List[ClauseDetail] = Field([], description="The raw chunks retrieved from the document store, for debugging/transparency.") # Added for transparency

@app.post("/hackrx/run", response_model=EvaluationResponse) # Endpoint changed to /evaluate
async def evaluate_claim_with_pdf(request: ChatRequest): # Function name changed
    """
    Evaluates an insurance claim query against document clauses and returns a structured decision.
    """
    user_query = request.query
    print(f"Received claim query: '{user_query}'")

    # 1. Generate embedding for the user's query
    query_embedding = await get_embedding(user_query)
    if not query_embedding:
        raise HTTPException(status_code=500, detail="Could not generate embedding for query.")

    # 2. Perform semantic search in MongoDB Atlas (increased limit to get more context for reasoning)
    relevant_chunks = await semantic_search(query_embedding, limit=10) # Fetch more chunks for better reasoning

    # 3. Generate structured response using OpenRouter
    # The AI is now expected to return a JSON object directly
    ai_structured_response = await generate_openrouter_response(user_query, relevant_chunks)

    # Prepare retrieved chunks for the response, excluding the 'embedding' field
    response_chunks_retrieved = [
        {"text": chunk['text'], "source_file": chunk.get('source_file', 'N/A'), "score": chunk['score']}
        for chunk in relevant_chunks
    ]

    # Combine AI's structured output with the retrieved chunks
    return EvaluationResponse(
        decision=ai_structured_response.get("decision", "Error"),
        amount=ai_structured_response.get("amount"),
        justification=ai_structured_response.get("justification", "AI response structure invalid."),
        clauses_used=ai_structured_response.get("clauses_used", []),
        source_chunks_retrieved=response_chunks_retrieved # Pass the raw retrieved chunks
    )

# --- Basic Health Check Endpoint ---
#@app.get("/")
#async def root():
#    return {"message": "Insurance Clause Evaluator API is running! Use /evaluate endpoint for claim queries."}
