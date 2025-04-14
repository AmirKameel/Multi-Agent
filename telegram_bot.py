from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Query, Body, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os
import uuid
import time
import json
import re
from supabase import create_client, Client
from dotenv import load_dotenv
import tiktoken
from openai import OpenAI
import numpy as np
from io import BytesIO
import PyPDF2

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Company Policy RAG System", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Supabase client
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase_client: Client = create_client(supabase_url, supabase_key)

# Models
class DocumentMetadata(BaseModel):
    title: str
    description: Optional[str] = None
    language: Optional[str] = "ar"  # Default to Arabic

class QueryRequest(BaseModel):
    query: str
    approach: str = Field(..., description="Either 'rag' or 'full_context'")
    
class QueryResponse(BaseModel):
    answer: str
    approach: str
    processing_time: float
    sources: Optional[List[Dict[str, Any]]] = None

# Utility functions
def get_embedding(text: str) -> list:
    """Get embedding for text using OpenAI API"""
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    try:
        # Try to use gpt-4o specific encoding
        encoding = tiktoken.encoding_for_model("gpt-4o")
    except KeyError:
        # Fall back to cl100k_base encoding used by gpt-4 and newer models if gpt-4o is not available
        encoding = tiktoken.get_encoding("cl100k_base")
    
    num_tokens = len(encoding.encode(string))
    return num_tokens

def chunk_text(text: str, chunk_size: int = 300, overlap: int = 100) -> List[Dict[str, Any]]:
    """
    Split text into chunks with token-based sizing and better Arabic text handling
    
    Returns a list of dictionaries with content and metadata
    """
    chunks = []
    
    # Enhanced pattern for Arabic section headers and bullet points
    section_pattern = r'(❖.*?|•.*?|[\u0600-\u06FF]+\s*[:：].*?)(?=❖|•|[\u0600-\u06FF]+\s*[:：]|\Z)'
    sections = re.findall(section_pattern, text, re.DOTALL)
    
    if not sections:  # If no sections found, create chunks by paragraphs
        paragraphs = text.split('\n\n')
        for para_idx, paragraph in enumerate(paragraphs):
            if not paragraph.strip():
                continue
                
            # Get encoding for the model
            try:
                encoding = tiktoken.encoding_for_model("gpt-4o")
            except KeyError:
                encoding = tiktoken.get_encoding("cl100k_base")
                
            para_tokens = encoding.encode(paragraph)
            
            # If paragraph is small enough, keep it as one chunk
            if len(para_tokens) <= chunk_size:
                chunks.append({
                    "content": paragraph,
                    "metadata": {
                        "section": f"Paragraph {para_idx + 1}",
                        "section_idx": para_idx
                    }
                })
            else:
                # Split large paragraphs into chunks
                for i in range(0, len(para_tokens), chunk_size - overlap):
                    chunk_tokens = para_tokens[i:i + chunk_size]
                    chunk_text = encoding.decode(chunk_tokens)
                    
                    chunks.append({
                        "content": chunk_text,
                        "metadata": {
                            "section": f"Paragraph {para_idx + 1}",
                            "section_idx": para_idx,
                            "chunk_idx": i // (chunk_size - overlap)
                        }
                    })
    else:
        for section_idx, section in enumerate(sections):
            # Clean the section text
            section = section.strip()
            
            # Extract section title - enhanced for Arabic
            title_match = re.match(r'(❖.*?|•.*?|[\u0600-\u06FF]+\s*[:：])(?=\s|$)', section)
            section_title = title_match.group(1).strip() if title_match else f"Section {section_idx + 1}"
            
            # Get encoding for the model
            try:
                encoding = tiktoken.encoding_for_model("gpt-4o")
            except KeyError:
                encoding = tiktoken.get_encoding("cl100k_base")
                
            section_tokens = encoding.encode(section)
            
            # If section is small enough, keep it as one chunk
            if len(section_tokens) <= chunk_size:
                chunks.append({
                    "content": section,
                    "metadata": {
                        "section": section_title,
                        "section_idx": section_idx
                    }
                })
            else:
                # Split large sections into chunks with more overlap
                for i in range(0, len(section_tokens), chunk_size - overlap):
                    chunk_tokens = section_tokens[i:i + chunk_size]
                    chunk_text = encoding.decode(chunk_tokens)
                    
                    # Include section title in each chunk for context
                    if i > 0:
                        chunk_text = f"{section_title}:\n{chunk_text}"
                    
                    chunks.append({
                        "content": chunk_text,
                        "metadata": {
                            "section": section_title,
                            "section_idx": section_idx,
                            "chunk_idx": i // (chunk_size - overlap)
                        }
                    })
    
    return chunks

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF file"""
    pdf_reader = PyPDF2.PdfReader(BytesIO(file_bytes))
    text = ""
    for page in pdf_reader.pages:
        extracted_text = page.extract_text()
        if extracted_text:
            text += extracted_text + "\n\n"
    return text

def translate_query_if_needed(query: str, target_language: str = "ar") -> str:
    """Translate query to the document language if needed"""
    # Detect if query is not in Arabic
    is_arabic = any('\u0600' <= c <= '\u06FF' for c in query)
    
    if not is_arabic and target_language == "ar":
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a translator. Translate the following text to Arabic."},
                    {"role": "user", "content": query}
                ],
                temperature=0
            )
            translated_query = response.choices[0].message.content
            return translated_query
        except Exception as e:
            print(f"Translation error: {e}")
            return query
    return query

def translate_response_if_needed(response: str, query_language: str) -> str:
    """Translate response to query language if needed"""
    # Detect if response is in Arabic but query is not
    is_response_arabic = any('\u0600' <= c <= '\u06FF' for c in response)
    is_query_arabic = any('\u0600' <= c <= '\u06FF' for c in query_language)
    
    if is_response_arabic and not is_query_arabic:
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a translator. Translate the following Arabic text to English."},
                    {"role": "user", "content": response}
                ],
                temperature=0
            )
            translated_response = response.choices[0].message.content
            return translated_response
        except Exception as e:
            print(f"Translation error: {e}")
            return response
    return response

# Routes
@app.post("/upload-policy-document/")
async def upload_policy_document(
    file: UploadFile = File(...),
    metadata: str = Form(None)  # Make metadata optional
):
    """Upload and process the company policy PDF document"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        # Set default metadata if none provided
        if metadata is None:
            doc_metadata = {
                "title": file.filename,
                "description": "Company Policy Document",
                "language": "ar"
            }
        else:
            try:
                doc_metadata = json.loads(metadata)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid metadata JSON format")
        
        # Process PDF
        contents = await file.read()
        text = extract_text_from_pdf(contents)
        
        if not text:
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")
        
        # Create document in database
        document_data = {
            "title": doc_metadata.get("title", file.filename),
            "description": doc_metadata.get("description", "Company Policy Document"),
            "language": doc_metadata.get("language", "ar"),
            "full_text": text  # Store the full text for full_context approach
        }
        
        document_result = supabase_client.table("documents").insert(document_data).execute()
        document_id = document_result.data[0]["id"]
        
        # Chunk text and create embeddings
        chunks = chunk_text(text)
        
        # Add chunks to database
        for chunk in chunks:
            embedding = get_embedding(chunk["content"])
            
            chunk_data = {
                "document_id": document_id,
                "content": chunk["content"],
                "embedding": embedding,
                "metadata": chunk["metadata"]
            }
            
            supabase_client.table("chunks").insert(chunk_data).execute()
        
        return {"message": f"Successfully processed document with {len(chunks)} chunks", "document_id": document_id}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")

@app.post("/query/", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    """Query the company policy using either RAG or full-context approach"""
    start_time = time.time()
    original_query = request.query
    
    try:
        # Get all documents (should only be one in this case)
        documents_result = supabase_client.table("documents").select("*").execute()
        
        if not documents_result.data:
            raise HTTPException(status_code=404, detail="No policy document found in the database")
        
        document = documents_result.data[0]
        document_id = document["id"]
        document_language = document.get("language", "ar")
        
        # Translate query if needed
        query = translate_query_if_needed(request.query, document_language)
        
        if request.approach == "rag":
            # RAG approach
            # Get query embedding
            query_embedding = get_embedding(query)
            
            # Semantic search for relevant chunks with optimized parameters
            chunks_result = supabase_client.rpc(
                "match_policy_chunks",
                {
                    "query_embedding": query_embedding,
                    "match_threshold": 0.25,  # Lower threshold for better recall
                    "match_count": 10  # Increase number of chunks for better context
                }
            ).execute()
            
            if not chunks_result.data:
                response = "لم يتم العثور على معلومات ذات صلة في سياسة الشركة."
                if document_language != "ar":
                    response = "No relevant information found in the company policy."
                
                return {
                    "answer": translate_response_if_needed(response, original_query),
                    "approach": "rag", 
                    "processing_time": time.time() - start_time
                }
            
            # Format context from retrieved chunks with section titles
            contexts = []
            for chunk in chunks_result.data:
                section_title = chunk["metadata"].get("section", "")
                content = chunk["content"]
                if section_title and not content.startswith(section_title):
                    contexts.append(f"{section_title}:\n{content}")
                else:
                    contexts.append(content)
            
            formatted_context = "\n\n".join(contexts)
            
            # Enhanced system prompt for better Arabic responses and answer synthesis
            system_prompt = """أنت مساعد متخصص في تحليل وشرح سياسات وإجراءات الشركة. دورك هو:
1. قراءة محتوى السياسة بعناية
2. فهم سؤال المستخدم بدقة
3. دمج المعلومات من جميع المصادر ذات الصلة في إجابة واحدة متماسكة وشاملة
4. تقديم إجابة واضحة ودقيقة مبنية حصراً على المعلومات الموجودة في النص
5. تنظيم الإجابة بشكل منطقي ومتسلسل
6. إذا كان هناك تفاصيل مهمة في النص، قم بذكرها
7. إذا كانت المعلومات متناقضة أو غير كاملة، قم بتوضيح ذلك

يجب أن تكون إجابتك:
- دقيقة ومباشرة
- مدعومة بالمعلومات من النص فقط
- موحدة ومتماسكة (دمج المعلومات من جميع المصادر في إجابة واحدة)
- منظمة بشكل واضح
- باللغة العربية الفصحى السهلة"""

            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"السياق:\n{formatted_context}\n\nالسؤال: {query}\n\nملاحظة مهمة: قم بدمج جميع المعلومات ذات الصلة من النص في إجابة واحدة متماسكة وشاملة."}
                ],
                temperature=0
            )
            
            answer = response.choices[0].message.content
            
            # Translate response if needed
            answer = translate_response_if_needed(answer, original_query)
              # Format sources for citation
            sources = []
            for chunk in chunks_result.data:
                source = {
                    "content": chunk["content"],
                    "metadata": chunk["metadata"],
                    "relevance": chunk["similarity"]
                }
                sources.append(source)
            
            return {
                "answer": answer,
                "approach": "rag",
                "processing_time": time.time() - start_time,
                "sources": sources
            }
        elif request.approach == "full_context":
            # Full context approach - read from full_text column
            document_result = supabase_client.table("documents").select("full_text").eq("id", document_id).execute()
            
            if not document_result.data or not document_result.data[0].get("full_text"):
                response = "لم يتم العثور على محتوى لسياسة الشركة."
                if document_language != "ar":
                    response = "No content found for the company policy."
                
                return {
                    "answer": translate_response_if_needed(response, original_query),
                    "approach": "full_context", 
                    "processing_time": time.time() - start_time
                }
            
            # Get full text directly from documents table
            full_text = document_result.data[0]["full_text"]
            
            # Check token count and truncate if needed
            token_count = num_tokens_from_string(full_text)
            max_tokens = 120000  # GPT-4o context limit
            
            if token_count > max_tokens - 1000:  # Leave room for query and response
                try:
                    encoding = tiktoken.encoding_for_model("gpt-4o")
                except KeyError:
                    encoding = tiktoken.get_encoding("cl100k_base")
                    
                tokens = encoding.encode(full_text)
                full_text = encoding.decode(tokens[:max_tokens - 1000])
            
            # Generate response with GPT-4o
            system_prompt = """أنت مساعد ذكي مختص بسياسات الشركة وإجراءاتها. مهمتك هي الإجابة على الأسئلة المتعلقة بسياسة الشركة وإجراءاتها بناءً على المعلومات المقدمة فقط.
            
قم بالإجابة بشكل دقيق ومباشر على السؤال باستخدام المعلومات الموجودة في الوثيقة فقط. إذا لم تتوفر المعلومات الكافية للإجابة على السؤال، فيرجى الإشارة إلى ذلك."""
            
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"الوثيقة:\n{full_text}\n\nالسؤال: {query}"}
                ],
                temperature=0
            )
            
            answer = response.choices[0].message.content
            
            # Translate response if needed
            answer = translate_response_if_needed(answer, original_query)
            
            return {
                "answer": answer,
                "approach": "full_context",
                "processing_time": time.time() - start_time
            }
        else:
            raise HTTPException(status_code=400, detail="Approach must be either 'rag' or 'full_context'")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.get("/document-info/")
async def get_document_info():
    """Get information about the uploaded policy document"""
    try:
        document_result = supabase_client.table("documents").select("*").execute()
        
        if not document_result.data:
            return {"status": "No document uploaded yet"}
        
        document = document_result.data[0]
        
        # Get chunk count
        chunks_result = supabase_client.table("chunks").select("id").eq("document_id", document["id"]).execute()
        
        return {
            "document": document,
            "chunk_count": len(chunks_result.data) if chunks_result.data else 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get document info: {str(e)}")

@app.delete("/reset-database/")
async def reset_database():
    """Reset the database by removing all documents and chunks"""
    try:
        # Delete all chunks first (due to foreign key constraints)
        supabase_client.table("chunks").delete().neq('id', 0).execute()
        
        # Delete all documents
        supabase_client.table("documents").delete().neq('id', 0).execute()
        
        return {"status": "Database reset successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reset database: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": time.time()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
