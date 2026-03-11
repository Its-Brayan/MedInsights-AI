import os
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List,Dict,Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
import torch
from dotenv import load_dotenv
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
load_dotenv()
class VectorDB:
    """
    A simple vector database wrapped using chromaDB with HuggingFace embedding
    """
    def __init__(self,collection_name:str = None,embedding_model:str = None):
            """
            Initialize the vector database.

            Args:
                collection_name: Name of the ChromaDB collection
                embedding_model: HuggingFace model name for embeddings
            """
            self.collection_name = collection_name or os.getenv(
                  "COLLECTION_NAME"
            )
            self.embedding_model_name = embedding_model or os.getenv(
                  "EMBEDDING_MODEL"
            )
            #Initialize ChromaDB client
            self.client = chromadb.PersistentClient('./chroma_db')

            #load embedding model
            print(f"Loading embedding model:{self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            self.collection = self.client.get_or_create_collection(
                  name=self.collection_name,
                  metadata={"description":"Medical document collection"}
            )
            print(f"Vector database initialized with collection:{self.collection_name}")
    
    def chunk_text(self,publication:str,chunk_size:int=10000,chunk_overlap:int = 400)->list[str]:
          text_splitter = RecursiveCharacterTextSplitter(
                chunk_size = chunk_size,
                chunk_overlap = chunk_overlap    
          )
          return text_splitter.split_text(publication)
    
    def insert_documents(self,documents:List) -> None:
            """
            Add documents to the vector database.

            Args:
                documents: List of documents
            """
            #embed documents using a model
            device = (
                   "cuda"
                   if torch.cuda.is_available()
                   else "mps" if torch.backends.mps.is_available() else "cpu"
            )
            model = HuggingFaceEmbeddings(
                   model_name = os.getenv("EMBEDDING_MODEL"),
                   model_kwargs={"device":device}
            )
           
            next_id = self.collection.count()
            for publication in documents:
                   content = getattr(publication,"page_content",str(publication))
                   if not content.strip():
                          continue
                   chunked_publications = self.chunk_text(content)
                   print(f"Chunks created: {len(chunked_publications)}")
                   embeddings = model.embed_documents(chunked_publications)
                   print("Embeddings created")
                   ids = list(range(next_id,next_id + len(chunked_publications)))
                   ids = [f"document_{id}" for id in ids]
                   self.collection.add(
                          embeddings=embeddings,
                          ids = ids,
                          documents=chunked_publications
                   )
                   next_id += len(chunked_publications)
    
    def search(self,query:str,top_k=5):
           """Find the most relevant research chunks for a query"""
           #convert query to vector
           query_vector = self.embedding_model.encode(query).tolist()

           #search for similar vector
           result = self.collection.query(
                  query_embeddings = [query_vector],
                  n_results = top_k,
                  include = ["documents","metadatas","distances"]
           )

           #format results
           relevant_chunks = []
           for i, doc in enumerate(result['documents'][0]):
                   relevant_chunks.append({
                    "content": doc,
                    "title": result["metadatas"][0][i].get("title", "No title") if result["metadatas"][0][i] else "No title",
                    "similarity": 1 - result["distances"][0][i]  # Convert distance to similarity
                })
           return relevant_chunks
                

