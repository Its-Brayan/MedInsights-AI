import os
from dotenv import load_dotenv
from vectordb import VectorDB
from typing import List
from langchain_community.document_loaders import PyMuPDFLoader,DirectoryLoader
from config_loader import load_config,load_prompt
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
config = load_config()
prompt_config = load_prompt()
load_dotenv()
def load_documents() -> List[str]:
   """
    Load documents for demonstration.

    Returns:
        List of sample documents
    """
   try:
      documents = []
      loader = DirectoryLoader(
         "/home/brayan/Aiprojects/MedInsights-AI/data",
         loader_cls=PyMuPDFLoader
      )
      loaded_docs = loader.load()
      print(f"Sucessfully loaded f{loaded_docs}")
      documents.extend(loaded_docs)
   except Exception as e:
      print(f"Error loading {str(e)}")
   publications = []
   for doc in documents:
      publications.append(doc)
   return publications

class RAGAssistant:
    """
    A simple RAG-based AI assistant using ChromaDB and multiple LLM providers.
    Supports OpenAI, Groq, and Google Gemini APIs.
    """

    def __init__(self):
       self.llm = self._initialize_llm()
       if not self.llm:
          raise ValueError(
              "No valid API key found. Please set one Gemini API Key "
          )
       
       #initialize vector database
       self.vector_db = VectorDB()
       self.qa_prompt = prompt_config['qa_prompt']
       self.prompt_template = ChatPromptTemplate.from_template(
        template="""
       
       {qa_prompt}

        context:
        {context}

        Question:
        {question}
        """
       )
       
       #create the chain
       self.chain = self.prompt_template | self.llm | StrOutputParser()
    def _initialize_llm(self):
         """Initialize the LLM by checking for available API keys"""
         if os.getenv("GOOGLE_API_KEY"):
            model_name = config['llm']
            print(f"Using Ai model {model_name}")
            return ChatGoogleGenerativeAI(
               api_key = os.getenv("GOOGLE_API_KEY"),
               model = model_name,
               temperature = 0.0
            )
         
    
    def add_documents(self,documents:List) -> None:
        """
        Add documents to the knowledge base.

        Args:
            documents: List of documents
        """
        self.vector_db.insert_documents(documents)
    
    def invoke(self,input:str,n_results:int = 3) -> str:
        """
        Query the RAG assistant.

        Args:
            input: User's input
            n_results: Number of relevant chunks to retrieve

        Returns:
            Dictionary containing the answer and retrieved context
        """
        #Retrieve relevant documents from the vector database
        docs = self.vector_db.search(input,top_k=n_results)
         #combine retrieved chunks into a context string
        context = "\n\n".join(doc['content'] for doc in docs)

        #Generate responses using the RAG chain
        response = self.chain.invoke({
           "context":context,
           "question":input,
           "qa_prompt":self.qa_prompt
        })

        llm_answer = response.content if hasattr(response,"content") else response

        return llm_answer

def main():
   """Main function to demonstrate the RAG assistant"""
   try:
      #initialize RAG assistant
      print("Initializing RAG assistant")
      assistant = RAGAssistant()


      #load sample documents
      sample_docs = load_documents()
      print(f"Loaded {len(sample_docs)} sample documents")

      assistant.add_documents(sample_docs)

      done = False

      while not done:
         question = input("Enter a question or 'quit' to exit: ")
         if question.lower() == 'quit':
            done = True
         else:
            result = assistant.invoke(question)
            print(result)
   except Exception as e:
        print(f"Error running RAG assistant: {e}")
        print("Make sure you have set up your .env file with at least one API key:")
        print("- GOOGLE_API_KEY (Google Gemini models)")

if __name__ == '__main__':
   main()
