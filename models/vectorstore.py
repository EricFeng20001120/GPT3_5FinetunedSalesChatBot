from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import TextLoader, PyPDFLoader
# from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import FAISS

from langchain.retrievers import BM25Retriever, EnsembleRetriever

from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.document import Document
import uuid

import status

# use summary multivector retriever, if false, use smaller chunks multivector retriever
use_summary = status.multivector_retriever_use_summary

class my_vectors_store():
    def __init__(self, databases):
        self.databases = databases
        self.documents = []
        self.ald_documents = []
        # get embeddings
        # self.embeddings = OpenAIEmbeddings(model_name="ada")
        # model_name = "BAAI/bge-large-en-v1.5"
        # encode_kwargs = {'normalize_embeddings': True}
        # self.embeddings = HuggingFaceBgeEmbeddings(
        #     model_name=model_name,
        #     encode_kwargs=encode_kwargs
        # )
        self.embeddings = HuggingFaceBgeEmbeddings() # using huggingface embedding which is free
        self.vectorstore = None
        self.get_always_load_documents()
        for database in databases:
            self.get_database_documents(database)
        self.update_vectorstore()
    
    # when documents are ready, call this function to update the vectorstore
    def update_vectorstore(self):
        if len(self.documents) > 0:
            if use_summary: # uses summary MultiVectorRetriever - process the documents by generating their summaries by gpt-3.5
                # split the text
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000)
                text = text_splitter.split_documents(self.documents)
                
                chain = (
                    {"doc": lambda x: x.page_content}
                    | ChatPromptTemplate.from_template("Summarize the following document:\n\n{doc}")
                    | ChatOpenAI(model="gpt-3.5-turbo", max_retries=0)
                    | StrOutputParser()
                )
                summaries = chain.batch(text, {"max_concurrency": 5})
                
                id_key = "doc_id"
                doc_ids = [str(uuid.uuid4()) for _ in text]
                summary_docs = [
                    Document(page_content=s, metadata={id_key: doc_ids[i]})
                    for i, s in enumerate(summaries)
                ]
                
                # storeages
                self.vectorstore = FAISS.from_documents(summary_docs, self.embeddings)
                self.store = InMemoryStore()
                self.store.mset(list(zip(doc_ids, text)))
                
                # The retriever
                multivector_retriever = MultiVectorRetriever(
                    vectorstore=self.vectorstore,
                    docstore=self.store,
                    id_key=id_key,
                )
            else: # uses smaller chunks MultiVectorRetriever
                # split the text
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
                text = text_splitter.split_documents(self.documents)
                
                # split the text into smaller chunks
                child_text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
                id_key = "doc_id"
                doc_ids = [str(uuid.uuid4()) for _ in text]
                sub_docs = []
                for i, doc in enumerate(text):
                    _id = doc_ids[i]
                    _sub_docs = child_text_splitter.split_documents([doc])
                    for _doc in _sub_docs:
                        _doc.metadata[id_key] = _id
                    sub_docs.extend(_sub_docs)
                
                # storeages
                self.vectorstore = FAISS.from_documents(sub_docs, self.embeddings)
                self.store = InMemoryStore()
                self.store.mset(list(zip(doc_ids, text)))
                
                # the retriever
                multivector_retriever = MultiVectorRetriever(
                    vectorstore=self.vectorstore,
                    docstore=self.store,
                    id_key=id_key,
                )
            
            # Ensemble Retriever - combines a sparse retriever (BM25) with multivector retriever
            bm25_retriever = BM25Retriever.from_documents(text)
            bm25_retriever.k = 4
            multivector_retriever = multivector_retriever
            ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, multivector_retriever], weights=[0.4, 0.6]
            )
            self.retriever = ensemble_retriever
        else:
            self.vectorstore = None
            self.store = None # store layer, needed for multivector retriever
            self.retriever = None
    
    # this function will place the always_load documents into self.ald_documents
    # this file should be small and having high informamtion dencity
    def get_always_load_documents(self):
        text_loader = TextLoader(status.resource_dir + "ald.txt")
        documents = text_loader.load()
        self.ald_documents = self.ald_documents + documents
    
    # this function will parce the target database files into the document
    def get_database_documents(self, database):
        # load pdf files
        pdf_loader = DirectoryLoader(status.resource_dir + database + '/', glob="*.pdf", loader_cls=PyPDFLoader)
        # try to load all files, do not raise error when error encountered
        text_loader = DirectoryLoader(status.resource_dir + database + '/', glob="**/[!.]*", loader_cls=TextLoader, loader_kwargs={'autodetect_encoding': True}, silent_errors=True)
        # load csv files
        documents = pdf_loader.load() + text_loader.load()
        self.documents = self.documents + documents
    
    # this function will remove all files in documents that are not in the current database list
    def remove_unneeded_database_documents(self, new_database):
        temp_documents = self.documents.copy()
        for document in temp_documents:
            # if this document does not exist in the new selected database, remove it
            if document.metadata['source'].split('\\')[-2] not in new_database:
                self.documents.remove(document)
    
    # this function will remove all files for a selected database
    def remove_datase_documents(self, database):
        temp_documents = self.documents.copy()
        for document in temp_documents:
            # if this document does not exist in the new selected database, remove it
            if document.metadata['source'].split('\\')[-2] == database:
                self.documents.remove(document)
    
    def get_vectorstore(self):
        return self.vectorstore
    
    def get_documents(self):
        return self.documents
        
    def get_ald_documents(self):
        return self.ald_documents
    
    def get_retriever(self):
        return self.retriever