"""
Module for initializing and managing a Chroma vector store for restaurant reviews.

This module loads restaurant reviews from a CSV file, generates embeddings using
OllamaEmbeddings, and creates a Chroma vector store for efficient retrieval. It
supports one-time document addition to avoid redundant processing and provides a
configured retriever for querying relevant reviews.

@Author: Md Rezwanul Haque
"""

import os
import pandas as pd
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from typing import List, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class VectorStoreManager:
    """Manages the creation and retrieval of a Chroma vector store for restaurant reviews."""
    
    def __init__(
        self,
        csv_path: str = "realistic_restaurant_reviews.csv",
        db_location: str = "./chrome_langchain_db",
        collection_name: str = "restaurant_reviews",
        embedding_model: str = "mxbai-embed-large",
        k: int = 5
    ):
        """
        Initialize the VectorStoreManager.

        Args:
            csv_path (str): Path to the CSV file containing restaurant reviews.
            db_location (str): Directory to persist the Chroma database.
            collection_name (str): Name of the Chroma collection.
            embedding_model (str): Name of the embedding model for OllamaEmbeddings.
            k (int): Number of documents to retrieve in search results.
        """
        self.csv_path = csv_path
        self.db_location = db_location
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.k = k
        self.vector_store = None
        self.retriever = None
        
    def load_reviews(self) -> pd.DataFrame:
        """
        Load restaurant reviews from a CSV file.

        Returns:
            pd.DataFrame: DataFrame containing the reviews.

        Raises:
            FileNotFoundError: If the CSV file is not found.
            pd.errors.EmptyDataError: If the CSV file is empty.
        """
        try:
            logger.info(f"Loading reviews from {self.csv_path}")
            df = pd.read_csv(self.csv_path)
            if df.empty:
                raise pd.errors.EmptyDataError("CSV file is empty")
            return df
        except FileNotFoundError:
            logger.error(f"CSV file not found at {self.csv_path}")
            raise
        except pd.errors.EmptyDataError:
            logger.error("CSV file is empty")
            raise
        except Exception as e:
            logger.error(f"Error loading CSV: {str(e)}")
            raise

    def create_documents(self, df: pd.DataFrame) -> Tuple[List[Document], List[str]]:
        """
        Create LangChain Document objects from the DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing review data.

        Returns:
            Tuple[List[Document], List[str]]: List of Document objects and their IDs.
        """
        logger.info("Creating Document objects from reviews")
        documents = []
        ids = []
        
        for i, row in df.iterrows():
            try:
                # Validate required columns
                if not all(col in row for col in ["Title", "Review", "Rating", "Date"]):
                    logger.warning(f"Skipping row {i}: Missing required columns")
                    continue
                
                document = Document(
                    page_content=f"{row['Title']} {row['Review']}",
                    metadata={"rating": row["Rating"], "date": row["Date"]},
                    id=str(i)
                )
                documents.append(document)
                ids.append(str(i))
            except Exception as e:
                logger.warning(f"Error processing row {i}: {str(e)}")
                continue
        
        if not documents:
            logger.error("No valid documents created from CSV")
            raise ValueError("No valid documents could be created from the CSV")
        
        logger.info(f"Created {len(documents)} documents")
        return documents, ids

    def initialize_vector_store(self) -> None:
        """
        Initialize the Chroma vector store and retriever.

        Checks if the vector store exists; if not, loads reviews, creates documents,
        and adds them to the store. Configures the retriever for querying.

        Raises:
            Exception: If vector store initialization fails.
        """
        try:
            # Initialize embeddings
            logger.info(f"Initializing embeddings with model {self.embedding_model}")
            embeddings = OllamaEmbeddings(model=self.embedding_model)
            
            # Initialize Chroma vector store
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                persist_directory=self.db_location,
                embedding_function=embeddings
            )
            
            # Check if documents need to be added
            add_documents = not os.path.exists(self.db_location)
            if add_documents:
                logger.info("Vector store does not exist, creating and populating")
                df = self.load_reviews()
                documents, ids = self.create_documents(df)
                
                logger.info(f"Adding {len(documents)} documents to vector store")
                self.vector_store.add_documents(documents=documents, ids=ids)
                logger.info("Documents successfully added to vector store")
            
            # Configure retriever
            self.retriever = self.vector_store.as_retriever(
                search_kwargs={"k": self.k}
            )
            logger.info(f"Retriever initialized with k={self.k}")
            
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            raise

    def get_retriever(self):
        """
        Get the configured retriever.

        Returns:
            The retriever object for querying the vector store.

        Raises:
            ValueError: If the retriever is not initialized.
        """
        if self.retriever is None:
            raise ValueError("Retriever not initialized. Call initialize_vector_store first.")
        return self.retriever

def main():
    """
    Main function to initialize the vector store and retriever.

    This function is executed when the module is run directly and serves as an
    entry point for testing or standalone execution.

    @Author: Md Rezwanul Haque
    """
    try:
        manager = VectorStoreManager()
        manager.initialize_vector_store()
        logger.info("Vector store and retriever initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize vector store: {str(e)}")
        raise

if __name__ == "__main__":
    main()
