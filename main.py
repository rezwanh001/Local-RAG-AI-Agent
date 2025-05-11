"""
@Author: Md Rezwanul Haque
"""
import streamlit as st
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import VectorStoreManager
import time

# Initialize the model
try:
    model = OllamaLLM(model="llama3.2")
except Exception as e:
    st.error(f"Error initializing model: {str(e)}")
    st.stop()

# Initialize the vector store manager
# Export the retriever for use in other modules
manager = VectorStoreManager()
manager.initialize_vector_store()
retriever = manager.get_retriever()

# Define the prompt template
template = """
You are an expert in answering questions about a pizza restaurant.

Here are some relevant reviews: {reviews}

Here is the question to answer: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def process_question(question, max_reviews=5):
    """Process a user question and return the answer with relevant reviews."""
    start_time = time.time()
    
    try:
        # Retrieve relevant reviews
        reviews = retriever.invoke(question)
        if not reviews:
            return "No relevant reviews found.", [], 0
        
        # Limit the number of reviews to avoid overwhelming the model
        reviews = reviews[:max_reviews]
        
        # Invoke the chain to get the answer
        result = chain.invoke({"reviews": reviews, "question": question})
        
        processing_time = time.time() - start_time
        return result, reviews, processing_time
    except Exception as e:
        return f"Error processing question: {str(e)}", [], 0

def main():
    st.set_page_config(page_title="Pizza Restaurant Q&A", layout="wide")
    
    st.title("🍕 Pizza Restaurant Q&A Reviews")
    st.write("Ask questions about pizza restaurants and get answers based on reviews!")
    
    # Create sidebar for input options
    with st.sidebar:
        st.header("Query Options")
        
        # Allow users to specify the maximum number of reviews to use
        max_reviews = st.slider(
            "Maximum number of reviews to use",
            min_value=1,
            max_value=10,
            value=5,
            help="Limits the number of reviews used to answer the question."
        )
        
        # Example questions for quick selection
        example_questions = [
            "What is the best pizza place in town?",
            "Which pizza restaurant has the best pepperoni pizza?",
            "Are there any family-friendly pizza places nearby?",
            "What do customers say about the service at local pizzerias?",
            ""
        ]
        selected_question = st.selectbox(
            "Choose an example question or type your own",
            example_questions,
            index=len(example_questions)-1,  # Default to empty
            help="Select an example or enter a custom question below."
        )
    
    # Main content area
    st.subheader("Ask Your Question")
    
    # Input field for custom question, pre-filled with selected example if any
    user_question = st.text_input(
        "Enter your question about pizza restaurants:",
        value=selected_question,
        placeholder="E.g., What is the best pizza place in town?"
    )
    
    # Process button
    if st.button("Get Answer"):
        if not user_question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Processing your question..."):
                answer, reviews, processing_time = process_question(user_question, max_reviews)
                
                # Display results
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("Answer")
                    st.markdown(answer)
                
                with col2:
                    st.subheader("Relevant Reviews")
                    if reviews:
                        for i, review in enumerate(reviews, 1):
                            with st.expander(f"Review {i}"):
                                st.write(review)
                    else:
                        st.write("No reviews available.")
                
                st.success(f"Question processed in {processing_time:.2f} seconds")
    
    # Information section
    with st.expander("About Pizza Restaurant Q&A"):
        st.write("""
        This app uses a language model (LLaMA 3.2 via LangChain) to answer questions about pizza restaurants 
        based on retrieved customer reviews. The system retrieves relevant reviews and generates a response 
        tailored to your question.
        
        Features:
        - Ask any question about pizza restaurants.
        - View relevant customer reviews used to generate the answer.
        - Customize the number of reviews used for context.
        - Select from example questions for quick queries.
        
        Note: Ensure the `retriever` module is properly configured to fetch reviews.
        """)

if __name__ == "__main__":
    main()