"""
@Author: Md Rezwanul Haque
"""
import streamlit as st
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import VectorStoreManager
import time
from tools import web_search
from langchain.agents import create_react_agent, AgentExecutor

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

# Create a custom prompt
template = """
You are an expert on pizza restaurants. First, try to answer the question using the provided reviews. If the reviews don't contain the answer, you may use the available tools.

Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Here is the chat history: {chat_history}

Here are some relevant reviews: {reviews}

Question: {input}
Thought:{agent_scratchpad}
"""
prompt = ChatPromptTemplate.from_template(template)

# Create the agent
tools = [web_search]
agent = create_react_agent(
    llm=model,
    tools=tools,
    prompt=prompt,
)

# Create an agent executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)

def process_question(question, history, max_reviews=5):
    """Process a user question and return the answer with relevant reviews."""
    start_time = time.time()
    
    try:
        # Retrieve relevant reviews
        reviews = retriever.invoke(question)
        if not reviews:
            reviews = []
        
        # Limit the number of reviews to avoid overwhelming the model
        reviews = reviews[:max_reviews]
        
        # Invoke the agent to get the answer
        result = agent_executor.invoke({
            "input": question,
            "chat_history": history,
            "reviews": reviews
        })
        
        processing_time = time.time() - start_time
        return result["output"], reviews, processing_time
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
    
    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Process button
    if st.button("Get Answer"):
        if not user_question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Processing your question..."):
                # Append user question to history
                st.session_state.chat_history.append({"role": "user", "content": user_question})
                
                answer, reviews, processing_time = process_question(user_question, st.session_state.chat_history, max_reviews)
                
                # Append assistant response to history
                st.session_state.chat_history.append({"role": "assistant", "content": answer, "reviews": reviews, "processing_time": processing_time})

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                st.markdown(message["content"])
                if "reviews" in message and message["reviews"]:
                    with st.expander("Relevant Reviews"):
                        for i, review in enumerate(message["reviews"], 1):
                            st.write(f"**Review {i}:** {review.page_content}")
                            st.write(f"**Rating:** {review.metadata['rating']}")
                            st.write(f"**Date:** {review.metadata['date']}")
                if "processing_time" in message:
                    st.success(f"Question processed in {message['processing_time']:.2f} seconds")
    
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

    # Add a new review
    with st.sidebar:
        st.header("Add a New Review")
        with st.form("new_review_form"):
            review_title = st.text_input("Review Title")
            review_text = st.text_area("Review Text")
            review_rating = st.slider("Rating", 1, 5, 5)
            review_date = st.date_input("Date")

            submitted = st.form_submit_button("Submit Review")
            if submitted:
                if review_title and review_text:
                    new_review = {
                        "Title": review_title,
                        "Review": review_text,
                        "Rating": review_rating,
                        "Date": review_date.strftime("%Y-%m-%d")
                    }
                    try:
                        manager.add_reviews([new_review])
                        st.success("New review added successfully!")
                    except Exception as e:
                        st.error(f"Error adding review: {str(e)}")
                else:
                    st.warning("Please fill out all fields.")

if __name__ == "__main__":
    main()