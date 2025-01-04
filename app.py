import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()
api_key = os.getenv("API_KEY")

# Load model from Groq API
def load_model():
    llm = ChatGroq(
        temperature=0, 
        groq_api_key=api_key, 
        model_name="llama-3.1-70b-versatile"
    )
    return llm

# Define LangChain prompts
def get_code_explanation_prompt():
    template = """
    Explain the following code in simple terms suitable for a beginner:

    Code:
    {code}

    Explanation:
    """
    return PromptTemplate(template=template, input_variables=["code"])

def get_time_space_complexity_prompt():
    template = """
    Analyze the time and space complexity of the following code. Provide a Big-O notation for the code's time and space complexity:

    Code:
    {code}

    Time Complexity:
    """
    return PromptTemplate(template=template, input_variables=["code"])

def get_code_feedback_prompt():
    template = """
    Review the following code and provide constructive feedback by acting as a professional code reviewer, including suggestions for improvement, efficiency, and best practices:

    Code:
    {code}

    Code Review Feedback:
    """
    return PromptTemplate(template=template, input_variables=["code"])

# Streamlit UI
def main():
    st.title("Code Insight App")
    st.write("Enter your code below to get a beginner-friendly explanation, time and space complexity analysis, and a code review!")

    # Initialize session state for code input if not already initialized
    if "code_input" not in st.session_state:
        st.session_state.code_input = ""

    # Input area for the code
    code_input = st.text_area("Paste your code here:", height=200, value=st.session_state.code_input)

    # Buttons for each functionality (placed in the same row)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        explain_button = st.button("Explain Code")
    with col2:
        complexity_button = st.button("Analyze Complexity")
    with col3:
        feedback_button = st.button("Code Review")
    with col4:
        clear_button = st.button("Clear Text")

    if clear_button:
        # Clear the text area and reset session state
        st.session_state.code_input = ""  # Clear the session state value

    if explain_button or complexity_button or feedback_button:
        if not code_input.strip():
            st.warning("Please enter some code for analysis.")
        else:
            with st.spinner("Generating results..."):
                chatgroq = load_model()

                # Code explanation
                if explain_button:
                    explanation_prompt = get_code_explanation_prompt()
                    explanation_chain = LLMChain(llm=chatgroq, prompt=explanation_prompt)
                    explanation = explanation_chain.run({"code": code_input})
                    st.subheader("Explanation:")
                    st.write(explanation)

                # Time complexity analysis
                if complexity_button:
                    time_complexity_prompt = get_time_space_complexity_prompt()
                    time_complexity_chain = LLMChain(llm=chatgroq, prompt=time_complexity_prompt)
                    time_complexity = time_complexity_chain.run({"code": code_input})
                    st.subheader("Time and Space Complexity Analysis:")
                    st.write(time_complexity)

                # Code review/feedback
                if feedback_button:
                    feedback_prompt = get_code_feedback_prompt()
                    feedback_chain = LLMChain(llm=chatgroq, prompt=feedback_prompt)
                    code_feedback = feedback_chain.run({"code": code_input})
                    st.subheader("Code Review Feedback:")
                    st.write(code_feedback)

# Entry point for the Streamlit app
if __name__ == "__main__":
    main()
