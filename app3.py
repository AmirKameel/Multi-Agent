import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load API key from .env file
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

# Configure generative model
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-pro')
chat = model.start_chat(history=[])

def main():
    st.title('AI multi-agent and self-ask')

    # Description of the app
    st.write("Welcome to the AI multi-agent and self-ask app. This app utilizes multiple AI agents to assist you with your queries.")
    st.write("🤖 Research agent: This agent employs self-asking, a questioning technique where you ask yourself a series of questions to help understand a topic or solve a problem. It involves breaking down a complex question into smaller, more manageable questions, eventually leading to answering the main question.")
    st.write("🤖 Code agent: Based on the final answer provided by the research agent, this agent generates the code necessary to draw a chart.")

    instruction = "You are An AI researcher that can do any complex tasks by using self asking method. Use self asking to answer any question, and show the intermediate questions and answers. Put the final answer after (Final answer) section and so on with any user questions."
    input_key = 0  # Initialize key for text input widgets

    while True:
        question = st.text_input(f"You{input_key}:")
        input_key += 1  # Increment key for each text input widget
        if question.strip() == '':
            break

        response = chat.send_message(instruction + question)

        # Check if the response contains the final answer
        if "Final answer:" in response.text:
            # Extract the final answer from the response
            final_answer_index = response.text.find("Final answer:")
            final_answer = response.text[final_answer_index + len("Final answer:"):].strip()
            # Display the final answer as a list
            st.write("Final Answer:")
            st.write(final_answer.split('\n'))
        else:
            st.write(f"AI researcher: {response.text}")

        instruction = ''

if __name__ == "__main__":
    main()
