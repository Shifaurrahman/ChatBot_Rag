import os
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS

# Load your OpenAI API key from environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")  # Set this in your environment

# Load the FAISS index
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
db = FAISS.load_local("./faiss_index", embeddings, allow_dangerous_deserialization=True)

# Define the Streamlit UI
st.title("SriLankan Airlines Annual Report 2023/24")

question = st.text_input("What specific information do you need about SriLankan Airlines?")

if st.button("Submit"):
    if question:
        # Define your modified prompt template
        prompt_template = """
        You are an assistant tasked with summarizing tables and text for retrieval. 
        These summaries will be embedded and used to retrieve the raw text or table elements. 
        Give a concise summary of the table or text that is well optimized for retrieval. 
        Table or text: {element} 
        Just return the helpful answer in as much detail as possible.
        Answer:
        """
        
        qa_chain = LLMChain(
            llm=ChatOpenAI(model="gpt-4", openai_api_key=openai_api_key, max_tokens=1024),
            prompt=PromptTemplate.from_template(prompt_template)
        )

        relevant_docs = db.similarity_search(question)
        context = ""
        for d in relevant_docs:
            context += f"[{d.metadata['type']}] {d.metadata['original_content']}\n"

        result = qa_chain.run({'element': context})  # Updated to use 'element' instead of 'context'
        st.write(result)
    else:
        st.write("Please enter a question.")
