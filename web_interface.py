import streamlit as st
import pandas as pd
from data_analyst_optimized import DataAnalystAssistant
import os

st.set_page_config(page_title="Data Analyst Assistant", layout="wide")

# Initialize session state
if 'assistant' not in st.session_state:
    st.session_state.assistant = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

st.title("🤖 LLM-Based Data Analyst Assistant")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # Model selection
    model_choice = st.selectbox("Choose LLM:", ["Ollama (Llama3)", "Gemini API"])
    
    if model_choice == "Ollama (Llama3)":
        api_input = st.text_input("Ollama URL", value="http://localhost:11434")
    else:
        api_input = st.text_input("Gemini API Key", type="password")
    
    if st.button("Initialize Assistant") and not st.session_state.assistant:
        try:
            if model_choice == "Gemini API":
                # You'll need to modify DataAnalystAssistant constructor
                st.session_state.assistant = DataAnalystAssistant(api_input, use_gemini=True)
            else:
                st.session_state.assistant = DataAnalystAssistant(api_input)
            st.success(f"Assistant initialized with {model_choice}!")
        except Exception as e:
            st.error(f"Failed to initialize: {str(e)}")
    
    st.header("Upload Data")
    
    # File upload
    uploaded_files = st.file_uploader(
        "Upload Excel/CSV files", 
        type=['xlsx', 'xls', 'csv'], 
        accept_multiple_files=True
    )
    
    if uploaded_files and st.session_state.assistant:
        for file in uploaded_files:
            temp_path = f"temp_{file.name}"
            with open(temp_path, "wb") as f:
                f.write(file.getbuffer())
            
            table_name = st.text_input(f"Table name for {file.name}", 
                                     value=os.path.splitext(file.name)[0])
            
            if st.button(f"Load {file.name}"):
                try:
                    result = st.session_state.assistant.load_file(temp_path, table_name)
                    st.success(result)
                    os.remove(temp_path)
                except Exception as e:
                    st.error(f"Error loading file: {str(e)}")

# Main interface (rest remains the same)
if st.session_state.assistant:
    if st.session_state.assistant.tables:
        st.subheader("Available Tables")
        for table_name, info in st.session_state.assistant.tables.items():
            with st.expander(f"Table: {table_name}"):
                st.write(f"Columns: {', '.join(info['columns'])}")
                st.write("Sample data:")
                st.json(info['sample_data'][:2])
    
    st.subheader("Ask Questions About Your Data")
    
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat['question'])
        
        with st.chat_message("assistant"):
            if chat['success']:
                st.write(chat['insights'])
                with st.expander("View SQL Query & Results"):
                    st.code(chat['sql_query'], language='sql')
                    if chat['results']:
                        st.dataframe(pd.DataFrame(chat['results']))
            else:
                st.error(chat['error'])
    
    question = st.chat_input("Ask a question about your data...")
    
    if question:
        with st.chat_message("user"):
            st.write(question)
        
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                result = st.session_state.assistant.analyze(question)
                st.session_state.chat_history.append(result)
                
                if result['success']:
                    st.write(result['insights'])
                    with st.expander("View SQL Query & Results"):
                        st.code(result['sql_query'], language='sql')
                        if result['results']:
                            st.dataframe(pd.DataFrame(result['results']))
                else:
                    st.error(result['error'])

else:
    st.warning("Please initialize the assistant in the sidebar with your preferred LLM.")
