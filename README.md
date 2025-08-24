# LLM-Based Data Analyst Assistant

An AI assistant that analyzes Excel datasets using natural language queries, converts them to SQL, and provides insights.

## Features

- **Excel File Processing**: Load multiple Excel files into in-memory SQLite database
- **Natural Language to SQL**: Convert questions like "What were top selling products?" to SQL queries
- **Query Execution**: Run SQL queries on your data
- **Natural Language Insights**: Get human-readable explanations of results
- **Web Interface**: Easy-to-use Streamlit interface

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Get OpenAI API Key**:
   - Sign up at https://platform.openai.com/
   - Create an API key

3. **Run Web Interface**:
   ```bash
   streamlit run web_interface.py
   ```

4. **Or use programmatically**:
   ```python
   from data_analyst import DataAnalystAssistant
   
   assistant = DataAnalystAssistant("your-api-key")
   assistant.load_excel("data.xlsx", "sales")
   result = assistant.analyze("What were top products by revenue?")
   print(result['insights'])
   ```

## Example Questions

- "What were last quarter's top-selling products?"
- "Which region has the highest revenue?"
- "Show me products with sales over $1000"
- "What's the average revenue per product?"

## Architecture

1. **Excel → SQLite**: Load Excel files into in-memory database
2. **NL → SQL**: Use GPT to convert questions to SQL queries
3. **Execute**: Run queries on the database
4. **SQL Results → NL**: Generate human-readable insights

## Security Features

- In-memory database (no persistent storage)
- API key input validation
- Query sanitization through LLM
- Temporary file cleanup