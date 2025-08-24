import pandas as pd
import sqlite3
import requests
from typing import Dict, Any
import os
import json

class DataAnalystAssistant:
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url
        self.db_connection = sqlite3.connect(':memory:', check_same_thread=False)
        self.tables = {}
        
    def load_file(self, file_path: str, table_name: str = None) -> str:
        """Load Excel or CSV file into SQLite database"""
        if not table_name:
            table_name = os.path.splitext(os.path.basename(file_path))[0]
        
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
            
        df.to_sql(table_name, self.db_connection, if_exists='replace', index=False)
        
        # Store table schema
        self.tables[table_name] = {
            'columns': list(df.columns),
            'sample_data': df.head(3).to_dict('records')
        }
        
        return f"Loaded {len(df)} rows into table '{table_name}'"
    
    def load_excel(self, file_path: str, table_name: str = None) -> str:
        """Load Excel file - kept for backward compatibility"""
        return self.load_file(file_path, table_name)
    
    def get_schema_context(self) -> str:
        """Generate schema context for LLM"""
        context = "Available tables and their schemas:\n"
        for table_name, info in self.tables.items():
            context += f"\nTable: {table_name}\n"
            context += f"Columns: {', '.join(info['columns'])}\n"
            context += f"Sample data: {json.dumps(info['sample_data'][:2], indent=2)}\n"
        return context
    
    def nl_to_sql(self, question: str) -> str:
        """Convert natural language question to SQL query"""
        schema = self.get_schema_context()
        table_names = list(self.tables.keys())
        
        if not table_names:
            raise Exception("No tables loaded. Please upload a file first.")
        
        prompt = f"""{schema}

TASK: Convert question to SQL using ONLY the tables above.

QUESTION: {question}

STRICT RULES:
1. Use ONLY these exact table names: {table_names}
2. Use ONLY column names from the schema above
3. Return ONLY the SQL query, no explanations
4. If unsure, use SELECT * FROM {table_names[0]}

SQL:"""
        
        # Try SQLCoder first, fallback to Llama3
        for model in ["sqlcoder", "llama3"]:
            try:
                response = requests.post(f"{self.ollama_url}/api/generate", json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False
                })
                
                if response.status_code == 200:
                    result = response.json()
                    if result and "response" in result and result["response"]:
                        sql = result["response"].strip()
                        sql = sql.replace('```sql', '').replace('```', '').strip()
                        return sql
            except Exception:
                continue
        
        # Fallback: generate simple query
        return f"SELECT * FROM {table_names[0]} LIMIT 10"
    
    def execute_query(self, sql_query: str) -> pd.DataFrame:
        """Execute SQL query and return results"""
        try:
            return pd.read_sql_query(sql_query, self.db_connection)
        except Exception as e:
            if "no such table" in str(e).lower():
                available = list(self.tables.keys())
                raise Exception(f"Table not found. Available tables: {available}. Try asking about these tables instead.")
            raise Exception(f"Query execution failed: {str(e)}")
    
    def generate_insights(self, question: str, query: str, results: pd.DataFrame) -> str:
        """Generate natural language insights from query results"""
        results_summary = f"Query returned {len(results)} rows"
        if len(results) > 0:
            results_summary += f"\nSample results:\n{results.head().to_string()}"
        
        prompt = f"""Original question: {question}
SQL query executed: {query}
Results: {results_summary}

Provide a clear, concise answer to the original question based on the results.
Focus on key insights and numbers. Keep it under 100 words.

Answer:"""
        
        response = requests.post(f"{self.ollama_url}/api/generate", json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        })
        
        return response.json()["response"].strip()
    
    def analyze(self, question: str) -> Dict[str, Any]:
        """Main analysis pipeline"""
        try:
            # Convert NL to SQL
            sql_query = self.nl_to_sql(question)
            
            # Execute query
            results = self.execute_query(sql_query)
            
            # Generate insights
            insights = self.generate_insights(question, sql_query, results)
            
            return {
                'question': question,
                'sql_query': sql_query,
                'results': results.to_dict('records'),
                'insights': insights,
                'success': True
            }
            
        except Exception as e:
            return {
                'question': question,
                'error': str(e),
                'success': False
            }

# Example usage
if __name__ == "__main__":
    # Initialize assistant (make sure Ollama is running)
    assistant = DataAnalystAssistant()
    
    # Load Excel file
    # assistant.load_excel("sales_data.xlsx", "sales")
    
    # Ask questions
    # result = assistant.analyze("What were the top 5 products by revenue last quarter?")
    # print(result['insights'])