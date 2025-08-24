import pandas as pd
import sqlite3
import requests
from typing import Dict, Any
import os
import json
import re

class DataAnalystAssistant:
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url
        self.db_connection = sqlite3.connect(':memory:', check_same_thread=False)
        self.tables = {}
        
    def load_file(self, file_path: str, table_name: str = None) -> str:
        if not table_name:
            table_name = os.path.splitext(os.path.basename(file_path))[0].lower()
        
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
            
        df.to_sql(table_name, self.db_connection, if_exists='replace', index=False)
        
        self.tables[table_name] = {
            'columns': list(df.columns),
            'sample_data': df.head(3).to_dict('records')
        }
        
        return f"Loaded {len(df)} rows into table '{table_name}'"
    
    def load_excel(self, file_path: str, table_name: str = None) -> str:
        return self.load_file(file_path, table_name)
    
    def get_schema_context(self) -> str:
        context = "DATABASE SCHEMA:\n"
        for table_name, info in self.tables.items():
            context += f"\nTABLE: {table_name}\n"
            context += f"COLUMNS: {', '.join(info['columns'])}\n"
            context += f"SAMPLE: {info['sample_data'][0] if info['sample_data'] else 'No data'}\n"
        return context
    
    def clean_sql(self, sql: str) -> str:
        # Remove markdown and extra text
        sql = re.sub(r'```sql\n?|```\n?|SQL:|Query:', '', sql, flags=re.IGNORECASE)
        sql = sql.strip()
        
        # Find actual SQL query - look for SELECT, INSERT, UPDATE, DELETE
        sql_pattern = r'(SELECT.*?(?:;|$)|INSERT.*?(?:;|$)|UPDATE.*?(?:;|$)|DELETE.*?(?:;|$))'
        match = re.search(sql_pattern, sql, re.IGNORECASE | re.DOTALL)
        
        if match:
            return match.group(1).strip().rstrip(';')
        
        # If no SQL found, extract lines that look like SQL
        lines = []
        for line in sql.split('\n'):
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('--'):
                # Skip explanatory text
                if any(word in line.lower() for word in ['to find', 'you would', 'here\'s how', 'this query']):
                    continue
                if line.upper().startswith(('SELECT', 'INSERT', 'UPDATE', 'DELETE', 'FROM', 'WHERE')):
                    lines.append(line)
        
        return ' '.join(lines).strip()
    
    def nl_to_sql(self, question: str) -> str:
        schema = self.get_schema_context()
        table_names = list(self.tables.keys())
        
        if not table_names:
            raise Exception("No tables loaded. Please upload a file first.")
        
        # Better prompt for accurate SQL
        prompt = f"""{schema}

Generate ONLY a SQL query for this question: {question}

IMPORTANT:
- Return ONLY the SQL query
- No explanations or text
- Start with SELECT
- Use exact table/column names from schema

SQL:"""
        
        try:
            response = requests.post(f"{self.ollama_url}/api/generate", json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0, "num_predict": 50}
            })
            
            if response.status_code == 200:
                result = response.json()
                if result and "response" in result and result["response"]:
                    return self.clean_sql(result["response"])
        except Exception:
            pass
        
        # Smart fallback
        if 'count' in question.lower():
            return f"SELECT COUNT(*) FROM {table_names[0]}"
        elif any(word in question.lower() for word in ['top', 'highest', 'max']):
            return f"SELECT * FROM {table_names[0]} LIMIT 10"
        else:
            return f"SELECT * FROM {table_names[0]} LIMIT 5"
    
    def execute_query(self, sql_query: str) -> pd.DataFrame:
        try:
            return pd.read_sql_query(sql_query, self.db_connection)
        except Exception as e:
            if "no such table" in str(e).lower():
                available = list(self.tables.keys())
                raise Exception(f"Table not found. Available: {available}")
            raise Exception(f"Query failed: {str(e)}")
    
    def generate_insights(self, question: str, query: str, results: pd.DataFrame) -> str:
        """Generate natural language insights using LLM"""
        if len(results) == 0:
            return "No results found for your query."
        
        # Prepare concise results summary
        summary = f"Query returned {len(results)} rows."
        if len(results) > 0:
            summary += f" Sample: {results.head(2).to_dict('records')}"
        
        prompt = f"Question: {question}\nResults: {summary}\n\nAnswer the question naturally based on the results. Be conversational and helpful."
        
        try:
            response = requests.post(f"{self.ollama_url}/api/generate", json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2, "num_predict": 80}
            })
            
            if response.status_code == 200:
                result = response.json()
                if result and "response" in result and result["response"]:
                    return result["response"].strip()
        except Exception:
            pass
        
        # Simple fallback
        return f"Found {len(results)} results for your query."
        

    
    def analyze(self, question: str) -> Dict[str, Any]:
        try:
            sql_query = self.nl_to_sql(question)
            results = self.execute_query(sql_query)
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