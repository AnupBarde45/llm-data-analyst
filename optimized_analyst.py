import pandas as pd
import sqlite3
import requests
from typing import Dict, Any
import os
import json
import re

class OptimizedDataAnalyst:
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
            'sample_data': df.head(3).to_dict('records'),
            'row_count': len(df)
        }
        
        return f"Loaded {len(df)} rows into table '{table_name}'"
    
    def get_enhanced_schema(self) -> str:
        context = "DATABASE SCHEMA:\n"
        for table_name, info in self.tables.items():
            context += f"\nTABLE: {table_name} ({info['row_count']} rows)\n"
            context += f"COLUMNS: {', '.join(info['columns'])}\n"
            context += "SAMPLE DATA:\n"
            for i, row in enumerate(info['sample_data'][:2]):
                context += f"  {row}\n"
        return context
    
    def clean_sql(self, sql: str) -> str:
        """Clean and validate SQL query"""
        # Remove markdown and extra text
        sql = re.sub(r'```sql\n?|```\n?|SQL:|Query:', '', sql, flags=re.IGNORECASE)
        sql = sql.strip()
        
        # Extract only SQL lines
        lines = []
        for line in sql.split('\n'):
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('--') and not line.lower().startswith('note'):
                lines.append(line)
        
        sql = ' '.join(lines)
        
        # Ensure it starts with SELECT, INSERT, UPDATE, DELETE
        if not re.match(r'^\s*(SELECT|INSERT|UPDATE|DELETE)', sql, re.IGNORECASE):
            # Try to find SQL in the text
            sql_match = re.search(r'(SELECT.*?(?:;|$))', sql, re.IGNORECASE | re.DOTALL)
            if sql_match:
                sql = sql_match.group(1)
        
        return sql.strip()
    
    def nl_to_sql(self, question: str) -> str:
        schema = self.get_enhanced_schema()
        table_names = list(self.tables.keys())
        
        if not table_names:
            raise Exception("No tables loaded. Please upload a file first.")
        
        # Enhanced prompt with examples
        prompt = f"""{schema}

You are a SQL expert. Convert the natural language question to SQL.

QUESTION: {question}

CRITICAL RULES:
1. Use ONLY table names: {table_names}
2. Use ONLY columns from schema above
3. Return ONLY valid SQL query
4. No explanations, no markdown, just SQL
5. End with semicolon

EXAMPLES:
- "show all data" → SELECT * FROM {table_names[0]};
- "count rows" → SELECT COUNT(*) FROM {table_names[0]};
- "top 5 by revenue" → SELECT * FROM {table_names[0]} ORDER BY revenue DESC LIMIT 5;

SQL QUERY:"""
        
        try:
            response = requests.post(f"{self.ollama_url}/api/generate", json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "max_tokens": 150
                }
            })
            
            if response.status_code == 200:
                result = response.json()
                if result and "response" in result and result["response"]:
                    sql = self.clean_sql(result["response"])
                    return sql
        except Exception as e:
            print(f"LLM Error: {e}")
        
        # Smart fallback based on question
        if any(word in question.lower() for word in ['count', 'how many', 'total']):
            return f"SELECT COUNT(*) FROM {table_names[0]};"
        elif any(word in question.lower() for word in ['top', 'highest', 'maximum', 'best']):
            return f"SELECT * FROM {table_names[0]} LIMIT 10;"
        else:
            return f"SELECT * FROM {table_names[0]} LIMIT 5;"
    
    def execute_query(self, sql_query: str) -> pd.DataFrame:
        try:
            return pd.read_sql_query(sql_query, self.db_connection)
        except Exception as e:
            if "no such table" in str(e).lower():
                available = list(self.tables.keys())
                raise Exception(f"Table not found. Available: {available}")
            raise Exception(f"SQL Error: {str(e)}")
    
    def generate_simple_insights(self, question: str, results: pd.DataFrame) -> str:
        """Generate insights without LLM for speed"""
        if len(results) == 0:
            return "No results found for your query."
        
        insight = f"Found {len(results)} results. "
        
        if len(results) > 0:
            # Show key statistics
            if len(results.columns) == 1:
                col = results.columns[0]
                if results[col].dtype in ['int64', 'float64']:
                    insight += f"Values range from {results[col].min()} to {results[col].max()}."
                else:
                    insight += f"Sample values: {list(results[col].head(3))}."
            else:
                insight += f"Columns: {list(results.columns)}. "
                insight += f"Sample row: {results.iloc[0].to_dict()}"
        
        return insight
    
    def analyze(self, question: str) -> Dict[str, Any]:
        try:
            sql_query = self.nl_to_sql(question)
            results = self.execute_query(sql_query)
            insights = self.generate_simple_insights(question, results)
            
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

# Test the optimized version
if __name__ == "__main__":
    analyst = OptimizedDataAnalyst()
    
    # Create test data
    test_data = {
        'product': ['Laptop', 'Mouse', 'Keyboard'],
        'price': [1000, 25, 75],
        'sales': [100, 500, 200]
    }
    
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        pd.DataFrame(test_data).to_csv(f.name, index=False)
        analyst.load_file(f.name, 'products')
    
    # Test queries
    questions = [
        "show all products",
        "what product has highest price",
        "count total products"
    ]
    
    for q in questions:
        result = analyst.analyze(q)
        print(f"Q: {q}")
        print(f"SQL: {result.get('sql_query', 'N/A')}")
        print(f"Result: {result.get('insights', result.get('error', 'N/A'))}")
        print("-" * 50)