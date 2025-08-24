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
            
            # Show actual data patterns
            context += "SAMPLE DATA:\n"
            for i, sample in enumerate(info['sample_data'][:3]):
                context += f"  Row {i+1}: {sample}\n"
        
        return context
    
    def extract_entities(self, question: str) -> Dict[str, str]:
        """Extract meaningful entities from the question"""
        entities = {}
        
        # Comprehensive patterns for movie titles and names
        patterns = [
            r'director of\s+(.+?)(?:\?|$|who|what)',  # "director of 3 idiots"
            r'who directed\s+(.+?)(?:\?|$)',          # "who directed 3 idiots"
            r'cast of\s+(.+?)(?:\?|$)',               # "cast of 3 idiots"
            r'actor in\s+(.+?)(?:\?|$)',              # "actor in 3 idiots"
            r'starring in\s+(.+?)(?:\?|$)',           # "starring in 3 idiots"
            r'movie\s+(.+?)(?:\?|$)',                 # "movie 3 idiots"
            r'film\s+(.+?)(?:\?|$)',                  # "film 3 idiots"
            r'["\'](.*?)["\']',                       # quoted text
        ]
        
        for pattern in patterns:
            match = re.search(pattern, question.lower())
            if match:
                title = match.group(1).strip()
                if title and len(title) > 1:
                    entities['title'] = title
                    break
        
        return entities
    
    def find_similar_titles(self, search_term: str, table_name: str) -> str:
        """Find similar titles in actual data"""
        try:
            # Get all titles from the data
            cursor = self.db_connection.cursor()
            cursor.execute(f"SELECT DISTINCT Title FROM {table_name}")
            titles = [row[0] for row in cursor.fetchall() if row[0]]
            
            # Find best match
            search_lower = search_term.lower()
            
            # Exact match first
            for title in titles:
                if search_lower == title.lower():
                    return title
            
            # Partial match
            for title in titles:
                if search_lower in title.lower() or title.lower() in search_lower:
                    return title
            
            # Word-by-word match
            search_words = search_lower.split()
            for title in titles:
                title_lower = title.lower()
                if all(word in title_lower for word in search_words):
                    return title
            
            return search_term  # fallback
        except:
            return search_term
    
    def smart_search(self, question: str) -> str:
        """Multi-strategy search for robust results"""
        table_names = list(self.tables.keys())
        
        if not table_names:
            raise Exception("No tables loaded.")
        
        # Strategy 1: Entity extraction
        entities = self.extract_entities(question)
        
        if 'title' in entities:
            # Find actual matching title in data
            actual_title = self.find_similar_titles(entities['title'], table_names[0])
            
            # Determine what to select based on question
            if any(word in question.lower() for word in ['director', 'directed']):
                return f"SELECT Director FROM {table_names[0]} WHERE Title LIKE '%{actual_title}%'"
            elif any(word in question.lower() for word in ['cast', 'actor', 'starring']):
                return f"SELECT Cast FROM {table_names[0]} WHERE Title LIKE '%{actual_title}%'"
            else:
                return f"SELECT * FROM {table_names[0]} WHERE Title LIKE '%{actual_title}%'"
        
        # Strategy 2: Keyword-based search
        search_terms = re.findall(r'\b\w+\b', question.lower())
        search_terms = [term for term in search_terms if len(term) > 2 and 
                       term not in ['who', 'what', 'the', 'and', 'director', 'actor', 'cast', 'movie', 'film']]
        
        if search_terms:
            conditions = []
            for term in search_terms:
                conditions.append(f"Title LIKE '%{term}%'")
            
            where_clause = " AND ".join(conditions)
            
            if any(word in question.lower() for word in ['director', 'directed']):
                return f"SELECT Director FROM {table_names[0]} WHERE {where_clause}"
            elif any(word in question.lower() for word in ['cast', 'actor', 'starring']):
                return f"SELECT Cast FROM {table_names[0]} WHERE {where_clause}"
            else:
                return f"SELECT * FROM {table_names[0]} WHERE {where_clause}"
        
        # Fallback
        return f"SELECT * FROM {table_names[0]} LIMIT 5"
    
    def clean_sql(self, sql: str) -> str:
        sql = re.sub(r'```sql\n?|```\n?|SQL:|Query:', '', sql, flags=re.IGNORECASE)
        sql = sql.strip()
        
        sql_pattern = r'(SELECT.*?(?:;|$)|INSERT.*?(?:;|$)|UPDATE.*?(?:;|$)|DELETE.*?(?:;|$))'
        match = re.search(sql_pattern, sql, re.IGNORECASE | re.DOTALL)
        
        if match:
            sql = match.group(1).strip().rstrip(';')
        
        sql = re.sub(r'"([^"]*?)"', r"'\1'", sql)
        return sql
    
    def nl_to_sql(self, question: str) -> str:
        # Use smart search as primary method
        try:
            return self.smart_search(question)
        except Exception:
            # Fallback to LLM if smart search fails
            schema = self.get_schema_context()
            table_names = list(self.tables.keys())
            
            prompt = f"""{schema}

QUESTION: {question}

Generate SQL query. Use LIKE for partial text matches.

SQL:"""
            
            try:
                response = requests.post(f"{self.ollama_url}/api/generate", json={
                    "model": "llama3",
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0, "num_predict": 100}
                })
                
                if response.status_code == 200:
                    result = response.json()
                    if result and "response" in result and result["response"]:
                        return self.clean_sql(result["response"])
            except Exception:
                pass
            
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