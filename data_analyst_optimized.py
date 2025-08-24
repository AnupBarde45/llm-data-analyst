import pandas as pd
import sqlite3
import requests
from typing import Dict, Any
import os
import json
import re

class DataAnalystAssistant:
    def __init__(self, api_input: str, use_gemini: bool = False):
        self.use_gemini = use_gemini
        self.db_connection = sqlite3.connect(':memory:', check_same_thread=False)
        self.tables = {}
        
        if use_gemini:
            import google.generativeai as genai
            genai.configure(api_key=api_input)
            self.model = genai.GenerativeModel('gemini-pro')
        else:
            self.ollama_url = api_input
    
    def call_llm(self, prompt: str, max_tokens: int = 100) -> str:
        """Call either Gemini or Ollama"""
        try:
            if self.use_gemini:
                import google.generativeai as genai
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.1,
                        max_output_tokens=max_tokens
                    )
                )
                result = response.text.strip()
                print(f"Gemini raw response: {result}")  # Debug
                return result
            else:
                response = requests.post(f"{self.ollama_url}/api/generate", json={
                    "model": "llama3",
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0, "num_predict": max_tokens}
                })
                if response.status_code == 200:
                    result = response.json()
                    if result and "response" in result:
                        ollama_result = result["response"].strip()
                        print(f"Ollama raw response: {ollama_result}")  # Debug
                        return ollama_result
            return ""
        except Exception as e:
            print(f"LLM error: {e}")
            return ""
        
    def smart_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean column names based on actual data"""
        col_mapping = {}
        
        for col in df.columns:
            # Basic cleaning only
            clean_col = str(col).strip()
            clean_col = clean_col.replace('(', '_').replace(')', '')
            clean_col = clean_col.replace('[', '_').replace(']', '')
            clean_col = clean_col.replace('-', '_').replace(' ', '_')
            clean_col = clean_col.replace('__', '_').strip('_')
            
            # Just capitalize words, no hardcoded replacements
            clean_col = '_'.join(word.capitalize() for word in clean_col.split('_'))
            col_mapping[col] = clean_col
        
        df = df.rename(columns=col_mapping)
        print(f"Column mapping: {col_mapping}")
        return df
    
    def analyze_column_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """Analyze column types based on actual data content"""
        column_info = {}
        
        for col in df.columns:
            sample_values = df[col].dropna().head(20)
            
            # Analyze based on actual data type and content
            if df[col].dtype in ['int64', 'float64']:
                # Check if it's an ID (sequential or unique integers)
                if df[col].dtype == 'int64' and df[col].nunique() == len(df[col].dropna()):
                    column_info[col] = 'identifier'
                else:
                    column_info[col] = 'numeric'
            elif df[col].dtype == 'object':
                # Check if it's mostly text
                avg_length = sample_values.astype(str).str.len().mean()
                if avg_length > 10:  # Longer strings are likely text/names
                    column_info[col] = 'text'
                else:
                    column_info[col] = 'categorical'
            else:
                column_info[col] = 'other'
        
        return column_info
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare data for analysis"""
        print(f"Original data shape: {df.shape}")
        print(f"Original columns: {list(df.columns)}")
        
        # 1. Smart column name cleaning
        df = self.smart_column_names(df)
        
        # 2. Remove completely empty rows and columns
        df = df.dropna(how='all')
        df = df.dropna(axis=1, how='all')
        
        # 3. Clean string columns
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].replace(['nan', 'NaN', 'null', 'NULL', ''], None)
        
        # 4. Smart numeric conversion
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try to convert to numeric
                numeric_series = pd.to_numeric(df[col], errors='coerce')
                # Only convert if more than 80% of values are numeric
                if numeric_series.notna().sum() / len(df) > 0.8:
                    df[col] = numeric_series
        
        # 5. Handle missing values intelligently
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna('Unknown')
        
        # 6. Analyze column types
        column_types = self.analyze_column_types(df)
        
        print(f"Cleaned data shape: {df.shape}")
        print(f"Final columns: {list(df.columns)}")
        print(f"Column categories: {column_types}")
        
        return df
    
    def load_file(self, file_path: str, table_name: str = None) -> str:
        if not table_name:
            table_name = os.path.splitext(os.path.basename(file_path))[0].lower()
        
        # Load data
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        
        # Clean data before processing
        df = self.clean_data(df)
        
        # Load into database
        df.to_sql(table_name, self.db_connection, if_exists='replace', index=False)
        
        # Analyze column types for better query generation
        column_types = self.analyze_column_types(df)
        
        self.tables[table_name] = {
            'columns': list(df.columns),
            'sample_data': df.head(3).to_dict('records'),
            'dtypes': dict(df.dtypes),
            'column_types': column_types
        }
        
        return f"Loaded and cleaned {len(df)} rows into table '{table_name}'. Columns: {list(df.columns)}"
    
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
        
        table_name = table_names[0]
        columns = self.tables[table_name]['columns']
        question_lower = question.lower()
        
        # Dynamic query detection based on actual data
        numbers = re.findall(r'\d+', question)
        if numbers and any(word in question_lower for word in ['greater', 'less', 'more', 'than', '>', '<', '=', 'is']):
            # Get column types from stored analysis
            column_types = self.tables[table_name].get('column_types', {})
            
            # Find numeric columns
            numeric_cols = [col for col, col_type in column_types.items() if col_type in ['numeric', 'numeric_discrete', 'numeric_continuous']]
            
            # If no column types stored, check data types
            if not numeric_cols:
                dtypes = self.tables[table_name].get('dtypes', {})
                numeric_cols = [col for col, dtype in dtypes.items() if 'int' in str(dtype) or 'float' in str(dtype)]
            
            # Find best matching column by comparing question words with column names
            question_words = set(question_lower.split())
            best_col = None
            max_score = 0
            
            for col in numeric_cols:
                col_words = set(col.lower().replace('_', ' ').split())
                # Score based on word overlap
                score = len(question_words.intersection(col_words))
                if score > max_score:
                    max_score = score
                    best_col = col
            
            # If no word match, use first numeric column
            if not best_col and numeric_cols:
                best_col = numeric_cols[0]
            
            if best_col:
                amount = numbers[0]
                if 'greater' in question_lower or 'more' in question_lower or '>' in question:
                    return f"SELECT * FROM {table_name} WHERE {best_col} > {amount}"
                elif 'less' in question_lower or '<' in question:
                    return f"SELECT * FROM {table_name} WHERE {best_col} < {amount}"
                else:
                    return f"SELECT * FROM {table_name} WHERE {best_col} = {amount}"
        
        # Strategy 1: Entity extraction for movie/title data
        entities = self.extract_entities(question)
        
        if 'title' in entities:
            # Find title column
            title_col = None
            for col in columns:
                if 'title' in col.lower() or 'name' in col.lower() or 'movie' in col.lower():
                    title_col = col
                    break
            
            if title_col:
                actual_title = self.find_similar_titles(entities['title'], table_name)
                
                if any(word in question_lower for word in ['director', 'directed']):
                    director_col = next((col for col in columns if 'director' in col.lower()), None)
                    if director_col:
                        return f"SELECT {director_col} FROM {table_name} WHERE {title_col} LIKE '%{actual_title}%'"
                elif any(word in question_lower for word in ['cast', 'actor', 'starring']):
                    cast_col = next((col for col in columns if 'cast' in col.lower() or 'actor' in col.lower()), None)
                    if cast_col:
                        return f"SELECT {cast_col} FROM {table_name} WHERE {title_col} LIKE '%{actual_title}%'"
                else:
                    return f"SELECT * FROM {table_name} WHERE {title_col} LIKE '%{actual_title}%'"
        
        # Always fallback to LLM with better prompt
        schema = self.get_schema_context()
        
        # Simple, direct prompt
        prompt = f"""Database Schema:
{schema}

Question: {question}

Write a SQL SELECT query using the exact column names shown above.

SQL:"""
        
        print(f"Calling LLM with prompt: {prompt[:200]}...")  # Debug
        llm_response = self.call_llm(prompt, 150)
        print(f"LLM response: {llm_response}")  # Debug
        
        if llm_response:
            cleaned = self.clean_sql(llm_response)
            print(f"Cleaned SQL: {cleaned}")  # Debug
            return cleaned
        
        # If LLM fails, try simple pattern matching
        numbers = re.findall(r'\d+', question)
        if numbers:
            # Just use first numeric column and first number
            for col in columns:
                try:
                    cursor = self.db_connection.cursor()
                    cursor.execute(f"SELECT typeof({col}) FROM {table_name} LIMIT 1")
                    col_type = cursor.fetchone()[0]
                    if col_type in ['integer', 'real']:
                        return f"SELECT * FROM {table_name} WHERE {col} = {numbers[0]}"
                except:
                    continue
        
        raise Exception(f"Could not generate SQL. Available columns: {columns}")
    
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
            # Always fallback to LLM - no LIMIT 5
            schema = self.get_schema_context()
            prompt = f"{schema}\n\nQUESTION: {question}\n\nGenerate SQL query using EXACT column names from schema above. Return ONLY the SQL query.\n\nSQL:"
            
            llm_response = self.call_llm(prompt, 100)
            if llm_response:
                return self.clean_sql(llm_response)
            
            # If LLM also fails, raise error instead of LIMIT 5
            raise Exception("Could not generate SQL query for your question.")
    
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
        
        summary = f"Query returned {len(results)} rows. Sample: {results.head(2).to_dict('records')}"
        prompt = f"Question: {question}\nResults: {summary}\n\nAnswer naturally and conversationally:"
        
        llm_response = self.call_llm(prompt, 150)
        return llm_response if llm_response else f"Found {len(results)} results for your query."
    
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