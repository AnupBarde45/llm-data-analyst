import pandas as pd
import sqlite3
import mysql.connector
import requests
from typing import Dict, Any
import os
import json
import re
from sqlalchemy import create_engine

class DataAnalystAssistant:
    def __init__(self, api_input: str, use_gemini: bool = False, mysql_config: dict = None):
        self.use_gemini = use_gemini
        self.mysql_config = mysql_config
        self.tables = {}
        
        # Database setup
        if mysql_config:
            self.setup_mysql_connection()
        else:
            self.db_connection = sqlite3.connect(':memory:', check_same_thread=False)
            self.db_type = 'sqlite'
        
        # LLM setup
        if use_gemini:
            import google.generativeai as genai
            genai.configure(api_key=api_input)
            self.model = genai.GenerativeModel('gemini-pro')
        else:
            self.ollama_url = api_input
    
    def setup_mysql_connection(self):
        """Setup MySQL database connection"""
        try:
            # Create SQLAlchemy engine for pandas
            connection_string = f"mysql+mysqlconnector://{self.mysql_config['user']}:{self.mysql_config['password']}@{self.mysql_config['host']}:{self.mysql_config['port']}/{self.mysql_config['database']}"
            self.engine = create_engine(connection_string)
            
            # Create direct MySQL connection for queries
            self.db_connection = mysql.connector.connect(**self.mysql_config)
            self.db_type = 'mysql'
            
            print(f"Connected to MySQL database: {self.mysql_config['database']}")
        except Exception as e:
            print(f"MySQL connection failed: {e}")
            print("Falling back to SQLite...")
            self.db_connection = sqlite3.connect(':memory:', check_same_thread=False)
            self.db_type = 'sqlite'
    
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
        if self.db_type == 'mysql':
            df.to_sql(table_name, self.engine, if_exists='replace', index=False)
        else:
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
    
    def handle_complex_query(self, question: str, table_name: str, columns: list) -> str:
        """Handle AND/OR queries with multiple conditions"""
        question_lower = question.lower()
        
        # Split by AND/OR
        if ' and ' in question_lower:
            parts = question_lower.split(' and ')
            operator = ' AND '
        elif ' or ' in question_lower:
            parts = question_lower.split(' or ')
            operator = ' OR '
        else:
            return None
        
        conditions = []
        detected_columns = []
        
        for part in parts:
            part = part.strip()
            numbers = re.findall(r'\d+', part)
            
            # Find matching column for this part
            part_words = set(part.split())
            best_col = None
            max_score = 0
            
            for col in columns:
                col_words = set(col.lower().replace('_', ' ').split())
                score = len(part_words.intersection(col_words))
                if score > max_score:
                    max_score = score
                    best_col = col
            
            if best_col and numbers:
                detected_columns.append(best_col)
                amount = numbers[0]
                
                # Build condition based on comparison words
                if 'greater' in part or 'more' in part or '>' in part:
                    conditions.append(f"{best_col} > {amount}")
                elif 'less' in part or '<' in part:
                    conditions.append(f"{best_col} < {amount}")
                elif 'between' in part and len(numbers) >= 2:
                    conditions.append(f"{best_col} BETWEEN {numbers[0]} AND {numbers[1]}")
                else:
                    conditions.append(f"{best_col} = {amount}")
        
        if conditions:
            where_clause = operator.join(conditions)
            print(f"Detected columns: {detected_columns}")
            print(f"Complex query conditions: {where_clause}")
            return f"SELECT * FROM {table_name} WHERE {where_clause}"
        
        return None
    
    def smart_search(self, question: str) -> str:
        """Multi-strategy search for robust results"""
        table_names = list(self.tables.keys())
        
        if not table_names:
            raise Exception("No tables loaded.")
        
        table_name = table_names[0]
        columns = self.tables[table_name]['columns']
        question_lower = question.lower()
        
        # Handle complex AND/OR queries first
        if ' and ' in question_lower or ' or ' in question_lower:
            complex_result = self.handle_complex_query(question, table_name, columns)
            if complex_result:
                return complex_result
        
        # Dynamic query detection based on actual data
        numbers = re.findall(r'\d+', question)
        
        # Get column types from stored analysis
        column_types = self.tables[table_name].get('column_types', {})
        dtypes = self.tables[table_name].get('dtypes', {})
        
        # Find numeric columns
        numeric_cols = [col for col, col_type in column_types.items() if col_type in ['numeric', 'numeric_discrete', 'numeric_continuous']]
        if not numeric_cols:
            numeric_cols = [col for col, dtype in dtypes.items() if 'int' in str(dtype) or 'float' in str(dtype)]
        
        # Find best matching column by comparing question words with column names
        question_words = set(question_lower.split())
        best_col = None
        max_score = 0
        
        for col in numeric_cols:
            col_words = set(col.lower().replace('_', ' ').split())
            score = len(question_words.intersection(col_words))
            if score > max_score:
                max_score = score
                best_col = col
        
        # If no word match, use first numeric column
        if not best_col and numeric_cols:
            best_col = numeric_cols[0]
        
        if best_col and numbers:
            print(f"Detected column: {best_col} (type: {column_types.get(best_col, 'unknown')})")
            
            # Handle BETWEEN queries
            if 'between' in question_lower and len(numbers) >= 2:
                return f"SELECT * FROM {table_name} WHERE {best_col} BETWEEN {numbers[0]} AND {numbers[1]}"
            
            # Handle range queries (from X to Y)
            elif any(pattern in question_lower for pattern in ['from', 'to']) and len(numbers) >= 2:
                return f"SELECT * FROM {table_name} WHERE {best_col} BETWEEN {numbers[0]} AND {numbers[1]}"
            
            # Handle single number comparisons
            elif len(numbers) >= 1:
                amount = numbers[0]
                if 'greater' in question_lower or 'more' in question_lower or '>' in question:
                    return f"SELECT * FROM {table_name} WHERE {best_col} > {amount}"
                elif 'less' in question_lower or '<' in question:
                    return f"SELECT * FROM {table_name} WHERE {best_col} < {amount}"
                else:
                    return f"SELECT * FROM {table_name} WHERE {best_col} = {amount}"
        
        # Always fallback to LLM with enhanced prompt
        schema = self.get_schema_context()
        
        prompt = f"""Database Schema:
{schema}

Question: {question}

Write a SQL SELECT query using the exact column names shown above.
Use BETWEEN for range queries.
Use AND/OR for multiple conditions.
Detect all column names mentioned in the question.

SQL:"""
        
        print(f"Available columns: {columns}")
        print(f"Calling LLM with enhanced prompt...")
        llm_response = self.call_llm(prompt, 150)
        print(f"LLM response: {llm_response}")
        
        if llm_response:
            cleaned = self.clean_sql(llm_response)
            print(f"Generated SQL: {cleaned}")
            return cleaned
        
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
            # Always fallback to LLM
            schema = self.get_schema_context()
            prompt = f"{schema}\n\nQUESTION: {question}\n\nGenerate SQL query using EXACT column names from schema above. Return ONLY the SQL query.\n\nSQL:"
            
            llm_response = self.call_llm(prompt, 100)
            if llm_response:
                return self.clean_sql(llm_response)
            
            raise Exception("Could not generate SQL query for your question.")
    
    def execute_query(self, sql_query: str) -> pd.DataFrame:
        try:
            if self.db_type == 'mysql':
                return pd.read_sql_query(sql_query, self.engine)
            else:
                return pd.read_sql_query(sql_query, self.db_connection)
        except Exception as e:
            if "no such table" in str(e).lower() or "doesn't exist" in str(e).lower():
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