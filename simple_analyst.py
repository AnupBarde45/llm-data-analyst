import sqlite3
import requests
import json

class SimpleAnalyst:
    def __init__(self):
        self.db = sqlite3.connect(':memory:')
        
    def load_csv(self, file_path, table_name):
        """Load CSV into SQLite"""
        import pandas as pd
        df = pd.read_csv(file_path)
        df.to_sql(table_name, self.db, if_exists='replace', index=False)
        return f"Loaded {len(df)} rows into {table_name}"
    
    def get_tables(self):
        """Get table info"""
        cursor = self.db.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        info = {}
        for table in tables:
            table_name = table[0]
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = [col[1] for col in cursor.fetchall()]
            info[table_name] = columns
        return info
    
    def nl_to_sql(self, question):
        """Convert question to SQL using Llama"""
        tables = self.get_tables()
        schema = "\n".join([f"Table {name}: {', '.join(cols)}" for name, cols in tables.items()])
        
        prompt = f"Schema:\n{schema}\n\nQuestion: {question}\n\nSQL (only the query):"
        
        response = requests.post("http://localhost:11434/api/generate", json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        })
        
        return response.json()["response"].strip()
    
    def execute_sql(self, sql):
        """Execute SQL query"""
        cursor = self.db.cursor()
        cursor.execute(sql)
        return cursor.fetchall()
    
    def analyze(self, question):
        """Full analysis pipeline"""
        try:
            sql = self.nl_to_sql(question)
            results = self.execute_sql(sql)
            return {"sql": sql, "results": results, "success": True}
        except Exception as e:
            return {"error": str(e), "success": False}

# Test
if __name__ == "__main__":
    analyst = SimpleAnalyst()
    
    # Create test data
    with open("test_data.csv", "w") as f:
        f.write("product,revenue,quarter\n")
        f.write("Laptop,1200,Q1\n")
        f.write("Mouse,25,Q1\n")
        f.write("Keyboard,75,Q2\n")
    
    analyst.load_csv("test_data.csv", "sales")
    result = analyst.analyze("What product has highest revenue?")
    print(result)