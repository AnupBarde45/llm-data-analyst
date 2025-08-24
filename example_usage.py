from data_analyst import DataAnalystAssistant
import pandas as pd

# Create sample data for testing
def create_sample_data():
    # Sample sales data
    sales_data = {
        'product_name': ['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Laptop', 'Mouse'],
        'revenue': [1200, 25, 75, 300, 1100, 30],
        'quantity': [1, 1, 1, 1, 1, 1],
        'quarter': ['Q1', 'Q1', 'Q1', 'Q2', 'Q2', 'Q2'],
        'region': ['North', 'North', 'South', 'North', 'South', 'North']
    }
    
    df = pd.DataFrame(sales_data)
    df.to_excel('sample_sales.xlsx', index=False)
    print("Created sample_sales.xlsx")

def main():
    # Create sample data
    create_sample_data()
    
    # Initialize assistant (make sure Ollama is running)
    assistant = DataAnalystAssistant()
    
    # Load Excel file
    result = assistant.load_excel('sample_sales.xlsx', 'sales')
    print(result)
    
    # Example questions
    questions = [
        "What were the top selling products by revenue?",
        "Which quarter had higher sales?",
        "What is the total revenue by region?",
        "Show me products with revenue over 100"
    ]
    
    for question in questions:
        print(f"\n{'='*50}")
        print(f"Question: {question}")
        print('='*50)
        
        result = assistant.analyze(question)
        
        if result['success']:
            print(f"SQL Query: {result['sql_query']}")
            print(f"Insights: {result['insights']}")
        else:
            print(f"Error: {result['error']}")

if __name__ == "__main__":
    main()