import os
import psutil
from product_search import ProductSearch

def check_system_resources():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024) # Return in MB

# Initialize the engine
search_engine = ProductSearch()
search_engine.load()

print(f"Initial Memory Usage: {check_system_resources():.2f} MB")

# Perform a search
results, keywords, note = search_engine.search("Gaming Laptop", top_n=5)

print(f"Memory Usage after Search: {check_system_resources():.2f} MB")