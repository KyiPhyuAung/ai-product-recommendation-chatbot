import matplotlib.pyplot as plt

# Sample data representing processing times or category distribution
categories = ['Laptop', 'Phones', 'Speakers', 'Headphones', 'Bags']
latencies = [0.16, 0.19, 0.15, 0.13, 0.15] # in seconds

plt.figure(figsize=(8, 5))
plt.bar(categories, latencies, color='#4CAF50')
plt.title('System Search Latency by Category')
plt.xlabel('Product Category')
plt.ylabel('Latency (Seconds)')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save the plot for the report
plt.savefig('latency_chart.png', dpi=300)
plt.show()