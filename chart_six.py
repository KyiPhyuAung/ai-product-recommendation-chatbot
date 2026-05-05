import matplotlib.pyplot as plt

# Data from your initial testing phases
categories = ['Electronics', 'Fashion', 'Home Goods', 'Accessories']
latency = [412, 195, 308, 172] # Latency in milliseconds

plt.figure(figsize=(10, 6))
bars = plt.bar(categories, latency, color='#003366', edgecolor='black', alpha=0.8)

# Adding labels and styling
plt.ylabel('Latency (ms)', fontsize=12)
plt.title('Figure 6: System Search Latency across Primary Categories', fontsize=14, pad=20)
plt.ylim(0, 500)
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Adding the exact values on top of the bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 5,
             f'{height}ms', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('search_latency.png', dpi=300)
print("Chart saved as search_latency.png")
plt.show()