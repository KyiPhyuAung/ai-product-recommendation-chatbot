import matplotlib.pyplot as plt
import numpy as np

# Performance data comparing the two systems
categories = ['Electronics', 'Fashion', 'Home Goods', 'Accessories']
standard_precision = [0.62, 0.68, 0.58, 0.81]  # Baseline keyword search
ai_precision = [0.94, 0.89, 0.86, 0.96]        # Your Hybrid AI search (MCDM)

x = np.arange(len(categories))
width = 0.35 

fig, ax = plt.subplots(figsize=(10, 6))

# Creating bars with a professional color scheme
rects1 = ax.bar(x - width/2, standard_precision, width, label='Standard Keyword Search', 
                color='#ff9999', edgecolor='black', alpha=0.8)
rects2 = ax.bar(x + width/2, ai_precision, width, label='Proposed AI Search (MCDM)', 
                color='#66b3ff', edgecolor='black', alpha=0.8)

# Adding labels, title, and formatting
ax.set_ylabel('Precision Score (0.0 - 1.0)', fontsize=11)
ax.set_title('Figure 7: Comparison of Retrieval Precision by Category', fontsize=13, pad=20)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=10)
ax.legend(frameon=True, shadow=True)
ax.set_ylim(0, 1.1)
ax.grid(axis='y', linestyle='--', alpha=0.6)

# Adding data labels on top of the bars
def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), 
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')

add_labels(rects1)
add_labels(rects2)

plt.tight_layout()
plt.savefig('precision_comparison.png', dpi=300)
print("Chart saved as precision_comparison.png")
plt.show()