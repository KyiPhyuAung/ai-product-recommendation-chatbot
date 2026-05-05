import matplotlib.pyplot as plt

# Example weights used in your ranking algorithm
criteria = ['Textual Similarity (TF-IDF)', 'Average Rating (Stars)', 'Review Volume', 'Category Relevance']
weights = [0.40, 0.30, 0.20, 0.10]
colors = ['#4285F4', '#FBBC05', '#EA4335', '#34A853']

plt.figure(figsize=(10, 6))
plt.bar(criteria, weights, color=colors, alpha=0.85, edgecolor='black')

plt.title('Figure 4: Multi-Criteria Scoring Weight Distribution', fontsize=13)
plt.ylabel('Weight Value (Sum = 1.0)')
plt.ylim(0, 0.5)
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Adding percentage labels
for i, v in enumerate(weights):
    plt.text(i, v + 0.01, f'{int(v*100)}%', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('mcdm_weights.png', dpi=300)
plt.show()