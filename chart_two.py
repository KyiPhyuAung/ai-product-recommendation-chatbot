import matplotlib.pyplot as plt

# Categorizing the 15 references by their core research theme
themes = [
    'Recommendation Systems\n(Refs [2], [7], [11], [13])',
    'Similarity & Retrieval\n(Refs [4], [8], [15])',
    'Decision-Making (MCDM)\n(Refs [5], [6], [9], [10])',
    'Modern AI & Datasets\n(Refs [1], [3], [12], [14])'
]
counts = [4, 3, 4, 4]  # Total of 15 papers

# Visual styling
colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2']

plt.figure(figsize=(12, 7))
bars = plt.barh(themes, counts, color=colors, edgecolor='black', alpha=0.8)

# Adding the exact count on top of the bars
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
             f'{width} Papers', va='center', fontweight='bold')

plt.title('Thematic Distribution of the Literature Review (N=15)', fontsize=14, pad=20)
plt.xlabel('Number of Referenced Papers', fontsize=12)
plt.xlim(0, 5) # Setting limit for clear spacing
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()

# Save the plot
plt.savefig('literature_distribution.png', dpi=300)
plt.show()