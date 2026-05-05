import matplotlib.pyplot as plt

def draw_clean_logic_comparison():
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Styles
    title_style = dict(fontsize=14, fontweight='bold', ha='center')
    trad_style = dict(boxstyle="round,pad=0.6", fc="#F8D7DA", ec="#721C24", lw=1.5) # Reddish
    ai_style = dict(boxstyle="round,pad=0.6", fc="#D1E7DD", ec="#0F5132", lw=1.5)  # Greenish
    header_style = dict(boxstyle="square,pad=0.5", fc="#E2E3E5", ec="#383D41", lw=2)

    # Column Headers
    ax.text(2.5, 9, "Traditional Keyword Search", **title_style)
    ax.text(7.5, 9, "Proposed Hybrid AI Search", **title_style)

    # Stages for Traditional
    ax.text(2.5, 7.5, "User Input", bbox=trad_style, ha='center')
    ax.text(2.5, 5.5, "Exact Word Matching", bbox=trad_style, ha='center')
    ax.text(2.5, 3.5, "Unfiltered Results\n(High Accessory Noise)", bbox=trad_style, ha='center', fontsize=9)

    # Stages for AI Proposed
    ax.text(7.5, 8.0, "User Input", bbox=ai_style, ha='center')
    ax.text(7.5, 6.5, "TF-IDF + Cosine Similarity\n(Semantic Retrieval)", bbox=ai_style, ha='center', fontsize=9)
    ax.text(7.5, 4.5, "MCDM Reranking\n(Ratings & Reviews)", bbox=ai_style, ha='center', fontsize=9)
    ax.text(7.5, 2.5, "Filtered & Ranked Top 5\n(High Quality)", bbox=ai_style, ha='center', fontsize=9)

    # Drawing Arrows
    arrow_props = dict(arrowstyle='->', lw=1.5, color='#495057')
    
    # Trad Arrows
    ax.annotate('', xy=(2.5, 6.2), xytext=(2.5, 6.8), arrowprops=arrow_props)
    ax.annotate('', xy=(2.5, 4.2), xytext=(2.5, 4.8), arrowprops=arrow_props)
    
    # AI Arrows
    ax.annotate('', xy=(7.5, 7.2), xytext=(7.5, 7.4), arrowprops=arrow_props)
    ax.annotate('', xy=(7.5, 5.3), xytext=(7.5, 5.8), arrowprops=arrow_props)
    ax.annotate('', xy=(7.5, 3.3), xytext=(7.5, 3.8), arrowprops=arrow_props)

    plt.title("Figure 5: Comparative logic between Traditional and Hybrid AI Search", fontsize=15, pad=30)
    plt.tight_layout()
    plt.savefig('logic_comparison_v2.png', dpi=300)
    plt.show()

