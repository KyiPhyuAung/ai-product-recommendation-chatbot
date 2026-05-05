import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_flowchart():
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Define box style
    box_style = dict(boxstyle='round,pad=0.5', facecolor='#E8F0FE', edgecolor='#1A73E8', linewidth=2)
    
    # 1. User Input
    ax.text(5, 9.5, 'User Query (Chatbot Interface)', ha='center', va='center', bbox=box_style, fontsize=12)
    
    # 2. NLP Preprocessing
    ax.text(5, 8, 'NLP Preprocessing\n(Cleaning, Stemming, Tokenization)', ha='center', va='center', bbox=box_style, fontsize=11)
    
    # 3. Vectorization
    ax.text(3, 6, 'TF-IDF Vectorization\n(Query vs. 1.4M items)', ha='center', va='center', bbox=box_style, fontsize=10)
    
    # 4. Initial Retrieval
    ax.text(3, 4, 'Cosine Similarity Scoring\n(Top K Candidates)', ha='center', va='center', bbox=box_style, fontsize=10)
    
    # 5. MCDM Reranking
    ax.text(7, 5, 'Multi-Criteria Reranking\n(MCDM Framework)\n- Ratings\n- Reviews\n- Price/Popularity', ha='center', va='center', 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF4E5', edgecolor='#D97706', linewidth=2), fontsize=10)
    
    # 6. Final Result
    ax.text(5, 2, 'Top 5 Personalized Recommendations', ha='center', va='center', 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#E6F4EA', edgecolor='#1E8E3E', linewidth=2), fontsize=12)

    # Drawing Arrows
    arrow_props = dict(arrowstyle='->', lw=1.5, color='gray')
    ax.annotate('', xy=(5, 8.5), xytext=(5, 9.1), arrowprops=arrow_props)
    ax.annotate('', xy=(3, 6.5), xytext=(4.5, 7.5), arrowprops=arrow_props)
    ax.annotate('', xy=(3, 4.5), xytext=(3, 5.5), arrowprops=arrow_props)
    ax.annotate('', xy=(6, 5), xytext=(3.5, 4), arrowprops=arrow_props)
    ax.annotate('', xy=(5, 2.5), xytext=(6.5, 4.5), arrowprops=arrow_props)

    plt.title('Figure 3: High-Level System Architecture of the AI Search Chatbot', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig('system_architecture.png', dpi=300)
    plt.show()

draw_flowchart()