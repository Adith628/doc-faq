"""
Generate architecture flowchart for Mini Doc-FAQ Agent
Requires: pip install graphviz matplotlib
"""

try:
    from graphviz import Digraph
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False
    print("Graphviz not available. Install with: pip install graphviz")
    print("Also install Graphviz system package: https://graphviz.org/download/")

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Matplotlib not available. Install with: pip install matplotlib")


def generate_graphviz_flowchart():
    """Generate flowchart using Graphviz"""
    if not GRAPHVIZ_AVAILABLE:
        print("Graphviz not available. Skipping Graphviz flowchart generation.")
        return
    
    dot = Digraph(comment='Mini Doc-FAQ Agent Architecture', format='png')
    dot.attr(rankdir='TB', size='12,16')
    dot.attr('node', shape='box', style='rounded,filled')
    
    # Start
    dot.node('start', 'User Starts App', shape='ellipse', fillcolor='#e3f2fd')
    
    # API Check
    dot.node('check_api', 'API Key\nAvailable?', shape='diamond', fillcolor='#fff9c4')
    
    # Error
    dot.node('error', 'Show Error\nMessage', fillcolor='#ffcdd2')
    
    # Input methods
    dot.node('input', 'Input Method', shape='diamond', fillcolor='#fff9c4')
    dot.node('file', 'Upload .txt/.md\nFile', fillcolor='#c8e6c9')
    dot.node('text', 'Paste Text', fillcolor='#c8e6c9')
    
    # Index button
    dot.node('index_btn', 'Click Index\nDocument', fillcolor='#bbdefb')
    
    # Indexer Agent
    dot.node('indexer', 'Indexer Agent', fillcolor='#e1f5ff', style='rounded,filled,bold')
    dot.node('chunk', 'Chunk Text by\nSentences with Overlap', fillcolor='#b3e5fc')
    dot.node('embed', 'Generate Embeddings\n(OpenAI/Gemini)', fillcolor='#81d4fa')
    dot.node('normalize', 'Normalize Embeddings\n(L2 Normalization)', fillcolor='#4fc3f7')
    dot.node('faiss', 'Build FAISS Index\n(IndexFlatIP)', fillcolor='#e8f5e9')
    dot.node('cache', 'Cache Index', fillcolor='#c5e1a5')
    dot.node('ready', 'Index Ready', fillcolor='#a5d6a7', shape='ellipse')
    
    # Query
    dot.node('query', 'User Query', shape='diamond', fillcolor='#fff9c4')
    
    # Answerer Agent
    dot.node('answerer', 'Answerer Agent', fillcolor='#fff4e1', style='rounded,filled,bold')
    dot.node('qembed', 'Embed Query', fillcolor='#ffe0b2')
    dot.node('search', 'Search FAISS Index\n(Top-K Retrieval)', fillcolor='#ffcc80')
    dot.node('retrieve', 'Retrieve Top 4 Chunks\nwith Scores', fillcolor='#ffb74d')
    dot.node('llm', 'Generate Answer\nwith LLM', fillcolor='#fce4ec')
    dot.node('citations', 'Add Inline Citations\n[1], [2], etc.', fillcolor='#f8bbd0')
    dot.node('display', 'Display Answer', fillcolor='#f48fb1', shape='ellipse')
    
    # Edges
    dot.edge('start', 'check_api')
    dot.edge('check_api', 'error', label='No')
    dot.edge('check_api', 'input', label='Yes')
    dot.edge('input', 'file', label='File')
    dot.edge('input', 'text', label='Text')
    dot.edge('file', 'index_btn')
    dot.edge('text', 'index_btn')
    dot.edge('index_btn', 'indexer')
    dot.edge('indexer', 'chunk')
    dot.edge('chunk', 'embed')
    dot.edge('embed', 'normalize')
    dot.edge('normalize', 'faiss')
    dot.edge('faiss', 'cache')
    dot.edge('cache', 'ready')
    dot.edge('ready', 'query')
    dot.edge('query', 'answerer')
    dot.edge('answerer', 'qembed')
    dot.edge('qembed', 'search')
    dot.edge('search', 'retrieve')
    dot.edge('retrieve', 'llm')
    dot.edge('llm', 'citations')
    dot.edge('citations', 'display')
    
    # Render
    output_file = 'architecture_flowchart'
    dot.render(output_file, cleanup=True)
    print(f"✅ Graphviz flowchart saved as: {output_file}.png")
    print(f"   Source file: {output_file}.gv")


def generate_matplotlib_flowchart():
    """Generate flowchart using Matplotlib"""
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Skipping Matplotlib flowchart generation.")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 18))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 22)
    ax.axis('off')
    
    # Define colors
    colors = {
        'start_end': '#e3f2fd',
        'decision': '#fff9c4',
        'error': '#ffcdd2',
        'input': '#c8e6c9',
        'indexer': '#e1f5ff',
        'indexer_step': '#b3e5fc',
        'answerer': '#fff4e1',
        'answerer_step': '#ffe0b2',
        'faiss': '#e8f5e9',
        'llm': '#fce4ec',
    }
    
    # Helper function to draw rounded rectangle
    def draw_box(x, y, width, height, text, color, bold=False):
        bbox = FancyBboxPatch((x-width/2, y-height/2), width, height,
                             boxstyle="round,pad=0.1", 
                             facecolor=color, edgecolor='black', linewidth=2 if bold else 1)
        ax.add_patch(bbox)
        fontweight = 'bold' if bold else 'normal'
        ax.text(x, y, text, ha='center', va='center', fontsize=9, fontweight=fontweight, wrap=True)
        return bbox
    
    # Helper function to draw diamond (decision)
    def draw_diamond(x, y, width, height, text, color):
        diamond = mpatches.RegularPolygon((x, y), 4, radius=width/2, 
                                         orientation=0.785, 
                                         facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(diamond)
        ax.text(x, y, text, ha='center', va='center', fontsize=8, wrap=True)
        return diamond
    
    # Helper function to draw ellipse
    def draw_ellipse(x, y, width, height, text, color):
        ellipse = mpatches.Ellipse((x, y), width, height, 
                                   facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(ellipse)
        ax.text(x, y, text, ha='center', va='center', fontsize=9, wrap=True)
        return ellipse
    
    # Helper function to draw arrow
    def draw_arrow(x1, y1, x2, y2, label=''):
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                               arrowstyle='->', mutation_scale=20, 
                               linewidth=1.5, color='black')
        ax.add_patch(arrow)
        if label:
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mid_x, mid_y + 0.3, label, ha='center', fontsize=7, 
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    # Draw flowchart elements
    y_pos = 21
    
    # Start
    draw_ellipse(5, y_pos, 2, 0.6, 'User Starts App', colors['start_end'])
    y_pos -= 1
    draw_arrow(5, y_pos + 0.3, 5, y_pos - 0.3)
    y_pos -= 0.5
    
    # API Check
    draw_diamond(5, y_pos, 1.5, 1.5, 'API Key\nAvailable?', colors['decision'])
    y_pos -= 1.2
    draw_arrow(5, y_pos + 0.75, 5, y_pos - 0.3)
    y_pos -= 0.5
    
    # Input Method
    draw_diamond(5, y_pos, 1.5, 1.5, 'Input Method', colors['decision'])
    y_pos -= 1.2
    
    # File and Text branches
    draw_box(2.5, y_pos, 1.8, 0.8, 'Upload\n.txt/.md File', colors['input'])
    draw_box(7.5, y_pos, 1.8, 0.8, 'Paste Text', colors['input'])
    draw_arrow(4.25, y_pos + 0.75, 2.5, y_pos + 0.4, 'File')
    draw_arrow(5.75, y_pos + 0.75, 7.5, y_pos + 0.4, 'Text')
    y_pos -= 1
    
    # Index Button
    draw_box(5, y_pos, 2, 0.6, 'Click Index Document', colors['indexer_step'])
    draw_arrow(2.5, y_pos + 0.4, 4, y_pos + 0.3)
    draw_arrow(7.5, y_pos + 0.4, 6, y_pos + 0.3)
    y_pos -= 0.8
    
    # Indexer Agent
    draw_box(5, y_pos, 2.5, 0.7, 'Indexer Agent', colors['indexer'], bold=True)
    y_pos -= 1
    
    # Indexer steps
    steps = [
        ('Chunk Text by\nSentences with Overlap', colors['indexer_step']),
        ('Generate Embeddings\n(OpenAI/Gemini)', colors['indexer_step']),
        ('Normalize Embeddings\n(L2 Normalization)', colors['indexer_step']),
        ('Build FAISS Index\n(IndexFlatIP)', colors['faiss']),
        ('Cache Index', colors['indexer_step']),
    ]
    
    for step_text, step_color in steps:
        draw_arrow(5, y_pos + 0.4, 5, y_pos - 0.3)
        y_pos -= 0.5
        draw_box(5, y_pos, 2.5, 0.7, step_text, step_color)
        y_pos -= 0.8
    
    # Index Ready
    draw_arrow(5, y_pos + 0.4, 5, y_pos - 0.3)
    y_pos -= 0.5
    draw_ellipse(5, y_pos, 2, 0.6, 'Index Ready', colors['start_end'])
    y_pos -= 1
    
    # Query
    draw_arrow(5, y_pos + 0.3, 5, y_pos - 0.3)
    y_pos -= 0.5
    draw_diamond(5, y_pos, 1.5, 1.5, 'User Query', colors['decision'])
    y_pos -= 1.2
    
    # Answerer Agent
    draw_box(5, y_pos, 2.5, 0.7, 'Answerer Agent', colors['answerer'], bold=True)
    y_pos -= 1
    
    # Answerer steps
    answerer_steps = [
        ('Embed Query', colors['answerer_step']),
        ('Search FAISS Index\n(Top-K Retrieval)', colors['answerer_step']),
        ('Retrieve Top 4 Chunks\nwith Scores', colors['answerer_step']),
        ('Generate Answer\nwith LLM', colors['llm']),
        ('Add Inline Citations\n[1], [2], etc.', colors['answerer_step']),
    ]
    
    for step_text, step_color in answerer_steps:
        draw_arrow(5, y_pos + 0.4, 5, y_pos - 0.3)
        y_pos -= 0.5
        draw_box(5, y_pos, 2.5, 0.7, step_text, step_color)
        y_pos -= 0.8
    
    # Display Answer
    draw_arrow(5, y_pos + 0.4, 5, y_pos - 0.3)
    y_pos -= 0.5
    draw_ellipse(5, y_pos, 2, 0.6, 'Display Answer', colors['start_end'])
    
    # Title
    ax.text(5, 21.5, 'Mini Doc-FAQ Agent Architecture', 
           ha='center', va='top', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('architecture_flowchart_matplotlib.png', dpi=300, bbox_inches='tight')
    print("✅ Matplotlib flowchart saved as: architecture_flowchart_matplotlib.png")
    plt.close()


if __name__ == "__main__":
    print("Generating architecture flowcharts...")
    print("-" * 50)
    
    if GRAPHVIZ_AVAILABLE:
        print("\n1. Generating Graphviz flowchart...")
        generate_graphviz_flowchart()
    else:
        print("\n1. Graphviz not available. Install with:")
        print("   pip install graphviz")
        print("   And install system package: https://graphviz.org/download/")
    
    if MATPLOTLIB_AVAILABLE:
        print("\n2. Generating Matplotlib flowchart...")
        generate_matplotlib_flowchart()
    else:
        print("\n2. Matplotlib not available. Install with:")
        print("   pip install matplotlib")
    
    print("\n" + "-" * 50)
    print("Done! Check the generated PNG files.")

