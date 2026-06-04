import graphviz
import os

# Create dot object
dot = graphviz.Digraph('cohort', format='png')
dot.attr(rankdir='TB', size='6,8')

# Graph properties
dot.attr('node', shape='box', style='rounded,filled', fillcolor='white', fontname='Helvetica', margin='0.3')
dot.attr('edge', fontname='Helvetica', fontsize='10')

# Nodes
dot.node('A', 'MIMIC-IV\nICU Stays\n(n ≈ 76,000)')
dot.node('B', 'Adult Patients\n(Age ≥ 18)\n(n ≈ 68,000)')
dot.node('C', 'Sepsis-3 Criteria Met\n(n ≈ 35,000)')
dot.node('D', '≥4 Timesteps Complete\nFinal Cohort\n(n = 25,847)')

# Exclusion nodes
dot.attr('node', fillcolor='#ffdddd', shape='box', style='filled')
dot.node('E1', 'Excluded:\nAge < 18')
dot.node('E2', 'Excluded:\nNo Sepsis-3')
dot.node('E3', 'Excluded:\nShort Stays (< 4 hours)')

# Split nodes
dot.attr('node', fillcolor='#ddffdd', shape='box', style='rounded,filled')
dot.node('S1', 'Train Split (80%)\nn = 20,677')
dot.node('S2', 'Validation Split (10%)\nn = 2,585')
dot.node('S3', 'Held-out Test (10%)\nn = 2,585')

# Edges
dot.edge('A', 'B')
dot.edge('A', 'E1')
dot.edge('B', 'C')
dot.edge('B', 'E2')
dot.edge('C', 'D')
dot.edge('C', 'E3')

# Final split edges
dot.edge('D', 'S1')
dot.edge('D', 'S2')
dot.edge('D', 'S3')

# Render
output_path = 'report/figures/fig1_cohort_flow'
dot.render(output_path, cleanup=True)
print(f"Generated {output_path}.png")