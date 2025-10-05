import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import json
import tempfile
import os
from datetime import datetime

class NotebookExecutor:
    def __init__(self):
        self.ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    
    def create_rainfall_analysis_notebook(self, rainfall_data, location_name="Unknown"):
        """Create a dynamic notebook for rainfall analysis"""
        nb = nbformat.v4.new_notebook()
        
        # Add markdown cell with title
        title_cell = nbformat.v4.new_markdown_cell(
            f"# Rainfall Analysis for {location_name}\n"
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        nb.cells.append(title_cell)
        
        # Add code cell for imports
        imports_code = """
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")
"""
        imports_cell = nbformat.v4.new_code_cell(imports_code)
        nb.cells.append(imports_cell)
        
        # Add code cell for data loading
        data_code = f"""
# Rainfall data
rainfall_data = {json.dumps(rainfall_data, indent=2)}

# Convert to DataFrame
df = pd.DataFrame(rainfall_data['history'])
df['date'] = pd.to_datetime(df['date'])
df = df.dropna(subset=['rainfall_mm'])  # Remove null values
df['year'] = df['date'].dt.year

print(f"Dataset contains {{len(df)}} valid data points")
print(f"Average rainfall: {{rainfall_data['average_rainfall_mm']}} mm")
print(f"Category: {{rainfall_data['category']}}")
"""
        data_cell = nbformat.v4.new_code_cell(data_code)
        nb.cells.append(data_cell)
        
        # Add visualization cells
        viz_code = """
# Create visualizations
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# 1. Time series plot
ax1.plot(df['date'], df['rainfall_mm'], marker='o', linewidth=2, markersize=6)
ax1.set_title('Rainfall Over Time', fontsize=14, fontweight='bold')
ax1.set_xlabel('Year')
ax1.set_ylabel('Rainfall (mm)')
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

# 2. Histogram
ax2.hist(df['rainfall_mm'], bins=10, alpha=0.7, color='skyblue', edgecolor='black')
ax2.set_title('Rainfall Distribution', fontsize=14, fontweight='bold')
ax2.set_xlabel('Rainfall (mm)')
ax2.set_ylabel('Frequency')
ax2.grid(True, alpha=0.3)

# 3. Box plot by year
if len(df) > 5:  # Only if we have enough data
    sns.boxplot(data=df, y='rainfall_mm', ax=ax3)
    ax3.set_title('Rainfall Distribution (Box Plot)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Rainfall (mm)')
else:
    ax3.text(0.5, 0.5, 'Insufficient data for box plot', 
             ha='center', va='center', transform=ax3.transAxes)
    ax3.set_title('Box Plot (Insufficient Data)', fontsize=14, fontweight='bold')

# 4. Statistical summary
stats_text = f'''Statistical Summary:
Mean: {df['rainfall_mm'].mean():.2f} mm
Median: {df['rainfall_mm'].median():.2f} mm
Std Dev: {df['rainfall_mm'].std():.2f} mm
Min: {df['rainfall_mm'].min():.2f} mm
Max: {df['rainfall_mm'].max():.2f} mm
Range: {df['rainfall_mm'].max() - df['rainfall_mm'].min():.2f} mm'''

ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=12,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)
ax4.axis('off')
ax4.set_title('Statistics', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

# Additional analysis
print("\\n=== DETAILED ANALYSIS ===")
print(f"Data points collected: {len(df)}")
print(f"Years covered: {df['year'].min()} to {df['year'].max()}")
print(f"Driest year: {df.loc[df['rainfall_mm'].idxmin(), 'year']} ({df['rainfall_mm'].min():.2f} mm)")
print(f"Wettest year: {df.loc[df['rainfall_mm'].idxmax(), 'year']} ({df['rainfall_mm'].max():.2f} mm)")
"""
        viz_cell = nbformat.v4.new_code_cell(viz_code)
        nb.cells.append(viz_cell)
        
        return nb
    
    def execute_notebook(self, notebook):
        """Execute a notebook and return the result"""
        try:
            # Execute the notebook
            self.ep.preprocess(notebook, {'metadata': {'path': './'}})
            return notebook, None
        except Exception as e:
            return None, str(e)
    
    def notebook_to_html(self, notebook):
        """Convert notebook to HTML"""
        from nbconvert import HTMLExporter
        html_exporter = HTMLExporter()
        html_exporter.template_name = 'classic'
        (body, resources) = html_exporter.from_notebook_node(notebook)
        return body
    
    def save_notebook(self, notebook, filename):
        """Save notebook to file"""
        with open(filename, 'w') as f:
            nbformat.write(notebook, f)