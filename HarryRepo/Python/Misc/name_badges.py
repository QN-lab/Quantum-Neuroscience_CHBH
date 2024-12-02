import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as patches
import math
from matplotlib.backends.backend_pdf import PdfPages

# Function to wrap text
def wrap_text(text, ax, x, y, max_width, fontsize, line_spacing=0.15, weight='normal'):
    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        # Test if adding the next word would exceed max_width
        test_line = f"{current_line} {word}".strip() if current_line else word
        
        # Create a temporary text object for measurement
        text_obj = ax.text(x, y, test_line, fontsize=fontsize, alpha=0, weight=weight)  # Invisible text for measuring
        plt.draw()  # Ensure the figure is rendered
        # Get the width of the current line
        if text_obj.get_window_extent().width / fig.dpi < max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word
        # Remove the temporary text object
        text_obj.remove()

    # Add the last line
    lines.append(current_line)

    # Draw the text on the plot with specified line spacing
    for i, line in enumerate(lines):
        ax.text(x, y - i * line_spacing, line, ha='center', va='top', fontsize=fontsize, weight=weight)

# Load CSV file
data = pd.read_csv(r'W:\Presentations\H\MEGUKI\Names_affils_1.csv', encoding='ISO-8859-1')  # Adjust filename as necessary

# Define the output PDF file path
output_pdf_path = r'W:\Presentations\H\Name_Badges_All_Pages.pdf'

# Define A4 dimensions in inches (8.27 x 11.69 inches)
page_width, page_height = 8.27, 11.69  

# Define badge dimensions in inches (54mm x 90mm is approx. 2.13 x 3.54 inches)
badge_width, badge_height = 90 / 25.4, 54 / 25.4

# Calculate rows and columns that fit A4 dimensions
cols = int(page_width // badge_width)
rows = int(page_height // badge_height)
badges_per_page = rows * cols

# Calculate the number of pages required
num_pages = math.ceil(len(data) / badges_per_page)

# Create a PdfPages object to manage the multi-page PDF
with PdfPages(output_pdf_path) as pdf:
    for page_num in range(num_pages):
        # Create a new figure for each page
        fig, ax = plt.subplots(figsize=(page_width, page_height))
        ax.set_xlim(0, page_width)
        ax.set_ylim(0, page_height)
        ax.axis('off')

        # Ensure the aspect ratio is equal
        ax.set_aspect('equal', adjustable='box')

        # Get the subset of badges for the current page
        start_idx = page_num * badges_per_page
        end_idx = min(start_idx + badges_per_page, len(data))
        page_data = data.iloc[start_idx:end_idx]
        
        # Add badges to each cell on the current page
        for idx, (first_name, last_name, organization) in enumerate(zip(page_data['First Name'], page_data['Last Name'], page_data['Organization'])):
            col = idx % cols
            row = idx // cols

            # Calculate badge position
            x = col * badge_width
            y = page_height - (row + 1) * badge_height  # Start from top

            # Draw badge background with white color (no border)
            ax.add_patch(patches.Rectangle((x, y), badge_width, badge_height, edgecolor='none', facecolor='white'))

            # Add "QMAG" text to the top left of the badge
            qmag_x = x + badge_width * 0.02  # Move more to the left
            qmag_y = y + badge_height * 0.95  # Near the top
            ax.text(qmag_x, qmag_y, "QMAG", fontsize=12, ha='left', va='top')  # Smaller font size

            # Combine first and last names
            full_name = f"{first_name} {last_name}"

            # Set vertical position for the name and organization
            name_y = y + badge_height * 0.65  # Move name slightly up
            org_y = y + badge_height * 0.35   # Position organization a little lower

            # Wrap the name text in bold
            wrap_text(full_name, ax, x + badge_width / 2, name_y, badge_width * 0.8, fontsize=12, line_spacing=0.25, weight='bold')

            # Wrap the organization text
            wrap_text(organization, ax, x + badge_width / 2, org_y, badge_width * 0.8, fontsize=10, line_spacing=0.25)

        # Save the figure with tight bounding box to avoid scaling issues
        pdf.savefig(fig, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

print(f"All pages of name badges have been saved to {output_pdf_path}.")
