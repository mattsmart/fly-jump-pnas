import os


# Directory structure for the project
DIR_DATA_EXPT   = '..' + os.sep + 'data' + os.sep + 'experiment'
DIR_FITS        = '..' + os.sep + 'fits'
DIR_OUTPUT      = '..' + os.sep + 'output'
DIR_STAN        = '..' + os.sep + 'stan'

for dir in [DIR_DATA_EXPT, DIR_FITS, DIR_OUTPUT, DIR_STAN]:
    if not os.path.exists(dir):
        os.makedirs(dir)

# Plot settings
days_palettes = {
        "Muted": ["#4C72B0", "#55A868", "#C44E52"],            # Professional, subtle
        "Vibrant": ["#E41A1C", "#377EB8", "#4DAF4A"],          # Bold, high-contrast
        "Deep": ["#1B9E77", "#7570B3", "#D95F02"],             # Deep hues
        "Earthy": ["#8C510A", "#BF812D", "#DFC27D"],           # Earth tones
        "Ocean": ["#1B4F72", "#2980B9", "#76D7C4"],            # Cool, oceanic tones
        "Sunset": ["#FF4500", "#FF8C00", "#FFD700"],           # Warm, gradient tones
        "Neon": ["#39FF14", "#FF073A", "#08F7FE"],             # High contrast neon
        "Minimal": ["#999999", "#666666", "#333333"],          # Grayscale minimalism
        "Pastel": ["#AEC6CF", "#FFB347", "#FF6961"],           # Soft pastel tones
        "Dark": ["#2C2C54", "#474787", "#AAA69D"]              # Darker, muted tones
    }
day_palette = days_palettes['Deep']    # Deep, Vibrant, Muted OK (contrast RGB);  try Ocean or Dark for similar cmaps

# Colors for jump heatmaps: originally "viridis"
# - viridis limits: 0.0 (purple -> #440154) to 1.0 (yellow -> #fde725)
heatmap_nodata = "whitesmoke"  # no data (default: whitesmoke)
heatmap_0 = "#440154"          # no jump (default: darkblue)
heatmap_1 = "#fde725"          # jump    (default: orange)

# Flies to omit from analysis due to technical issues (see README lines 56-73)
# - KK_a7, fly_id 31:  extreme cut_power values
# - KK_a7, fly_id 121: presumed dead (already removed from raw data)
# - KK_a14, fly_id 80: extreme low delay-jump values
# - KK_a14, fly_id 76: extreme low cut_power values
OMIT_FLY_IDS = {
    'KK': [31, 121, 80, 76],
    'GD': []
}
