import re

def natural_sort_key(s):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]

# Read the contents of both files
with open('train.txt', 'r') as f1, open('val.txt', 'r') as f2:
    lines1 = f1.readlines()
    lines2 = f2.readlines()

# Combine the lines from both files
all_lines = lines1 + lines2

# Sort the combined lines
sorted_lines = sorted(all_lines, key=lambda x: natural_sort_key(x.split()[0]))

# Write the sorted lines to a new file
with open('merged_sorted.txt', 'w') as outfile:
    outfile.writelines(sorted_lines)

print("Files merged and sorted. Result saved in 'merged_sorted.txt'.")
