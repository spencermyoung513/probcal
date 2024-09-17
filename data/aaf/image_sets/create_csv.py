import pandas as pd

def parse_line(line):
    filename, gender = line.strip().split()
    id = filename
    age = filename[6:8]
    age_stripped = str(int(age))
    return id, age_stripped, gender

# Read the input file
input_file = 'picture_data.txt'
with open(input_file, 'r') as file:
    lines = file.readlines()

# Parse the data
parsed_data = [parse_line(line) for line in lines]

# Create a DataFrame
columns = ['id', 'age', 'gender']
df = pd.DataFrame(parsed_data, columns=columns)

# Write to CSV
output_file = 'picture_data.csv'
df.to_csv(output_file, index=False)

print(f"Conversion complete. Output saved to {output_file}")
