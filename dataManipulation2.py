# Open the input file
input_file_path = 'chatDataSet.txt'  # Replace with your dataset file path
output_file_path = 'chatDataSet2.txt'  # Replace with the desired output file path
encoding = 'utf-8'  # Change the encoding to match your dataset if needed

with open(input_file_path, 'r', encoding=encoding) as input_file:
    # Read lines from the input file
    lines = input_file.readlines()

# Open the output file
with open(output_file_path, 'w', encoding=encoding) as output_file:
    # Add a comma at the end of each line and write to the output file
    for line in lines:
        line_with_comma = line.rstrip() + ','  # Remove trailing whitespaces and add a comma
        output_file.write(line_with_comma + '\n')  # Write the modified line to the output file

print(f"Commas added to each line. Output saved to: {output_file_path}")