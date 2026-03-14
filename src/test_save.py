import os



# File path
file_path = "Experiments/sample_test.txt"

# Write a sample file
with open(file_path, "w") as f:
    f.write("This is a test file to check saving in the Experiments folder.\n")

print("Sample file saved successfully at:", file_path)