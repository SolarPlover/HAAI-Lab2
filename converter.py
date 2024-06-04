def read_weights_matrices(filename):
    matrices = []
    with open(filename, 'r') as file:
        lines = file.readlines()
        matrix_name = None
        matrix_values = []
        for line in lines:
            if line.startswith("fc"):
                # New matrix name found, save the previous matrix
                if matrix_name and matrix_values:
                    matrices.append((matrix_name, matrix_values))
                matrix_name = line.strip()
                matrix_values = []
            else:
                # Read matrix values (convert to 0 or 1)
                row_values = [int(float(val)) for val in line.split()]
                matrix_values.append(row_values)
        # Save the last matrix
        if matrix_name and matrix_values:
            matrices.append((matrix_name, matrix_values))
    return matrices

def convert_to_2d_array(matrix_values):
    return [['1' if val == 1 else '0' for val in row] for row in matrix_values]

def write_to_text_file(matrices, output_filename):
    with open(output_filename, 'w') as output_file:
        for matrix_name, matrix_values in matrices:
            output_file.write(f"Matrix {matrix_name}:\n")
            for row in convert_to_2d_array(matrix_values):
                output_file.write("{" + ", ".join(row) + "},\n")

def main():
    input_filename = 'model.txt'  # Replace with your actual input file name
    output_filename = 'converted_weights.txt'  # Replace with your desired output file name
    matrices = read_weights_matrices(input_filename)
    write_to_text_file(matrices, output_filename)
    print(f"Converted matrices saved to {output_filename}")

if __name__ == "__main__":
    main()