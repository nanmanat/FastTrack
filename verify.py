import os
import xml.etree.ElementTree as ET

def count_word_in_xml(file_path, word):
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()

        # Convert the XML content to string
        xml_str = ET.tostring(root, encoding='utf-8').decode('utf-8')

        # Count occurrences of the specific word
        word_count = xml_str.count(word)

        return word_count / 2

    except ET.ParseError as e:
        print(f"Error parsing XML: {e}")
        return 0
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return 0

def process_directory(directory_path, word, output_file):
    results = []

    for filename in os.listdir(directory_path):
        if filename.endswith(".xml"):
            file_path = os.path.join(directory_path, filename)
            count = count_word_in_xml(file_path, word)
            jpg_filename = filename.replace(".xml", ".jpg")
            results.append((jpg_filename, count))

    # Sort results by file name
    results.sort()

    with open(output_file, 'w') as file:
        for jpg_filename, count in results:
            file.write(f"({jpg_filename}, {count})\n")

def compare_counts(file1, file2, output_file):
    counts1 = {}
    counts2 = {}

    # Read the first file and store counts in a dictionary
    with open(file1, 'r') as f:
        for line in f:
            jpg_filename, count = line.strip('()\n').split(', ')
            counts1[jpg_filename] = float(count)

    # Read the second file and store counts in a dictionary
    with open(file2, 'r') as f:
        for line in f:
            jpg_filename, _, count = line.strip('()\n').split('  ')
            counts2[jpg_filename] = float(count)

    mismatch_count = 0
    total_count = len(counts1)

    # Compare counts and write mismatches to the output file
    with open(output_file, 'w') as file:
        for jpg_filename in counts1:
            if jpg_filename in counts2 and counts1[jpg_filename] != counts2[jpg_filename]:
                mismatch_count += 1
                file.write(f"{jpg_filename} - {counts1[jpg_filename]} vs {counts2[jpg_filename]}\n")

    mismatch_percentage = (mismatch_count / total_count) * 100 if total_count > 0 else 0
    print(f"Mismatch percentage: {mismatch_percentage:.2f}%")

if __name__ == "__main__":
    directory_path = "Annotations"  # Replace with your directory path containing XML files
    word = "object"       # Replace with the word you want to count
    output_file = "results.txt"  # Replace with your desired output file path
    comparison_file = "ImageSets/Main/NNEW_test_4.txt"  # Replace with the path to the comparison file
    mismatch_output_file = "mismatches_test_4.txt"  # Replace with your desired output file path for mismatches


    process_directory(directory_path, word, output_file)
    print(f"Results saved to {output_file}")

    compare_counts(output_file, comparison_file, mismatch_output_file)
    print(f"Mismatches saved to {mismatch_output_file}")
