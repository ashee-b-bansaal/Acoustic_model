import csv
import os
import argparse

# Usage: python analysis.py --input /path/to/test_results.csv

# Mapping from index to label
index_to_label = {
    {0: '30', 1: 'address', 2: 'again', 3: 'alone', 4: 'always', 5: 'apple', 6: 'bad', 7: 'bath', 8: 'bathroom', 9: 'bicycle', 10: 'birthday', 11: 'black', 12: 'bread', 13: 'breath', 14: 'business', 15: 'busy', 16: 'buy', 17: 'celebrate', 18: 'cereal', 19: 'chocolate', 20: 'church', 21: 'class', 22: 'cleaning', 23: 'computer', 24: 'cry', 25: 'dance', 26: 'date(romantic_outing)', 27: 'dessert', 28: 'dirty', 29: 'discuss', 30: 'doctor', 31: "doesn't_matter", 32: 'dry', 33: 'each_other', 34: 'family', 35: 'favorite', 36: 'first', 37: 'football', 38: 'frog', 39: 'frustrated', 40: 'gain', 41: 'good', 42: 'gray', 43: 'gym', 44: 'important', 45: 'late', 46: 'live', 47: 'lonely', 48: 'lucky', 49: 'machine_factory', 50: 'mean(cruel)', 51: 'medium_average', 52: 'metal', 53: 'mine', 54: 'money', 55: 'movie', 56: 'music', 57: 'nice', 58: 'no', 59: 'normal', 60: 'not', 61: 'not_yet', 62: 'of_course', 63: 'onion', 64: 'only_just', 65: 'paper', 66: 'pet', 67: 'pig', 68: 'please', 69: 'punish', 70: 'read', 71: 'red', 72: 'roomate', 73: 'scared', 74: 'science', 75: 'sell', 76: 'sex', 77: 'share', 78: 'shave', 79: 'shool', 80: 'shopping', 81: 'show', 82: 'sick', 83: 'single', 84: 'sneakers_rubber', 85: 'socks', 86: 'sometime', 87: 'sorrowful', 88: 'sorry', 89: 'star', 90: 'stay', 91: 'store', 92: 'summer', 93: 'summon', 94: 'sunday', 95: 'sweet', 96: 'today', 97: 'tomorrow', 98: 'tuesday', 99: 'ugly', 100: 'warning', 101: 'weekend', 102: 'where', 103: 'wonderful', 104: 'wood', 105: 'worried', 106: 'wrestling', 107: 'your'}
}

parser = argparse.ArgumentParser(description='Convert index CSV to label CSV')
parser.add_argument('--input', type=str, required=True, help='Path to test_results.csv')
args = parser.parse_args()

input_file = args.input
output_file = os.path.join(os.path.dirname(input_file), 'test_results_analysis.csv')

with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    header = next(reader, None)
    # Write header as is, or skip it if not present
    if header and not header[0].isdigit():
        writer.writerow(header)
    else:
        # If no header, process the first row
        label_row = [index_to_label.get(int(idx), idx) for idx in header]
        writer.writerow(label_row)
    for row in reader:
        label_row = [index_to_label.get(int(idx), idx) for idx in row]
        writer.writerow(label_row)

print(f"Labelled results written to {output_file}") 