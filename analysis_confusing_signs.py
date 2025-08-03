import argparse
import csv
import os
from collections import Counter, defaultdict
import itertools

# Parse arguments
parser = argparse.ArgumentParser(description='Detailed misclassification analysis with three criteria')
parser.add_argument('--input', type=str, required=True, help='Path to test_results.csv')
args = parser.parse_args()

input_file = args.input
criteria1_file = './classification_confusing_signs.txt'
criteria2_file = './classifier_movement.txt'
output_dir = os.path.dirname(input_file)

# Parse criteria 1: signing space and hands involved (per label)
sign_to_criteria1 = {}
with open(criteria1_file, 'r') as f:
    for line in f:
        line = line.strip()
        if not line or ':' not in line:
            continue
        sign, crits = line.split(':', 1)
        crits = [c.strip() for c in crits.split(',')]
        if len(crits) == 2:
            sign_to_criteria1[sign.strip()] = (crits[0], crits[1])

# Parse criteria 2: differences (per pair/triple)
pair_to_criteria2 = {}
with open(criteria2_file, 'r') as f:
    for line in f:
        line = line.strip()
        if not line or ':' not in line:
            continue
        labels_part, crits_part = line.split(':', 1)
        labels = [label.strip() for label in labels_part.split('-')]
        crits = [c.strip() for c in crits_part.split(',')]
        # Create all possible pairs from the labels
        for pair in itertools.combinations(labels, 2):
            pair_key = tuple(sorted(pair))
            if pair_key not in pair_to_criteria2:
                pair_to_criteria2[pair_key] = set()
            pair_to_criteria2[pair_key].update(crits)

# Read test results
pairs = []
with open(input_file, 'r') as f:
    reader = csv.reader(f)
    header = next(reader, None)  # Skip header
    for row in reader:
        if len(row) < 2:
            continue
        true_label, pred_label = row[0].strip(), row[1].strip()
        pairs.append((true_label, pred_label))

total = len(pairs)
mistakes = [(t, p) for t, p in pairs if t != p]
num_mistakes = len(mistakes)
percent_mistakes = 100.0 * num_mistakes / total if total else 0.0

# Analysis for Criteria 1: Signing Space
signing_space_values = ['upper_face', 'lower_face', 'body']
signing_space_mistakes = Counter()
signing_space_total = Counter()

for t, p in pairs:
    if t in sign_to_criteria1:
        signing_space = sign_to_criteria1[t][0]
        signing_space_total[signing_space] += 1
        if t != p:
            signing_space_mistakes[signing_space] += 1

# Analysis for Criteria 1: Hands Involved
hands_involved_values = ['one_handed', 'two_handed_symmetric', 'two_handed_asymmetric']
hands_involved_mistakes = Counter()
hands_involved_total = Counter()

for t, p in pairs:
    if t in sign_to_criteria1:
        hands_involved = sign_to_criteria1[t][1]
        hands_involved_total[hands_involved] += 1
        if t != p:
            hands_involved_mistakes[hands_involved] += 1

# Analysis for Criteria 2: Differences
difference_values = ['movement', 'hand_shape', 'palm_orientation', 'location', 'facial_expressions']
difference_mistakes = Counter()
difference_total = Counter()

# Count mistakes for each difference category
for t, p in mistakes:
    # Check if this pair exists in our criteria2 mapping
    pair_key = tuple(sorted([t, p]))
    if pair_key in pair_to_criteria2:
        for diff_cat in pair_to_criteria2[pair_key]:
            difference_mistakes[diff_cat] += 1

# Count total occurrences for each difference category
for t, p in pairs:
    pair_key = tuple(sorted([t, p]))
    if pair_key in pair_to_criteria2:
        for diff_cat in pair_to_criteria2[pair_key]:
            difference_total[diff_cat] += 1

# Get top 5 misclassified signs per signing space
signing_space_top5 = {}
for space in signing_space_values:
    space_signs = [sign for sign in sign_to_criteria1.keys() if sign_to_criteria1[sign][0] == space]
    space_mistakes = Counter()
    for t, p in mistakes:
        if t in space_signs:
            space_mistakes[t] += 1
    signing_space_top5[space] = space_mistakes.most_common(5)

# Get top 5 misclassified signs per hands involved
hands_involved_top5 = {}
for hands in hands_involved_values:
    hands_signs = [sign for sign in sign_to_criteria1.keys() if sign_to_criteria1[sign][1] == hands]
    hands_mistakes = Counter()
    for t, p in mistakes:
        if t in hands_signs:
            hands_mistakes[t] += 1
    hands_involved_top5[hands] = hands_mistakes.most_common(5)

# Get top 5 misclassified pairs per difference category
difference_top5 = {}
for diff_cat in difference_values:
    diff_pairs = Counter()
    for t, p in mistakes:
        pair_key = tuple(sorted([t, p]))
        if pair_key in pair_to_criteria2 and diff_cat in pair_to_criteria2[pair_key]:
            diff_pairs[f"{t} - {p}"] += 1
    difference_top5[diff_cat] = diff_pairs.most_common(5)

# Find top 10 misclassified label pairs/triplets
pair_mistakes = Counter()
for t, p in mistakes:
    pair_key = tuple(sorted([t, p]))
    if pair_key in pair_to_criteria2:
        pair_mistakes[pair_key] += 1

top10_pairs = pair_mistakes.most_common(10)

# Write analysis to file
analysis_file = os.path.join(output_dir, 'detailed_analysis.txt')
with open(analysis_file, 'w') as f:
    f.write(f"Total Number of mistakes = {num_mistakes} / {total} = {percent_mistakes:.2f}%\n\n")
    
    # First Criteria: Signing Space
    f.write("First criteria: signing space:\n\n")
    for space in signing_space_values:
        mistakes_count = signing_space_mistakes[space]
        total_count = signing_space_total[space]
        percent = 100.0 * mistakes_count / total_count if total_count else 0.0
        f.write(f"{space}: {mistakes_count} / {total_count} = {percent:.2f}%\n")
    
    f.write("\nTop 5 misclassification signs per each subcriteria:\n\n")
    for space in signing_space_values:
        f.write(f"{space}:\n")
        for sign, count in signing_space_top5[space]:
            total_appearances = sum(1 for t, p in pairs if t == sign)
            percent = 100.0 * count / total_appearances if total_appearances else 0.0
            f.write(f"  {sign} [{count}/{total_appearances} = {percent:.2f}%]: ")
            
            # Get what this sign was misclassified as
            sign_mistakes = Counter()
            for t, p in mistakes:
                if t == sign:
                    sign_mistakes[p] += 1
            
            # Show top misclassifications
            top_mistakes = sign_mistakes.most_common(5)
            mistake_strs = [f"{p} ({c})" for p, c in top_mistakes]
            f.write(", ".join(mistake_strs) + "\n")
        f.write("\n")
    
    # Second Criteria: Hands Involved
    f.write("Second criteria: no. of hands involved in signing:\n\n")
    for hands in hands_involved_values:
        mistakes_count = hands_involved_mistakes[hands]
        total_count = hands_involved_total[hands]
        percent = 100.0 * mistakes_count / total_count if total_count else 0.0
        f.write(f"{hands}: {mistakes_count} / {total_count} = {percent:.2f}%\n")
    
    f.write("\nTop 5 misclassification signs per each subcriteria:\n\n")
    for hands in hands_involved_values:
        f.write(f"{hands}:\n")
        for sign, count in hands_involved_top5[hands]:
            total_appearances = sum(1 for t, p in pairs if t == sign)
            percent = 100.0 * count / total_appearances if total_appearances else 0.0
            f.write(f"  {sign} [{count}/{total_appearances} = {percent:.2f}%]: ")
            
            # Get what this sign was misclassified as
            sign_mistakes = Counter()
            for t, p in mistakes:
                if t == sign:
                    sign_mistakes[p] += 1
            
            # Show top misclassifications
            top_mistakes = sign_mistakes.most_common(5)
            mistake_strs = [f"{p} ({c})" for p, c in top_mistakes]
            f.write(", ".join(mistake_strs) + "\n")
        f.write("\n")
    
    # Third Criteria: Differences
    f.write("Third criteria: Differences in signing:\n\n")
    for diff_cat in difference_values:
        mistakes_count = difference_mistakes[diff_cat]
        total_count = difference_total[diff_cat]
        percent = 100.0 * mistakes_count / total_count if total_count else 0.0
        f.write(f"{diff_cat}: {mistakes_count} / {total_count} = {percent:.2f}%\n")
    
    f.write("\nTop 5 per each category:\n\n")
    for diff_cat in difference_values:
        f.write(f"{diff_cat}:\n")
        for pair_str, count in difference_top5[diff_cat]:
            f.write(f"  {pair_str}: {count} times\n")
        f.write("\n")
    
    # Top 10 misclassified label pairs/triplets
    f.write("Top 10 misclassified label pairs/triplets as each other:\n\n")
    for i, (pair, count) in enumerate(top10_pairs, 1):
        f.write(f"{i}. {pair[0]} - {pair[1]}: {count} times\n")
        # Get categories for this pair
        if pair in pair_to_criteria2:
            categories = ", ".join(sorted(pair_to_criteria2[pair]))
            f.write(f"   Categories: {categories}\n")
        
        # Show individual mistake breakdown for each label in the pair
        for label in pair:
            label_mistakes = Counter()
            for t, p in mistakes:
                if t == label:
                    label_mistakes[p] += 1
            
            total_appearances = sum(1 for t, p in pairs if t == label)
            percent = 100.0 * sum(label_mistakes.values()) / total_appearances if total_appearances else 0.0
            
            f.write(f"   {label} [{sum(label_mistakes.values())}/{total_appearances} = {percent:.2f}%]: ")
            top_mistakes = label_mistakes.most_common(5)
            mistake_strs = [f"{p} ({c})" for p, c in top_mistakes]
            f.write(", ".join(mistake_strs) + "\n")
        f.write("\n")

print(f"Total mistakes: {num_mistakes} / {total} ({percent_mistakes:.2f}%)")
print(f"Detailed analysis written to {analysis_file}") 