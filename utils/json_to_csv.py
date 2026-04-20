import json
import csv
import sys
import os
import argparse

def extract_letter(text, index2ans):
    """Extracts the choice letter (A, B, C, or D) from text or reverse-lookups in index2ans."""
    if not text:
        return None
    
    # 1. Try reverse lookup in index2ans (label -> letter)
    for letter, ans in index2ans.items():
        if ans.strip().lower() == text.strip().lower():
            return letter
            
    # 2. Check if it's already just a letter
    clean_text = text.strip().upper()
    if clean_text in ['A', 'B', 'C', 'D']:
        return clean_text
        
    # 3. Check for format "A) Text"
    if len(clean_text) > 1 and clean_text[1] == ')' and clean_text[0] in ['A', 'B', 'C', 'D']:
        return clean_text[0]
        
    # 4. Check for format "(A) Text"
    if len(clean_text) > 2 and clean_text[0] == '(' and clean_text[1] in ['A', 'B', 'C', 'D']:
        return clean_text[1]
        
    return None

def process_file(file_path):
    if not file_path.endswith('.json'):
        print(f"Error: {file_path} is not a JSON file.")
        return

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading JSON: {e}")
        return
    
    # Collect all unique models across all entries to ensure dynamic column handling
    all_model_identifiers = []
    for entry in data:
        for output in entry.get('model_outputs', []):
            model_id = output['model']
            if model_id not in all_model_identifiers:
                all_model_identifiers.append(model_id)
    
    # Sort for consistent column ordering
    all_model_identifiers.sort()
    
    # Prepare CSV headers
    headers = ['No', 'img_path', 'Question', 'Right Answer']
    
    # Headers for each model
    for model_id in all_model_identifiers:
        # Sanitize model name for header (e.g., 'MahmoodLab/conch' -> 'conch')
        m_name = model_id.split('/')[-1]
        headers.append(f'{m_name}_Correct')
        for choice in ['A', 'B', 'C', 'D']:
            headers.append(f'{m_name}_Prob_{choice}')
    
    # Headers for sums and judge
    for choice in ['A', 'B', 'C', 'D']:
        headers.append(f'Sum_Prob_{choice}')
    headers.append('Judge Answer')
    
    output_csv = file_path.rsplit('.', 1)[0] + '.csv'
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        
        for entry in data:
            index2ans = entry.get('index2ans', {})
            ans2index = {v.strip().lower(): k for k, v in index2ans.items()}
            
            gt_letter = extract_letter(entry.get('answer'), index2ans)
            
            row = {
                'No': entry.get('No'),
                'img_path': entry.get('img_path'),
                'Question': entry.get('question'),
                'Right Answer': gt_letter
            }
            
            sums = {'A': 0.0, 'B': 0.0, 'C': 0.0, 'D': 0.0}
            
            # Map model_outputs for easy lookup
            model_outputs_map = {m['model']: m for m in entry.get('model_outputs', [])}
            
            for model_id in all_model_identifiers:
                m_name = model_id.split('/')[-1]
                m_out = model_outputs_map.get(model_id)
                
                if m_out:
                    # Extract probabilities for A, B, C, D
                    probs = {'A': 0.0, 'B': 0.0, 'C': 0.0, 'D': 0.0}
                    for item in m_out.get('top5', []):
                        label = item.get('label', '').strip().lower()
                        prob = item.get('prob', 0.0)
                        choice_letter = ans2index.get(label)
                        if choice_letter in probs:
                            probs[choice_letter] = prob
                    
                    # Determine if correct (T/F)
                    # Use top1 text and map it to a letter
                    top1_text = m_out.get('top1', '').strip().lower()
                    top1_letter = ans2index.get(top1_text)
                    
                    row[f'{m_name}_Correct'] = 'T' if top1_letter == gt_letter else 'F'
                    
                    for choice in ['A', 'B', 'C', 'D']:
                        row[f'{m_name}_Prob_{choice}'] = probs[choice]
                        sums[choice] += probs[choice]
                else:
                    # Model missing for this entry
                    row[f'{m_name}_Correct'] = 'N/A'
                    for choice in ['A', 'B', 'C', 'D']:
                        row[f'{m_name}_Prob_{choice}'] = 0.0

            # Add sums
            for choice in ['A', 'B', 'C', 'D']:
                row[f'Sum_Prob_{choice}'] = round(sums[choice], 4)
            
            # Extract Judge Answer
            # First try pred_ans text lookup
            judge_letter = extract_letter(entry.get('pred_ans'), index2ans)
            # Fallback: try to extract letter from start of response if judge_letter is still None
            if not judge_letter and entry.get('response'):
                judge_letter = extract_letter(entry.get('response').split('\n')[0], index2ans)
            
            row['Judge Answer'] = judge_letter
            
            writer.writerow(row)

    print(f"Successfully generated: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a CSV dataset from a PathMMU evaluation JSON file.')
    parser.add_argument('file', help='Path to the JSON output file')
    args = parser.parse_args()
    
    if os.path.exists(args.file):
        process_file(args.file)
    else:
        print(f"Error: File not found at {args.file}")
