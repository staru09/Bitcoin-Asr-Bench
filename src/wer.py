import argparse
import re
import string
import sys
from typing import List, Tuple
import difflib
from collections import Counter

class WERCalculator:
    def __init__(self, ignore_case=True, ignore_punctuation=True, 
                 ignore_extra_spaces=True, remove_filler_words=True):
        self.ignore_case = ignore_case
        self.ignore_punctuation = ignore_punctuation
        self.ignore_extra_spaces = ignore_extra_spaces
        self.remove_filler_words = remove_filler_words
        
        self.filler_words = {
            'um', 'uh', 'er', 'ah', 'eh', 'mm', 'hmm', 'hm',
            'like', 'you know', 'i mean', 'actually', 'basically',
            'literally', 'sort of', 'kind of', 'well', 'so',
            'right', 'okay', 'ok', 'yeah', 'yes', 'yep', 'mhm'
        }
    
    def load_text_file(self, filepath: str) -> str:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {filepath}")
        except Exception as e:
            raise Exception(f"Error reading {filepath}: {e}")
    
    def normalize_text(self, text: str) -> List[str]:
        if self.ignore_case:
            text = text.lower()
        
        if self.ignore_punctuation:
            text = re.sub(r"[^\w\s']", ' ', text)
            text = re.sub(r"'ll", " will", text)
            text = re.sub(r"'re", " are", text)
            text = re.sub(r"'ve", " have", text)
            text = re.sub(r"'d", " would", text)
            text = re.sub(r"n't", " not", text)
            text = re.sub(r"'m", " am", text)
            text = re.sub(r"'s", " is", text)
        
        if self.ignore_extra_spaces:
            text = ' '.join(text.split())
        
        words = text.split()
        
        if self.remove_filler_words:
            words = [word for word in words if word.lower() not in self.filler_words]
        
        words = [word for word in words if word.strip()]
        
        return words
    
    def compute_word_alignment(self, reference: List[str], hypothesis: List[str]) -> Tuple[int, int, int]:
        len_ref = len(reference)
        len_hyp = len(hypothesis)
        
        dp = [[0] * (len_hyp + 1) for _ in range(len_ref + 1)]
        
        for i in range(len_ref + 1):
            dp[i][0] = i
        for j in range(len_hyp + 1):
            dp[0][j] = j
        
        for i in range(1, len_ref + 1):
            for j in range(1, len_hyp + 1):
                if reference[i-1] == hypothesis[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    substitution = dp[i-1][j-1] + 1
                    deletion = dp[i-1][j] + 1
                    insertion = dp[i][j-1] + 1
                    dp[i][j] = min(substitution, deletion, insertion)
        
        i, j = len_ref, len_hyp
        substitutions = deletions = insertions = 0
        
        while i > 0 or j > 0:
            if i > 0 and j > 0 and reference[i-1] == hypothesis[j-1]:
                i -= 1
                j -= 1
            elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
                substitutions += 1
                i -= 1
                j -= 1
            elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
                deletions += 1
                i -= 1
            elif j > 0 and dp[i][j] == dp[i][j-1] + 1:
                insertions += 1
                j -= 1
        
        return substitutions, deletions, insertions
    
    def calculate_wer(self, reference_file: str, hypothesis_file: str) -> dict:
        ref_text = self.load_text_file(reference_file)
        hyp_text = self.load_text_file(hypothesis_file)
        
        print(f"Loaded reference from: {reference_file}")
        print(f"Loaded hypothesis from: {hypothesis_file}")
        print(f"Reference length: {len(ref_text)} chars")
        print(f"Hypothesis length: {len(hyp_text)} chars")
        
        ref_words = self.normalize_text(ref_text)
        hyp_words = self.normalize_text(hyp_text)
        
        print(f"Normalized reference words: {len(ref_words)}")
        print(f"Normalized hypothesis words: {len(hyp_words)}")
        
        substitutions, deletions, insertions = self.compute_word_alignment(ref_words, hyp_words)
        
        total_words = len(ref_words)
        total_errors = substitutions + deletions + insertions
        correct_words = total_words - deletions - substitutions
        
        wer = (total_errors / total_words * 100) if total_words > 0 else 0
        accuracy = (correct_words / total_words * 100) if total_words > 0 else 0
        
        precision = (correct_words / len(hyp_words) * 100) if len(hyp_words) > 0 else 0
        recall = (correct_words / total_words * 100) if total_words > 0 else 0
        f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
        
        return {
            'wer': wer,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'total_words': total_words,
            'correct_words': correct_words,
            'substitutions': substitutions,
            'deletions': deletions,
            'insertions': insertions,
            'total_errors': total_errors,
            'reference_words': ref_words,
            'hypothesis_words': hyp_words
        }
    
    def show_alignment_sample(self, reference_words: List[str], hypothesis_words: List[str], 
                             num_samples: int = 10) -> None:
        print(f"\nSample word alignment (first {num_samples} words):")
        print("REF:", ' '.join(reference_words[:num_samples]))
        print("HYP:", ' '.join(hypothesis_words[:num_samples]))
        
        diff = list(difflib.unified_diff(
            reference_words[:50], hypothesis_words[:50], 
            fromfile='Reference', tofile='Hypothesis', lineterm=''
        ))
        
        if len(diff) > 2:
            print("\nDetailed differences (first 50 words):")
            for line in diff[2:12]:
                print(line)
    
    def analyze_errors(self, reference_words: List[str], hypothesis_words: List[str]) -> dict:
        ref_counter = Counter(reference_words)
        hyp_counter = Counter(hypothesis_words)
        
        only_in_ref = set(reference_words) - set(hypothesis_words)
        only_in_hyp = set(hypothesis_words) - set(reference_words)
        
        return {
            'most_common_ref_words': ref_counter.most_common(10),
            'most_common_hyp_words': hyp_counter.most_common(10),
            'likely_deleted_words': list(only_in_ref)[:10],
            'likely_inserted_words': list(only_in_hyp)[:10],
            'vocab_overlap': len(set(reference_words) & set(hypothesis_words)),
            'total_ref_vocab': len(set(reference_words)),
            'total_hyp_vocab': len(set(hypothesis_words))
        }
    
    def generate_report(self, results: dict, output_file: str = None) -> str:
        report = []
        report.append("=" * 60)
        report.append("WORD ERROR RATE (WER) EVALUATION REPORT")
        report.append("=" * 60)
        
        report.append(f"\nMETRICS:")
        report.append(f"Word Error Rate (WER):     {results['wer']:.2f}%")
        report.append(f"Word Accuracy:             {results['accuracy']:.2f}%")
        report.append(f"Precision:                 {results['precision']:.2f}%")
        report.append(f"Recall:                    {results['recall']:.2f}%")
        report.append(f"F1-Score:                  {results['f1_score']:.2f}%")
        
        report.append(f"\nERROR BREAKDOWN:")
        report.append(f"Total words (reference):   {results['total_words']}")
        report.append(f"Correct words:             {results['correct_words']}")
        report.append(f"Substitutions:             {results['substitutions']}")
        report.append(f"Deletions:                 {results['deletions']}")
        report.append(f"Insertions:                {results['insertions']}")
        report.append(f"Total errors:              {results['total_errors']}")
        
        total_words = results['total_words']
        if total_words > 0:
            report.append(f"\nERROR RATES:")
            report.append(f"Substitution rate:         {results['substitutions']/total_words*100:.2f}%")
            report.append(f"Deletion rate:             {results['deletions']/total_words*100:.2f}%")
            report.append(f"Insertion rate:            {results['insertions']/total_words*100:.2f}%")
        
        if 'error_analysis' in results:
            analysis = results['error_analysis']
            report.append(f"\nVOCABULARY ANALYSIS:")
            report.append(f"Reference vocabulary size: {analysis['total_ref_vocab']}")
            report.append(f"Hypothesis vocabulary size: {analysis['total_hyp_vocab']}")
            report.append(f"Vocabulary overlap:        {analysis['vocab_overlap']}")
            
            if analysis['likely_deleted_words']:
                report.append(f"\nMost likely deleted words: {', '.join(analysis['likely_deleted_words'])}")
            if analysis['likely_inserted_words']:
                report.append(f"Most likely inserted words: {', '.join(analysis['likely_inserted_words'])}")
        
        report.append("\n" + "=" * 60)
        
        report_text = '\n'.join(report)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"Report saved to: {output_file}")
        
        return report_text

def main():
    parser = argparse.ArgumentParser(description="Calculate Word Error Rate (WER) between ground truth and ASR output")
    parser.add_argument("reference", help="Ground truth text file")
    parser.add_argument("hypothesis", help="ASR output text file")
    parser.add_argument("-o", "--output", help="Output report file")
    parser.add_argument("--case-sensitive", action="store_true", 
                       help="Perform case-sensitive comparison")
    parser.add_argument("--keep-punctuation", action="store_true",
                       help="Keep punctuation in comparison")
    parser.add_argument("--keep-fillers", action="store_true",
                       help="Keep filler words (um, uh, etc.)")
    parser.add_argument("--show-alignment", action="store_true",
                       help="Show sample word alignments")
    parser.add_argument("--detailed-analysis", action="store_true",
                       help="Perform detailed error analysis")
    
    args = parser.parse_args()
    
    try:
        calculator = WERCalculator(
            ignore_case=not args.case_sensitive,
            ignore_punctuation=not args.keep_punctuation,
            ignore_extra_spaces=True,
            remove_filler_words=not args.keep_fillers
        )
        
        print("WER Calculator Configuration:")
        print(f"Case sensitive: {args.case_sensitive}")
        print(f"Keep punctuation: {args.keep_punctuation}")
        print(f"Keep filler words: {args.keep_fillers}")
        print("-" * 40)
        
        results = calculator.calculate_wer(args.reference, args.hypothesis)
        
        if args.show_alignment:
            calculator.show_alignment_sample(
                results['reference_words'], 
                results['hypothesis_words']
            )
        
        if args.detailed_analysis:
            error_analysis = calculator.analyze_errors(
                results['reference_words'],
                results['hypothesis_words']
            )
            results['error_analysis'] = error_analysis
        
        report = calculator.generate_report(results, args.output)
        print("\n" + report)
        
        exit_code = min(int(results['wer']), 100)
        sys.exit(exit_code)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

# Simple function for direct use
def calculate_wer(reference_file: str, hypothesis_file: str, 
                 ignore_case: bool = True, ignore_punctuation: bool = True,
                 remove_fillers: bool = True) -> float:
    calculator = WERCalculator(
        ignore_case=ignore_case,
        ignore_punctuation=ignore_punctuation,
        remove_filler_words=remove_fillers
    )
    
    results = calculator.calculate_wer(reference_file, hypothesis_file)
    return results['wer']

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("WER Calculator for ASR Evaluation")
        print("=" * 40)
        print("Usage examples:")
        print(f"python {sys.argv[0]} ground_truth.txt asr_output.txt")
        print(f"python {sys.argv[0]} ground_truth.txt asr_output.txt -o report.txt")
        print(f"python {sys.argv[0]} ground_truth.txt asr_output.txt --show-alignment")
        print(f"python {sys.argv[0]} ground_truth.txt asr_output.txt --detailed-analysis")
        print(f"python {sys.argv[0]} --help")
        
        common_files = [
            ("ground_truth.txt", "audioFile_transcript.txt"),
            ("reference.txt", "transcript.txt"),
            ("truth.txt", "output.txt")
        ]
        
        import os
        for ref_file, hyp_file in common_files:
            if os.path.exists(ref_file) and os.path.exists(hyp_file):
                print(f"\nFound {ref_file} and {hyp_file} - calculate WER? (y/n)")
                response = input().strip().lower()
                if response in ['y', 'yes']:
                    try:
                        wer = calculate_wer(ref_file, hyp_file)
                        print(f"WER: {wer:.2f}%")
                    except Exception as e:
                        print(f"Error: {e}")
                break
        
        sys.exit(0)
    
    main()