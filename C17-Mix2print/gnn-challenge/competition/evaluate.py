"""
Main evaluation script for scoring submissions.
"""

import sys
import os
import argparse

# Add competition directory to path
sys.path.insert(0, os.path.dirname(__file__))

from metrics import compute_scores


def format_score_result(scores: dict) -> str:
    """Format scores as a readable message."""
    msg = "## [Evaluation Results]\n\n"
    
    msg += f"\n### [Combined Score]\n\n"
    msg += f"**NMAE: {scores['combined_nmae']:.6f}** ({scores['combined_pct']:.2f}%)\n\n"
    msg += f"*Lower is better. This is the official ranking metric.*\n"
    
    return msg


def main():
    parser = argparse.ArgumentParser(description='Evaluate bioink predictions')
    parser.add_argument('predictions', help='Path to predictions.csv')
    parser.add_argument('ground_truth', help='Path to ground truth labels')
    parser.add_argument('--format', choices=['json', 'markdown', 'simple'], 
                       default='simple', help='Output format')
    
    args = parser.parse_args()
    
    try:
        scores = compute_scores(args.predictions, args.ground_truth)
        
        if args.format == 'json':
            import json
            print(json.dumps(scores, indent=2))
        elif args.format == 'markdown':
            print(format_score_result(scores))
        else:  # simple
            print(f"SCORE={scores['combined_nmae']:.8f}")
    
    except Exception as e:
        if args.format == 'markdown':
            print(f"## ❌ Evaluation Failed\n\n**Error:** `{e}`\n\nPlease check that your submission matches `sample_submission.csv` (correct IDs and columns).")
        elif args.format == 'json':
            import json
            print(json.dumps({"error": str(e)}))
        else:
            print(f"ERROR: {e}", file=sys.stderr)
        
        sys.exit(1)


if __name__ == '__main__':
    main()
