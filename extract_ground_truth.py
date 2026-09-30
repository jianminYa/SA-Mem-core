#!/usr/bin/env python3
"""
Extract ground truth block IDs from HaluMem evidence annotations.

This script maps evidence entries in QA pairs to memory block IDs by:
1. Content similarity matching
2. Keyword overlap
3. Memory type alignment

Usage:
    python extract_ground_truth.py --qa-file data/qa_1b846c59.json --blocks-file out/halupass2-1b846c59/final_boxes_content.jsonl --output ground_truth.json
"""

import argparse
import json
import re
from typing import List, Dict, Any, Optional, Set
from collections import defaultdict


def load_qa_data(qa_file: str) -> List[Dict[str, Any]]:
    """Load QA pairs from HaluMem format."""
    with open(qa_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # HaluMem format: list with one user's data
    if isinstance(data, list) and len(data) > 0:
        user_data = data[0]
        qa_pairs = user_data.get('qa', [])
        print(f"✅ Loaded {len(qa_pairs)} QA pairs")
        return qa_pairs

    return []


def load_blocks(blocks_file: str) -> List[Dict[str, Any]]:
    """Load memory blocks from JSONL file."""
    blocks = []
    with open(blocks_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                blocks.append(json.loads(line))

    print(f"✅ Loaded {len(blocks)} memory blocks")
    return blocks


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    # Lowercase
    text = text.lower()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()


def extract_keywords(text: str) -> Set[str]:
    """Extract meaningful keywords from text."""
    normalized = normalize_text(text)
    # Remove common stop words
    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                  'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                  'would', 'should', 'could', 'may', 'might', 'must', 'can',
                  'of', 'at', 'by', 'for', 'with', 'about', 'as', 'to', 'from',
                  'in', 'on', 'that', 'this', 'it', 'and', 'or', 'but'}

    words = normalized.split()
    keywords = {w for w in words if len(w) > 2 and w not in stop_words}
    return keywords


def calculate_keyword_overlap(text1: str, text2: str) -> float:
    """Calculate keyword overlap ratio between two texts."""
    keywords1 = extract_keywords(text1)
    keywords2 = extract_keywords(text2)

    if not keywords1 or not keywords2:
        return 0.0

    intersection = keywords1 & keywords2
    union = keywords1 | keywords2

    return len(intersection) / len(union) if union else 0.0


def match_evidence_to_block(
    evidence: Dict[str, Any],
    blocks: List[Dict[str, Any]],
    threshold: float = 0.3
) -> Optional[int]:
    """
    Match an evidence entry to a memory block.

    Args:
        evidence: Evidence dict with 'memory_content' and 'memory_type'
        blocks: List of memory blocks
        threshold: Minimum keyword overlap ratio to consider a match

    Returns:
        Block ID if match found, None otherwise
    """
    evidence_content = evidence.get('memory_content', '')
    evidence_type = evidence.get('memory_type', '')

    if not evidence_content:
        return None

    best_match = None
    best_score = 0.0

    for block in blocks:
        block_id = block.get('block_id')

        # Check events in the block
        events = block.get('events', [])
        for event in events:
            event_desc = event.get('description', '')

            # Calculate keyword overlap
            score = calculate_keyword_overlap(evidence_content, event_desc)

            # Bonus for memory type match
            if evidence_type == 'Persona Memory':
                # Persona memories are usually ATTRIBUTE type
                if event.get('event_temporal_type') == 'ATTRIBUTE':
                    score += 0.1

            if score > best_score and score >= threshold:
                best_score = score
                best_match = block_id

        # Also check topic keywords
        features = block.get('features', {})
        topic_kw = features.get('topic_kw_text', '')
        if topic_kw:
            score = calculate_keyword_overlap(evidence_content, topic_kw)
            if score > best_score and score >= threshold:
                best_score = score
                best_match = block_id

    return best_match


def extract_ground_truth(
    qa_pairs: List[Dict[str, Any]],
    blocks: List[Dict[str, Any]],
    threshold: float = 0.3
) -> Dict[str, Any]:
    """
    Extract ground truth block IDs for all QA pairs.

    Returns:
        Dict mapping QA index to ground truth info
    """
    ground_truth = {}
    stats = {
        'total_qa': len(qa_pairs),
        'qa_with_evidence': 0,
        'total_evidence': 0,
        'matched_evidence': 0,
        'unmatched_evidence': 0
    }

    for idx, qa in enumerate(qa_pairs):
        question = qa.get('question', '')
        evidence_list = qa.get('evidence', [])

        if not evidence_list:
            # No evidence for this QA
            ground_truth[str(idx)] = {
                'question': question,
                'has_evidence': False,
                'evidence_count': 0,
                'block_ids': [],
                'matched_count': 0
            }
            continue

        stats['qa_with_evidence'] += 1
        stats['total_evidence'] += len(evidence_list)

        # Match each evidence to blocks
        matched_blocks = []
        unmatched_evidence = []

        for evidence in evidence_list:
            block_id = match_evidence_to_block(evidence, blocks, threshold)
            if block_id is not None:
                matched_blocks.append(block_id)
                stats['matched_evidence'] += 1
            else:
                unmatched_evidence.append(evidence.get('memory_content', ''))
                stats['unmatched_evidence'] += 1

        # Remove duplicates while preserving order
        unique_blocks = []
        seen = set()
        for bid in matched_blocks:
            if bid not in seen:
                unique_blocks.append(bid)
                seen.add(bid)

        ground_truth[str(idx)] = {
            'question': question,
            'has_evidence': True,
            'evidence_count': len(evidence_list),
            'block_ids': unique_blocks,
            'matched_count': len(unique_blocks),
            'unmatched_evidence': unmatched_evidence
        }

    return ground_truth, stats


def main():
    parser = argparse.ArgumentParser(description='Extract ground truth from HaluMem evidence')
    parser.add_argument('--qa-file', required=True, help='QA pairs JSON file')
    parser.add_argument('--blocks-file', required=True, help='Memory blocks JSONL file')
    parser.add_argument('--output', default='ground_truth.json', help='Output JSON file')
    parser.add_argument('--threshold', type=float, default=0.3, help='Keyword overlap threshold')
    parser.add_argument('--verbose', action='store_true', help='Show detailed matching info')

    args = parser.parse_args()

    print("=" * 60)
    print("Ground Truth Extraction")
    print("=" * 60)

    # Load data
    qa_pairs = load_qa_data(args.qa_file)
    blocks = load_blocks(args.blocks_file)

    # Extract ground truth
    print(f"\n🔍 Matching evidence to blocks (threshold={args.threshold})...")
    ground_truth, stats = extract_ground_truth(qa_pairs, blocks, args.threshold)

    # Save results
    output_data = {
        'metadata': {
            'qa_file': args.qa_file,
            'blocks_file': args.blocks_file,
            'threshold': args.threshold,
            'stats': stats
        },
        'ground_truth': ground_truth
    }

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # Print statistics
    print("\n" + "=" * 60)
    print("Statistics")
    print("=" * 60)
    print(f"Total QA pairs: {stats['total_qa']}")
    print(f"QA with evidence: {stats['qa_with_evidence']} ({stats['qa_with_evidence']/stats['total_qa']*100:.1f}%)")
    print(f"Total evidence entries: {stats['total_evidence']}")
    print(f"Matched evidence: {stats['matched_evidence']} ({stats['matched_evidence']/stats['total_evidence']*100:.1f}%)")
    print(f"Unmatched evidence: {stats['unmatched_evidence']} ({stats['unmatched_evidence']/stats['total_evidence']*100:.1f}%)")

    # Show sample matches
    if args.verbose:
        print("\n" + "=" * 60)
        print("Sample Matches (first 5 QA with evidence)")
        print("=" * 60)
        count = 0
        for idx, gt in ground_truth.items():
            if gt['has_evidence'] and count < 5:
                print(f"\nQ{idx}: {gt['question'][:60]}...")
                print(f"  Evidence count: {gt['evidence_count']}")
                print(f"  Matched blocks: {gt['block_ids']}")
                if gt['unmatched_evidence']:
                    print(f"  Unmatched: {len(gt['unmatched_evidence'])} evidence entries")
                count += 1

    print(f"\n✅ Ground truth saved to: {args.output}")


if __name__ == '__main__':
    main()
