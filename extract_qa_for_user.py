#!/usr/bin/env python3
"""
Extract QA data for a specific user from HaluMem-Medium.jsonl
and convert to format expected by retrieval system.
"""
import json
import sys

def extract_qa_for_user(input_file, output_file, target_uuid):
    """Extract QA data for target_uuid and save in retrieval format."""

    # Read the JSONL file and find the target user
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)

            if data.get('uuid') == target_uuid:
                # Found the target user
                print(f"✓ Found user: {target_uuid}")
                print(f"  Total sessions: {len(data.get('sessions', []))}")

                # Extract all questions from all sessions
                all_questions = []
                for session_idx, session in enumerate(data.get('sessions', [])):
                    questions = session.get('questions', [])
                    print(f"  Session {session_idx}: {len(questions)} questions")

                    for q in questions:
                        all_questions.append({
                            'question': q.get('question', ''),
                            'answer': q.get('answer', ''),
                            'evidence': q.get('evidence', []),
                            'category': q.get('question_type', ''),
                            'difficulty': q.get('difficulty', '')
                        })

                # Create output in format expected by retrieval system
                # The system expects a list with one entry per conversation
                output_data = [{
                    'user_id': target_uuid,
                    'conversation': data.get('sessions', []),
                    'qa': all_questions,
                    'persona_info': data.get('persona_info', '')
                }]

                # Save to output file
                with open(output_file, 'w', encoding='utf-8') as out_f:
                    json.dump(output_data, out_f, ensure_ascii=False, indent=2)

                print(f"\n✓ Extracted {len(all_questions)} questions")
                print(f"✓ Saved to: {output_file}")
                return True

    print(f"✗ User {target_uuid} not found in {input_file}")
    return False

if __name__ == '__main__':
    input_file = '/data/HaluMem/yjm/data/HaluMem-Medium.jsonl'
    output_file = '/data/wjl/SA-Mem/data/qa_2e36c193.json'
    target_uuid = '2e36c193-a605-8cc7-70ff-80b2336d4980'

    extract_qa_for_user(input_file, output_file, target_uuid)
