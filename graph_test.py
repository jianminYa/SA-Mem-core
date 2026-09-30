#!/usr/bin/env python3
"""Quick Neo4j verification for one block.

Finds one block for a user and prints:
- events under the block
- 1-hop similar events (SIMILAR edges)
- entity relation triples linked to the block's events
"""

import argparse
from typing import Any, Dict, List

from graph_storage import GraphConfig, MemgraphEventGraph


def _pick_block(graph: MemgraphEventGraph, user_id: str, block_id: str | None) -> str | None:
	if block_id:
		return block_id
	rows = graph._run(
		"""
		MATCH (b:Block {user_id: $user_id})
		RETURN b.key AS block_key
		LIMIT 1
		""",
		{"user_id": user_id},
	)
	if not rows:
		return None
	block_key = str(rows[0].get("block_key") or "")
	if "|" in block_key:
		return block_key.split("|", 1)[1]
	return block_key or None


def _print_section(title: str, items: List[Dict[str, Any]]) -> None:
	print(f"\n== {title} ==")
	if not items:
		print("(empty)")
		return
	for item in items:
		print(item)


def main() -> None:
	parser = argparse.ArgumentParser(description="Verify Neo4j contents for a block")
	parser.add_argument("--user-id", required=True, help="User id to query")
	parser.add_argument("--block-id", default=None, help="Block id (optional)")
	parser.add_argument("--min-score", type=float, default=0.8, help="Min similarity score for SIMILAR edges")
	parser.add_argument("--limit", type=int, default=50, help="Max rows to return per query")
	args = parser.parse_args()

	cfg = GraphConfig()
	graph = MemgraphEventGraph(
		url=cfg.memgraph_url,
		username=cfg.memgraph_username,
		password=cfg.memgraph_password,
	)

	block_id = _pick_block(graph, str(args.user_id), args.block_id)
	if not block_id:
		print("No block found for user", args.user_id)
		return

	events_by_block = graph.get_events_by_block_ids(
		user_id=str(args.user_id),
		block_ids=[block_id],
	)
	events = events_by_block.get(str(block_id), []) if events_by_block else []
	_print_section(f"Block {block_id} events", events)

	event_ids = [str(e.get("event_id")) for e in events if e.get("event_id")]
	similar_edges = graph.get_similar_events_1hop(
		user_id=str(args.user_id),
		event_ids=event_ids,
		min_score=float(args.min_score),
		limit=int(args.limit),
	)
	_print_section("1-hop similar events", similar_edges)

	relations_by_block = graph.get_relations_by_block_ids(
		user_id=str(args.user_id),
		block_ids=[block_id],
		limit=int(args.limit),
	)
	relations = relations_by_block.get(str(block_id), []) if relations_by_block else []
	_print_section("Entity relation triples", relations)


if __name__ == "__main__":
	'''
	python graph_test.py --user-id <YOUR_USER_ID> --block-id <BLOCK_ID>
	'''
	main()
