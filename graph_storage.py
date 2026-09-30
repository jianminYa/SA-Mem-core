"""Graph storage support extracted from prototype memory_main.

This module provides:
- Graph-related config fields (Neo4j over Bolt).
- MemgraphEventGraph for event nodes, block nodes, and similarity edges.

It is intentionally minimal and safe to import even if graph is disabled.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class GraphConfig:
	"""Graph storage config used by the memory system."""

	# Feature toggle
	enable_graph: bool = False   # 临时关闭，build完成后改回True

	# Bolt connection settings (Neo4j)
	memgraph_url: str = os.getenv("NEO4J_URI") or "bolt://localhost:7687"
	memgraph_username: str = os.getenv("NEO4J_USERNAME") or "neo4j"
	memgraph_password: str = os.getenv("NEO4J_PASSWORD") or ""

	# Expansion parameters
	graph_expand_limit: int = 50

	# Similarity-edge building when inserting events
	graph_similarity_threshold: float = 0.8
	graph_similar_limit: int = 20


class MemgraphEventGraph:
	"""Event-level graph storage backed by Neo4j (Bolt).

	This graph is intentionally minimal:
	- Vector store remains the source of truth for embeddings and payload.
	- Neo4j stores references to events by vector id (memory_id).

	Stored graph entities:
	- (:Event) node per event
		- id / vector_id: event vector-store id (memory_id)
		- user_id: used for isolation
		- block_key: stable grouping key for the block
		- event_start_time, event_end_time: event time window (string)
		- created_at: event ingestion time
	- (:Block) node per block
		- key: block_key
		- user_id
	- Relationship:
		(:Block)-[:CONTAINS]->(:Event)

	Graph-assisted retrieval:
	- Expand from seed events to other events in the same block_key.
	"""

	def __init__(self, *, url: str, username: str, password: str) -> None:
		try:
			from neo4j import GraphDatabase  # type: ignore
		except Exception as e:
			raise ImportError(
				"Neo4j uses the Neo4j Python driver for Bolt. Install with: pip install neo4j"
			) from e

		# Neo4j uses the Neo4j driver for Bolt.
		self._driver = GraphDatabase.driver(url, auth=(username, password))
		self._ensure_schema_best_effort()

	def close(self) -> None:
		"""Close the underlying Bolt driver."""
		try:
			self._driver.close()
		except Exception:
			pass

	def reset(self) -> None:
		"""Delete all nodes and relationships in the graph (best-effort)."""
		self._run("MATCH (n) DETACH DELETE n")

	def _run(self, cypher: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
		"""Run a cypher query and return rows as a list of dict."""
		with self._driver.session() as session:
			result = session.run(cypher, params or {})
			return [dict(r) for r in result]

	def _ensure_schema_best_effort(self) -> None:
		"""Try to create helpful indexes/constraints without failing hard."""
		candidates = [
			"CREATE CONSTRAINT ON (e:Event) ASSERT e.id IS UNIQUE",
			"CREATE CONSTRAINT IF NOT EXISTS FOR (e:Event) REQUIRE e.id IS UNIQUE",
		]
		for q in candidates:
			try:
				self._run(q)
				break
			except Exception:
				continue
		for q in [
			"CREATE INDEX ON :Event(user_id)",
			"CREATE INDEX ON :Event(block_key)",
			"CREATE INDEX ON :Entity(user_id)",
			"CREATE INDEX ON :Entity(name)",
			"CREATE INDEX ON :Relation(user_id)",
			"CREATE INDEX ON :Block(key)",
		]:
			try:
				self._run(q)
			except Exception:
				pass

	def upsert_event_node(self, *, memory_id: str, payload: Dict[str, Any]) -> None:
		"""Upsert an (:Event) node for a vector record."""
		user_id = str(payload.get("user_id") or "")
		if not user_id:
			raise ValueError("Graph upsert requires payload.user_id")

		block_key = payload.get("block_key")
		created_at = payload.get("created_at")
		event_start_time = payload.get("event_start_time")
		event_end_time = payload.get("event_end_time")
		block_event_start_time = payload.get("block_event_start_time")
		block_event_end_time = payload.get("block_event_end_time")
		block_topic_category = payload.get("block_topic_category")
		block_topic_categories = payload.get("block_topic_categories")
		block_topic_kw_text = payload.get("block_topic_kw_text")
		block_features_categories = payload.get("block_features_categories")
		block_features_json = payload.get("block_features_json")

		event_description = payload.get("event_description")
		event_temporal_type = payload.get("event_temporal_type")

		self._run(
			"""
			MERGE (e:Event {id: $id})
			SET e.vector_id = $id,
				e.user_id = $user_id,
				e.block_key = $block_key,
				e.event_start_time = $event_start_time,
				e.event_end_time = $event_end_time,
				e.event_description = COALESCE(e.event_description, $event_description),
				e.event_temporal_type = COALESCE(e.event_temporal_type, $event_temporal_type),
				e.created_at = COALESCE(e.created_at, $created_at)
			""",
			{
				"id": str(memory_id),
				"user_id": user_id,
				"block_key": str(block_key) if block_key is not None else None,
				"event_start_time": str(event_start_time) if event_start_time is not None else None,
				"event_end_time": str(event_end_time) if event_end_time is not None else None,
				"event_description": str(event_description) if event_description is not None else None,
				"event_temporal_type": str(event_temporal_type) if event_temporal_type is not None else None,
				"created_at": str(created_at) if created_at is not None else _now_iso(),
			},
		)

		if block_key:
			self._run(
				"""
				MERGE (b:Block {key: $block_key, user_id: $user_id})
				SET b.block_event_start_time = $block_event_start_time,
					b.block_event_end_time = $block_event_end_time,
					b.topic_category = $block_topic_category,
					b.topic_categories = $block_topic_categories,
					b.topic_kw_text = $block_topic_kw_text,
					b.features_categories = $block_features_categories,
					b.features_json = $block_features_json
				WITH b
				MATCH (e:Event {id: $id})
				MERGE (b)-[:CONTAINS]->(e)
				""",
				{
					"block_key": str(block_key),
					"user_id": user_id,
					"id": str(memory_id),
					"block_event_start_time": str(block_event_start_time)
					if block_event_start_time is not None
					else None,
					"block_event_end_time": str(block_event_end_time)
					if block_event_end_time is not None
					else None,
					"block_topic_category": str(block_topic_category)
					if block_topic_category is not None
					else None,
					"block_topic_categories": list(block_topic_categories)
					if isinstance(block_topic_categories, list)
					else None,
					"block_topic_kw_text": str(block_topic_kw_text)
					if block_topic_kw_text is not None
					else None,
					"block_features_categories": list(block_features_categories)
					if isinstance(block_features_categories, list)
					else None,
					"block_features_json": str(block_features_json)
					if block_features_json is not None
					else None,
				},
			)

	def expand_event_ids_by_box(self, *, seed_ids: List[str], user_id: str, limit: int) -> List[str]:
		"""Expand seed event ids to other events in the same block(s)."""
		if not seed_ids:
			return []

		rows = self._run(
			"""
			MATCH (s:Event {user_id: $user_id})
			WHERE s.id IN $seed_ids AND s.block_key IS NOT NULL
			WITH COLLECT(DISTINCT s.block_key) AS block_keys
			MATCH (e:Event {user_id: $user_id})
			WHERE e.block_key IN block_keys
			RETURN DISTINCT e.id AS id
			LIMIT $limit
			""",
			{
				"seed_ids": [str(x) for x in seed_ids],
				"user_id": str(user_id),
				"limit": int(limit),
			},
		)

		expanded = [str(r.get("id")) for r in rows if r.get("id")]
		seed_set = {str(x) for x in seed_ids}
		return [x for x in expanded if x not in seed_set]

	def get_events_by_block_ids(self, *, user_id: str, block_ids: List[Any]) -> Dict[str, List[Dict[str, Any]]]:
		"""Fetch events grouped by block_id for a user.

		Returns: {"<block_id>": [{"event_id":..., "event_start_time":..., "event_end_time":..., "event_description":...}], ...}
		"""
		if not block_ids:
			return {}
		block_keys = [_make_block_key(user_id=str(user_id), block_id=bid) for bid in block_ids]
		rows = self._run(
			"""
			MATCH (b:Block {user_id: $user_id})-[:CONTAINS]->(e:Event)
			WHERE b.key IN $block_keys
			RETURN b.key AS block_key,
				e.id AS event_id,
				e.event_start_time AS event_start_time,
				e.event_end_time AS event_end_time,
				e.event_description AS event_description,
				e.event_temporal_type AS event_temporal_type
			""",
			{
				"user_id": str(user_id),
				"block_keys": [str(x) for x in block_keys],
			},
		)
		out: Dict[str, List[Dict[str, Any]]] = {}
		for r in rows:
			bk = str(r.get("block_key") or "")
			bid = ""
			if "|" in bk:
				bid = bk.split("|", 1)[1]
			else:
				bid = bk
			out.setdefault(bid, [])
			out[bid].append(
				{
					"event_id": r.get("event_id"),
					"event_start_time": r.get("event_start_time"),
					"event_end_time": r.get("event_end_time"),
					"event_description": r.get("event_description"),
					"event_temporal_type": r.get("event_temporal_type"),
				}
			)
		return out

	def get_similar_events_1hop(
		self,
		*,
		user_id: str,
		event_ids: List[str],
		min_score: float = 0.7,
		limit: int = 200,
	) -> List[Dict[str, Any]]:
		"""Fetch 1-hop similar events for a list of event ids."""
		if not event_ids:
			return []
		rows = self._run(
			"""
			MATCH (e:Event {user_id: $user_id})-[r:SIMILAR]-(s:Event {user_id: $user_id})
			WHERE e.id IN $event_ids AND r.score >= $min_score
			RETURN e.id AS from_event_id, s.id AS to_event_id, r.score AS score
			LIMIT $limit
			""",
			{
				"user_id": str(user_id),
				"event_ids": [str(x) for x in event_ids],
				"min_score": float(min_score),
				"limit": int(limit),
			},
		)
		out: List[Dict[str, Any]] = []
		for r in rows:
			out.append(
				{
					"from_event_id": r.get("from_event_id"),
					"to_event_id": r.get("to_event_id"),
					"score": r.get("score"),
				}
			)
		return out

	def get_events_by_ids(self, *, user_id: str, event_ids: List[str]) -> List[Dict[str, Any]]:
		"""Fetch event details by event ids for a user."""
		if not event_ids:
			return []
		rows = self._run(
			"""
			MATCH (e:Event {user_id: $user_id})
			WHERE e.id IN $event_ids
			RETURN e.id AS event_id,
				e.event_start_time AS event_start_time,
				e.event_end_time AS event_end_time,
				e.event_description AS event_description,
				e.event_temporal_type AS event_temporal_type
			""",
			{
				"user_id": str(user_id),
				"event_ids": [str(x) for x in event_ids],
			},
		)
		return [
			{
				"event_id": r.get("event_id"),
				"event_start_time": r.get("event_start_time"),
				"event_end_time": r.get("event_end_time"),
				"event_description": r.get("event_description"),
				"event_temporal_type": r.get("event_temporal_type"),
			}
			for r in rows
		]

	def get_relations_by_event_ids(
		self,
		*,
		user_id: str,
		event_ids: List[str],
		limit: int = 200,
	) -> List[Dict[str, Any]]:
		"""Fetch entity relation triples for a list of event ids."""
		if not event_ids:
			return []
		rows = self._run(
			"""
			MATCH (e:Event {user_id: $user_id})-[:MENTIONS_RELATION]->(rel:Relation)
			WHERE e.id IN $event_ids
			RETURN rel.source AS source,
				rel.type AS relationship,
				rel.destination AS destination,
				rel.event_id AS event_id
			LIMIT $limit
			""",
			{
				"user_id": str(user_id),
				"event_ids": [str(x) for x in event_ids],
				"limit": int(limit),
			},
		)
		out: List[Dict[str, Any]] = []
		for r in rows:
			out.append(
				{
					"source": r.get("source"),
					"relationship": r.get("relationship"),
					"destination": r.get("destination"),
					"event_id": r.get("event_id"),
				}
			)
		return out

	def get_relations_by_block_ids(self, *, user_id: str, block_ids: List[Any], limit: int = 200) -> Dict[str, List[Dict[str, Any]]]:
		"""Fetch entity relation triples grouped by block_id."""
		if not block_ids:
			return {}
		block_keys = [_make_block_key(user_id=str(user_id), block_id=bid) for bid in block_ids]
		rows = self._run(
			"""
			MATCH (b:Block {user_id: $user_id})-[:CONTAINS]->(e:Event)-[:MENTIONS_RELATION]->(rel:Relation)
			WHERE b.key IN $block_keys
			RETURN b.key AS block_key,
				rel.source AS source,
				rel.type AS relationship,
				rel.destination AS destination,
				rel.event_id AS event_id
			LIMIT $limit
			""",
			{
				"user_id": str(user_id),
				"block_keys": [str(x) for x in block_keys],
				"limit": int(limit),
			},
		)
		out: Dict[str, List[Dict[str, Any]]] = {}
		for r in rows:
			bk = str(r.get("block_key") or "")
			bid = ""
			if "|" in bk:
				bid = bk.split("|", 1)[1]
			else:
				bid = bk
			out.setdefault(bid, [])
			out[bid].append(
				{
					"source": r.get("source"),
					"relationship": r.get("relationship"),
					"destination": r.get("destination"),
					"event_id": r.get("event_id"),
				}
			)
		return out

	def link_similar_events(self, *, user_id: str, a_id: str, b_id: str, score: float) -> None:
		"""Create/update a similarity edge between two events."""
		left_id, right_id = (str(a_id), str(b_id))
		if left_id == right_id:
			return
		if left_id > right_id:
			left_id, right_id = right_id, left_id

		now = _now_iso()
		self._run(
			"""
			MERGE (a:Event {id: $a_id})
			MERGE (b:Event {id: $b_id})
			SET a.user_id = COALESCE(a.user_id, $user_id),
				b.user_id = COALESCE(b.user_id, $user_id)
			MERGE (a)-[r:SIMILAR]->(b)
			SET r.score = CASE
				WHEN r.score IS NULL THEN $score
				WHEN $score > r.score THEN $score
				ELSE r.score
			END,
				r.updated_at = $now,
				r.created_at = COALESCE(r.created_at, $now)
			""",
			{
				"a_id": left_id,
				"b_id": right_id,
				"user_id": str(user_id),
				"score": float(score),
				"now": now,
			},
		)

	def upsert_entity_node(self, *, user_id: str, name: str, entity_type: str | None = None) -> None:
		"""Upsert an (:Entity) node by (user_id, name)."""
		self._run(
			"""
			MERGE (n:Entity {user_id: $user_id, name: $name})
			SET n.type = COALESCE(n.type, $entity_type)
			""",
			{
				"user_id": str(user_id),
				"name": str(name),
				"entity_type": str(entity_type) if entity_type is not None else None,
			},
		)

	def link_event_entity(self, *, event_id: str, user_id: str, entity_name: str) -> None:
		"""Create (:Event)-[:MENTIONS]->(:Entity) edge."""
		self._run(
			"""
			MERGE (e:Event {id: $event_id})
			SET e.user_id = COALESCE(e.user_id, $user_id)
			MERGE (n:Entity {user_id: $user_id, name: $entity_name})
			MERGE (e)-[:MENTIONS]->(n)
			""",
			{
				"event_id": str(event_id),
				"user_id": str(user_id),
				"entity_name": str(entity_name),
			},
		)

	def link_entity_relation(
		self,
		*,
		event_id: str,
		user_id: str,
		source: str,
		relationship: str,
		destination: str,
	) -> None:
		"""Create Entity-Entity relation and link it to the event via a Relation node."""
		rel_id = f"{user_id}|{event_id}|{source}|{relationship}|{destination}"
		self._run(
			"""
			MERGE (src:Entity {user_id: $user_id, name: $source})
			MERGE (dst:Entity {user_id: $user_id, name: $destination})
			MERGE (src)-[r:RELATIONSHIP {type: $relationship}]->(dst)
			MERGE (rel:Relation {id: $rel_id})
			SET rel.user_id = $user_id,
				rel.type = $relationship,
				rel.source = $source,
				rel.destination = $destination,
				rel.event_id = $event_id
			MERGE (rel)-[:REL_SOURCE]->(src)
			MERGE (rel)-[:REL_DEST]->(dst)
			MERGE (e:Event {id: $event_id})
			SET e.user_id = COALESCE(e.user_id, $user_id)
			MERGE (e)-[:MENTIONS_RELATION]->(rel)
			""",
			{
				"rel_id": rel_id,
				"user_id": str(user_id),
				"event_id": str(event_id),
				"source": str(source),
				"destination": str(destination),
				"relationship": str(relationship),
			},
		)

	def upsert_event_entities(
		self,
		*,
		event_id: str,
		user_id: str,
		entity_type_map: Dict[str, str],
		relations: List[Dict[str, str]],
	) -> None:
		"""Upsert entity nodes, link them to the event, and store relations."""
		for name, ent_type in (entity_type_map or {}).items():
			if not name:
				continue
			self.upsert_entity_node(user_id=user_id, name=name, entity_type=ent_type)
			self.link_event_entity(event_id=event_id, user_id=user_id, entity_name=name)

		for rel in relations or []:
			source = rel.get("source")
			destination = rel.get("destination")
			relationship = rel.get("relationship")
			if not source or not destination or not relationship:
				continue
			self.link_entity_relation(
				event_id=event_id,
				user_id=user_id,
				source=str(source),
				relationship=str(relationship),
				destination=str(destination),
			)


def extract_and_store_event_entities(
	*,
	event_graph: MemgraphEventGraph,
	extractor: Any,
	event_id: str,
	user_id: str,
	event_description: str,
) -> None:
	"""Extract entities/relations from event description and store to Memgraph."""
	result = extractor.extract_from_event(description=event_description, user_id=user_id)
	event_graph.upsert_event_entities(
		event_id=event_id,
		user_id=user_id,
		entity_type_map=result.entity_type_map,
		relations=result.relations,
	)


def _now_iso() -> str:
	return datetime.utcnow().isoformat() + "Z"


def build_graph_payload_from_memblock(
	*,
	block: Dict[str, Any],
	event: Dict[str, Any],
	user_id: Optional[str] = None,
) -> Dict[str, Any]:
	"""Normalize a memblock + event into a flat payload for graph storage.

	This helper aligns to the memblock schema shown in out/halu1b846c59/1.json.
	"""
	resolved_user_id = str(user_id or block.get("user_id") or "")

	block_id = block.get("block_id")
	block_key = _make_block_key(user_id=resolved_user_id, block_id=block_id)

	coverage = block.get("coverage") or {}
	temporal_index = block.get("temporal_index") or {}
	features = block.get("features") or {}

	block_event_start_time = temporal_index.get("block_event_start_time")
	block_event_end_time = temporal_index.get("block_event_end_time")

	features_categories: List[str] = []
	if isinstance(features, dict):
		cats = features.get("categories")
		if isinstance(cats, list):
			features_categories = [str(x) for x in cats if x is not None and str(x).strip()]
	block_topic_category = features_categories[0] if features_categories else None
	block_topic_kw_text = features.get("topic_kw_text") if isinstance(features, dict) else None

	labels = event.get("labels") or []
	label_names: List[str] = []
	label_scores: Dict[str, float] = {}
	if isinstance(labels, list):
		for item in labels:
			if isinstance(item, dict):
				label = item.get("label")
				confidence = item.get("confidence")
				if label:
					label_names.append(str(label))
					if confidence is not None:
						try:
							label_scores[str(label)] = float(confidence)
						except Exception:
							pass
	label_names = sorted({x for x in label_names if x and str(x).strip()})
	primary_label = None
	primary_label_score = None
	if label_scores:
		primary_label, primary_label_score = max(label_scores.items(), key=lambda kv: kv[1])

	time_metadata = event.get("time_metadata") or {}
	event_start_time = event.get("start_time") or time_metadata.get("startTime")
	event_end_time = event.get("end_time") or time_metadata.get("endTime")
	if not event_start_time:
		event_start_time = block_event_start_time
	if not event_end_time:
		event_end_time = block_event_end_time

	return {
		"user_id": resolved_user_id,
		"block_key": block_key,
		"block_id": block_id,
		"coverage": {
			"session_id": coverage.get("session_id"),
			"start_idx": coverage.get("start_idx"),
			"end_idx": coverage.get("end_idx"),
		},
		"block_event_start_time": block_event_start_time,
		"block_event_end_time": block_event_end_time,
		"block_topic_category": str(block_topic_category).strip() if block_topic_category else None,
		"block_topic_categories": features_categories or None,
		"block_topic_kw_text": str(block_topic_kw_text).strip() if block_topic_kw_text else None,
		"block_features_categories": features_categories or None,
		"block_features_json": _safe_json_dumps(features),
		"event_id": event.get("event_id"),
		"event_description": event.get("description"),
		"event_temporal_type": event.get("event_temporal_type"),
		"event_time_metadata": dict(time_metadata) if isinstance(time_metadata, dict) else None,
		"event_start_time": event_start_time,
		"event_end_time": event_end_time,
		"labels": label_names,
		"label_scores": label_scores,
		"primary_label": primary_label,
		"primary_label_score": primary_label_score,
	}


def _make_block_key(*, user_id: str, block_id: Any) -> str:
	"""Build a stable key for (user, block) grouping."""
	if block_id is not None:
		return f"{user_id}|{block_id}"
	return f"{user_id}|unknown"


def _safe_json_dumps(obj: Any) -> Optional[str]:
	try:
		import json

		return json.dumps(obj, ensure_ascii=False, sort_keys=True)
	except Exception:
		return None
