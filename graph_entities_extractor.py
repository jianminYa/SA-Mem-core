"""Entity and relation extraction for event descriptions.

This module provides a light wrapper around an LLM tool-calling interface
to extract entities and relations from event descriptions, plus a small
CLI for quick testing against a jsonl line.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from graph_utils import EXTRACT_RELATIONS_PROMPT

logger = logging.getLogger(__name__)


EXTRACT_ENTITIES_TOOL = {
	"type": "function",
	"function": {
		"name": "extract_entities",
		"description": "Extract entities and their types from the text.",
		"parameters": {
			"type": "object",
			"properties": {
				"entities": {
					"type": "array",
					"items": {
						"type": "object",
						"properties": {
							"entity": {"type": "string"},
							"entity_type": {"type": "string"},
						},
						"required": ["entity", "entity_type"],
						"additionalProperties": False,
					},
				}
			},
			"required": ["entities"],
			"additionalProperties": False,
		},
	},
}


EXTRACT_ENTITIES_STRUCT_TOOL = {
	"type": "function",
	"function": {
		"name": "extract_entities",
		"description": "Extract entities and their types from the text.",
		"strict": True,
		"parameters": EXTRACT_ENTITIES_TOOL["function"]["parameters"],
	},
}


RELATIONS_TOOL = {
	"type": "function",
	"function": {
		"name": "establish_relationships",
		"description": "Establish relationships among the entities based on the provided text.",
		"parameters": {
			"type": "object",
			"properties": {
				"entities": {
					"type": "array",
					"items": {
						"type": "object",
						"properties": {
							"source": {"type": "string"},
							"relationship": {"type": "string"},
							"destination": {"type": "string"},
						},
						"required": ["source", "relationship", "destination"],
						"additionalProperties": False,
					},
				}
			},
			"required": ["entities"],
			"additionalProperties": False,
		},
	},
}


RELATIONS_STRUCT_TOOL = {
	"type": "function",
	"function": {
		"name": "establish_relationships",
		"description": "Establish relationships among the entities based on the provided text.",
		"strict": True,
		"parameters": RELATIONS_TOOL["function"]["parameters"],
	},
}


_PRONOUN_RE = re.compile(r"\b(i|me|my|mine|myself)\b", re.IGNORECASE)
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-]*")


@dataclass
class ExtractionResult:
	entity_type_map: Dict[str, str]
	relations: List[Dict[str, str]]


def _normalize_entity(value: str) -> str:
	return str(value or "").strip().lower().replace(" ", "_")


def _sanitize_relationship(value: str) -> str:
	cleaned = re.sub(r"[^a-z0-9_]+", "_", str(value or "").strip().lower())
	cleaned = re.sub(r"_+", "_", cleaned).strip("_")
	if not cleaned:
		return "related_to"
	if cleaned[0].isdigit():
		return f"rel_{cleaned}"
	return cleaned


def _parse_tool_arguments(raw: Any) -> Optional[Dict[str, Any]]:
	if raw is None:
		return None
	if isinstance(raw, dict):
		return raw
	if isinstance(raw, str):
		try:
			return json.loads(raw)
		except Exception:
			return None
	return None


def _extract_tool_calls(response: Any) -> List[Dict[str, Any]]:
	if not response:
		return []
	if isinstance(response, dict) and "tool_calls" in response:
		return response.get("tool_calls") or []
	return []


class GraphEntitiesExtractor:
	def __init__(
		self,
		*,
		llm: Any = None,
		llm_provider: Optional[str] = None,
		custom_prompt: Optional[str] = None,
		use_heuristic_relations: bool = False,
		drop_weak_relations: bool = True,
	) -> None:
		self.llm = llm
		self.llm_provider = llm_provider or "openai"
		self.custom_prompt = custom_prompt
		self.use_heuristic_relations = bool(use_heuristic_relations)
		self.drop_weak_relations = bool(drop_weak_relations)

	def extract_from_event(self, *, description: str, user_id: str) -> ExtractionResult:
		entity_type_map = self._retrieve_nodes_from_text(description, user_id)
		relations = self._establish_relations_from_text(description, user_id, entity_type_map)
		return ExtractionResult(entity_type_map=entity_type_map, relations=relations)

	def _retrieve_nodes_from_text(self, text: str, user_id: str) -> Dict[str, str]:
		tools = [EXTRACT_ENTITIES_TOOL]
		if self.llm_provider in {"azure_openai_structured", "openai_structured"}:
			tools = [EXTRACT_ENTITIES_STRUCT_TOOL]

		system_prompt = (
			"You are a smart assistant who understands entities and their types in a given text. "
			f"If user message contains self reference such as 'I', 'me', 'my' etc. then use {user_id} "
			"as the source entity. Extract all the entities from the text. ***DO NOT*** answer the "
			"question itself if the given text is a question."
		)

		response = self._call_llm(
			messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": text}],
			tools=tools,
		)

		entity_type_map: Dict[str, str] = {}
		for tool_call in _extract_tool_calls(response):
			if tool_call.get("name") != "extract_entities":
				continue
			arguments = _parse_tool_arguments(tool_call.get("arguments")) or {}
			for item in arguments.get("entities", []) or []:
				if not isinstance(item, dict):
					continue
				entity = _normalize_entity(item.get("entity"))
				entity_type = _normalize_entity(item.get("entity_type")) or "__user__"
				if entity:
					entity_type_map[entity] = entity_type

		if not entity_type_map:
			entity_type_map = self._heuristic_entities(text, user_id)

		return entity_type_map

	def _establish_relations_from_text(
		self,
		text: str,
		user_id: str,
		entity_type_map: Dict[str, str],
	) -> List[Dict[str, str]]:
		if not entity_type_map:
			return []

		if self.custom_prompt:
			system_content = EXTRACT_RELATIONS_PROMPT.replace("USER_ID", user_id).replace(
				"CUSTOM_PROMPT", f"4. {self.custom_prompt}"
			)
			messages = [{"role": "system", "content": system_content}, {"role": "user", "content": text}]
		else:
			system_content = EXTRACT_RELATIONS_PROMPT.replace("USER_ID", user_id)
			messages = [
				{"role": "system", "content": system_content},
				{"role": "user", "content": f"List of entities: {list(entity_type_map.keys())}.\n\nText: {text}"},
			]

		tools = [RELATIONS_TOOL]
		if self.llm_provider in {"azure_openai_structured", "openai_structured"}:
			tools = [RELATIONS_STRUCT_TOOL]

		response = self._call_llm(messages=messages, tools=tools)

		relations: List[Dict[str, str]] = []
		for tool_call in _extract_tool_calls(response):
			if tool_call.get("name") != "establish_relationships":
				continue
			arguments = _parse_tool_arguments(tool_call.get("arguments")) or {}
			for item in arguments.get("entities", []) or []:
				if not isinstance(item, dict):
					continue
				source = _normalize_entity(item.get("source"))
				destination = _normalize_entity(item.get("destination"))
				relationship = _sanitize_relationship(item.get("relationship"))
				if source and destination and relationship:
					relations.append(
						{"source": source, "relationship": relationship, "destination": destination}
					)

		relations = self._filter_relations(relations, entity_type_map)
		if not relations and self.use_heuristic_relations:
			relations = self._heuristic_relations(entity_type_map)

		return relations

	def _filter_relations(
		self,
		relations: List[Dict[str, str]],
		entity_type_map: Dict[str, str],
	) -> List[Dict[str, str]]:
		allowed_entities = set(entity_type_map.keys())
		weak_relations = {
			"related_to",
			"associated_with",
			"connected_to",
			"linked_to",
			"mentions",
			"mentioned",
			"talked_about",
			"discussed",
			"said",
			"told",
			"asked",
			"chatted_with",
		}
		filtered: List[Dict[str, str]] = []
		for item in relations:
			source = _normalize_entity(item.get("source"))
			destination = _normalize_entity(item.get("destination"))
			relationship = _sanitize_relationship(item.get("relationship"))
			if not source or not destination or not relationship:
				continue
			if source == destination:
				continue
			if source not in allowed_entities or destination not in allowed_entities:
				continue
			if self.drop_weak_relations and relationship in weak_relations:
				continue
			filtered.append(
				{"source": source, "relationship": relationship, "destination": destination}
			)
		return filtered

	def _call_llm(self, *, messages: List[Dict[str, str]], tools: List[Dict[str, Any]]) -> Dict[str, Any]:
		if self.llm is None:
			return {}
		if hasattr(self.llm, "generate_response"):
			try:
				return self.llm.generate_response(messages=messages, tools=tools)
			except Exception as exc:
				logger.exception("LLM generate_response failed: %s", exc)
				return {}
		if callable(self.llm):
			try:
				return self.llm(messages=messages, tools=tools) or {}
			except Exception as exc:
				logger.exception("LLM callable failed: %s", exc)
				return {}
		return {}

	def _heuristic_entities(self, text: str, user_id: str) -> Dict[str, str]:
		entities: Dict[str, str] = {}

		if _PRONOUN_RE.search(text):
			entities[_normalize_entity(user_id)] = "__user__"

		tokens = _TOKEN_RE.findall(text)
		for token in tokens:
			if token.isupper() and len(token) > 1:
				entities[_normalize_entity(token)] = "misc"

		words = re.findall(r"\b[A-Z][a-z]+\b", text)
		for w in words:
			entities[_normalize_entity(w)] = "person"

		return entities

	def _heuristic_relations(self, entity_type_map: Dict[str, str]) -> List[Dict[str, str]]:
		entities = list(entity_type_map.keys())
		if len(entities) < 2:
			return []
		return [
			{
				"source": entities[0],
				"relationship": "related_to",
				"destination": entities[1],
			}
		]


class OpenAIToolCaller:
	def __init__(
		self,
		*,
		model: str,
		api_key: Optional[str] = None,
		base_url: Optional[str] = None,
	) -> None:
		try:
			from openai import OpenAI  # type: ignore
		except Exception as exc:
			raise ImportError(
				"OpenAI client not available. Install with: pip install openai"
			) from exc

		client_kwargs: Dict[str, Any] = {}
		if api_key:
			client_kwargs["api_key"] = api_key
		if base_url:
			client_kwargs["base_url"] = base_url

		self._client = OpenAI(**client_kwargs)
		self._model = model

	def generate_response(self, *, messages: List[Dict[str, str]], tools: List[Dict[str, Any]]) -> Dict[str, Any]:
		response = self._client.chat.completions.create(
			model=self._model,
			messages=messages,
			tools=tools,
			tool_choice="auto",
		)

		tool_calls: List[Dict[str, Any]] = []
		if response and response.choices:
			message = response.choices[0].message
			for call in message.tool_calls or []:
				tool_calls.append(
					{
						"name": getattr(call.function, "name", None),
						"arguments": getattr(call.function, "arguments", None),
					}
				)
		return {"tool_calls": tool_calls}


class OllamaToolCaller(OpenAIToolCaller):
	def __init__(
		self,
		*,
		model: str,
		base_url: str,
	) -> None:
		normalized = _normalize_ollama_base_url(base_url)
		super().__init__(model=model, base_url=normalized, api_key="ollama")


def _load_jsonl_line(path: str, line_index: int) -> Dict[str, Any]:
	with open(path, "r", encoding="utf-8") as f:
		for idx, raw in enumerate(f):
			if idx == line_index:
				return json.loads(raw)
	raise IndexError(f"Line {line_index} not found in {path}")


def _iter_event_descriptions(record: Dict[str, Any]) -> Iterable[Tuple[str, str]]:
	user_id = str(record.get("user_id") or "")
	events = record.get("events") or []
	for ev in events:
		if isinstance(ev, dict):
			desc = str(ev.get("description") or "").strip()
			if desc:
				yield user_id, desc


def _normalize_ollama_base_url(base_url: str) -> str:
	url = str(base_url or "").strip()
	if not url:
		return "http://localhost:11434/v1"
	if url.endswith("/v1"):
		return url
	return url.rstrip("/") + "/v1"


def main() -> None:
	parser = argparse.ArgumentParser(description="Extract entities and relations from event descriptions.")
	parser.add_argument("--jsonl", required=True, help="Path to jsonl file.")
	parser.add_argument("--line", type=int, default=0, help="0-based line index in jsonl.")
	parser.add_argument("--user-id", default=None, help="Override user_id in the jsonl line.")
	parser.add_argument("--max-events", type=int, default=10, help="Max events to extract.")
	parser.add_argument("--use-heuristic", action="store_true", help="Use heuristic fallback only.")
	parser.add_argument("--openai-model", default=None, help="OpenAI model name for tool-calling.")
	parser.add_argument("--openai-api-key", default=None, help="OpenAI API key override.")
	parser.add_argument("--openai-base-url", default=None, help="OpenAI base URL override.")
	parser.add_argument(
		"--ollama-model",
		default="alibayram/Qwen3-30B-A3B-Instruct-2507:latest",
		help="Ollama model name for tool-calling.",
	)
	parser.add_argument(
		"--ollama-base-url",
		default="http://localhost:11434",
		help="Ollama base URL (without /v1 is ok).",
	)
	args = parser.parse_args()

	record = _load_jsonl_line(args.jsonl, args.line)
	override_user_id = args.user_id
	llm = None
	if not args.use_heuristic:
		if args.openai_model:
			llm = OpenAIToolCaller(
				model=args.openai_model,
				api_key=args.openai_api_key,
				base_url=args.openai_base_url,
			)
		else:
			llm = OllamaToolCaller(
				model=args.ollama_model,
				base_url=args.ollama_base_url,
			)
	extractor = GraphEntitiesExtractor(llm=llm)

	count = 0
	for user_id, desc in _iter_event_descriptions(record):
		if override_user_id:
			user_id = override_user_id
		result = extractor.extract_from_event(description=desc, user_id=user_id)
		print("=== EVENT ===")
		print(desc)
		print("entities:", result.entity_type_map)
		print("relations:", result.relations)
		count += 1
		if count >= args.max_events:
			break


if __name__ == "__main__":
	main()
