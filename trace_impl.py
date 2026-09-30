from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Tuple

from sklearn.metrics.pairwise import cosine_similarity


def _mx():
    """
    Lazy import to avoid circular imports:
    memblock_extractor imports trace_impl, but trace_impl must not import memblock_extractor at import-time.
    """
    import memblock_extractor as mx  # local import by design

    return mx


class TraceLinker:
    """离线将 box 按 topic+keywords 近邻 + LLM 判定合并为 trace。"""

    def __init__(self, worker: Any, trace_metrics: List[str] | None = None):
        mx = _mx()
        self.worker = worker
        self.trace_metrics = trace_metrics or mx.Config.TRACE_METRICS
        self.relation_cache: Dict[Tuple[Any, int, int], bool] = {}

    @staticmethod
    def _box_text(box: Dict[str, Any]) -> str:
        feat = box.get("features", {})
        return feat.get("content_text", "")

    @staticmethod
    def _get_box_events(box: Dict[str, Any]) -> List[Any]:
        """Get events from box, supporting both old format (features.events) and new format (top-level events)."""
        # New format: top-level events list
        top_events = box.get("events", [])
        if top_events:
            return top_events
        # Old format: features.events
        return box.get("features", {}).get("events", []) or []

    @staticmethod
    def _get_box_start_time(box: Dict[str, Any]) -> str:
        """Get block event start time, supporting temporal_index nesting."""
        # Try temporal_index first (LoCoMo/LME format)
        ti = box.get("temporal_index", {})
        if ti:
            v = ti.get("block_event_start_time") or ti.get("sessionstart_time")
            if v:
                return str(v)
        # Fallback: top-level (old format)
        v = box.get("block_event_start_time") or box.get("box_event_start_time")
        if v:
            return str(v)
        return "Unknown"

    @staticmethod
    def _event_to_line(ev: Any) -> str:
        """Convert event dict to descriptive line, supporting time_metadata nesting."""
        if isinstance(ev, str):
            return ev.strip()
        if not isinstance(ev, dict):
            return ""
        desc = str(ev.get("description", "") or "").strip()
        if not desc:
            return ""
        # Try time_metadata (LoCoMo/LME format)
        tm = ev.get("time_metadata", {})
        start = str(tm.get("startTime") or tm.get("observedTime") or "Unknown")
        end = str(tm.get("endTime") or start)
        # Fallback: top-level start_time/end_time (old format)
        if start == "Unknown":
            start = str(ev.get("start_time", "Unknown") or "Unknown")
            end = str(ev.get("end_time", start) or start)
        return f"[{start}, {end}] {desc}"

    def _entry(self, box: Dict[str, Any], order: int, events: List[str] | None = None) -> Dict[str, Any]:
        mx = _mx()
        events_list = events if events is not None else self._get_box_events(box)
        events_clean = [self._event_to_line(e) for e in (events_list or []) if self._event_to_line(e)]
        return {
            "block_id": mx._get_block_id(box),
            "start_time": self._get_box_start_time(box),
            "events": events_clean,
            "order": order,
        }

    @staticmethod
    def _metric_text(box: Dict[str, Any], metric: str) -> str:
        mx = _mx()
        feat = box.get("features", {})
        content = feat.get("content_text", "")
        # Support both old (features.events) and new (top-level events) format
        events_raw = box.get("events", []) or feat.get("events", [])
        evt = mx._events_to_text(events_raw)
        topic_kw = feat.get("topic_kw_text", "")
        if metric == "content_event_topic_kw":
            return f"{content} {evt} {topic_kw}".strip()
        return content

    @staticmethod
    def _trace_event_lines(trace: Dict[str, Any]) -> List[str]:
        lines: List[str] = []
        for entry in trace.get("entries", []):
            ts = str(entry.get("start_time", "Unknown"))
            for ev in entry.get("events") or []:
                ev_clean = str(ev).strip()
                if ev_clean:
                    lines.append(f"{ts}: {ev_clean}")
        return lines

    def _llm_event_filter(self, trace: Dict[str, Any], events: List[str], user_id: Any) -> Tuple[set, set]:
        mx = _mx()
        chain_text = "\n".join(self._trace_event_lines(trace)) or "None"
        events_text = "\n".join(events) or "None"
        prompt = mx.Config.PROMPT_TRACE_EVENT_FILTER.format(content_a=chain_text, content_b=events_text)
        res = self.worker.chat_completion(
            prompt,
            note=f"S{user_id}_TraceLinker_EventFilter",
            extra={"prompt_tokens_est": self.worker.count_tokens(prompt), "stage": "trace"},
        )
        mx.TraceLogger.log(
            mx.Config.TRACE_PROMPT_LOG_FILE,
            {
                "type": "event_filter",
                "user_id": user_id,
                "trace_id": trace.get("trace_id"),
                "prompt": prompt,
                "response": res,
            },
        )
        related = set()
        unrelated = set()
        try:
            parsed = json.loads(res)
            related_list = parsed.get("related_events") or []
            unrelated_list = parsed.get("unrelated_events") or []
            for ev in events:
                if ev in related_list:
                    related.add(ev)
                elif ev in unrelated_list:
                    unrelated.add(ev)
        except Exception:
            related = set(events)

        if not related and not unrelated:
            related = set(events)
        return related, unrelated

    def _llm_init_chain(self, events: List[str], user_id: Any) -> Dict[str, Any]:
        mx = _mx()
        prompt = mx.Config.PROMPT_TRACE_INIT.format(events="\n".join(events))
        res = self.worker.chat_completion(
            prompt,
            note=f"S{user_id}_TraceLinker_Init",
            extra={"prompt_tokens_est": self.worker.count_tokens(prompt), "stage": "trace"},
        )
        mx.TraceLogger.log(
            mx.Config.TRACE_PROMPT_LOG_FILE,
            {
                "type": "init_chain",
                "user_id": user_id,
                "prompt": prompt,
                "response": res,
            },
        )
        try:
            return json.loads(res)
        except Exception:
            return {}

    def run(self, skip_user_ids: set | None = None):
        mx = _mx()
        if not os.path.exists(mx.Config.FINAL_CONTENT_FILE):
            mx.logger.error("❌ Need build outputs first.")
            return

        with open(mx.Config.FINAL_CONTENT_FILE, "r", encoding="utf-8") as f:
            boxes = [json.loads(line) for line in f if line.strip()]

        metrics = [m for m in (self.trace_metrics or []) if m == "content_event_topic_kw"] or ["content_event_topic_kw"]

        limit = mx.Config.LIMIT_CONVERSATIONS
        total_boxes = 0
        user_ids = sorted({b["user_id"] for b in boxes}, key=lambda x: str(x))
        if limit is not None:
            user_ids = user_ids[:limit]

        # Resume support: skip already-completed users
        if skip_user_ids:
            skipped = [uid for uid in user_ids if uid in skip_user_ids or str(uid) in skip_user_ids]
            if skipped:
                mx.logger.info("⏭️ Resuming: skipping %d already-completed users: %s", len(skipped), skipped)
            user_ids = [uid for uid in user_ids if uid not in skip_user_ids and str(uid) not in skip_user_ids]
            if not user_ids:
                mx.logger.info("✅ All users already completed. Nothing to do.")
                return

        for user_id in user_ids:
            store = mx.EmbeddingStore(self.worker, user_id)
            sample_boxes = [b for b in boxes if b["user_id"] == user_id]
            sample_boxes.sort(key=lambda x: mx._get_block_id(x))
            total_boxes += len(sample_boxes)

            sample_traces: List[Dict[str, Any]] = []

            for metric in metrics:
                traces_merged: List[Dict[str, Any]] = []

                for box in sample_boxes:
                    bid = mx._get_block_id(box)
                    events_raw = self._get_box_events(box)
                    events = [self._event_to_line(e) for e in events_raw if self._event_to_line(e)]
                    if not events:
                        continue

                    trace_lookup = {t["trace_id"]: t for t in traces_merged}
                    selected_trace_ids = set()

                    # Stage 1: Select candidate traces based on similarity (event->existing trace events)
                    for ev_idx, ev in enumerate(events):
                        ev_vec = store.get_vector(
                            f"S{user_id}_B{bid}_E{ev_idx}",
                            "event",
                            ev,
                            note=f"S{user_id}_B{bid}_event",
                        )
                        if not ev_vec:
                            continue

                        best_trace_id = None
                        best_score = -1.0
                        for tr in traces_merged:
                            trace_best = -1.0
                            for entry_idx, entry in enumerate(tr.get("entries", [])):
                                for tev_idx, tev in enumerate(entry.get("events") or []):
                                    key = f"S{user_id}_T{tr['trace_id']}_E{entry_idx}_{tev_idx}"
                                    tvec = store.get_vector(
                                        key,
                                        "event",
                                        tev,
                                        note=f"S{user_id}_T{tr['trace_id']}_event",
                                    )
                                    if not tvec:
                                        continue
                                    try:
                                        score = cosine_similarity([ev_vec], [tvec])[0][0]
                                    except Exception:
                                        score = 0.0
                                    if score > trace_best:
                                        trace_best = score
                            if trace_best > best_score:
                                best_score = trace_best
                                best_trace_id = tr["trace_id"]

                        if best_trace_id is not None and best_score >= mx.Config.TRACE_SIMILARITY_THRESHOLD:
                            selected_trace_ids.add(best_trace_id)

                    # Stage 2: LLM Filter for selected traces
                    matched_events = set()
                    for tr_id in selected_trace_ids:
                        trace = trace_lookup.get(tr_id)
                        if not trace:
                            continue

                        related, _ = self._llm_event_filter(trace, events, user_id)
                        if related:
                            if bid not in trace["box_ids"]:
                                trace["box_ids"].append(bid)
                            trace["entries"].append(self._entry(box, len(trace["entries"]), list(related)))
                            matched_events.update(related)

                    unmatched_events = [e for e in events if e and e not in matched_events]
                    if unmatched_events:

                        def _create_trace_with_events(ev_list: List[str]):
                            ev_clean = [str(e).strip() for e in ev_list if str(e).strip()]
                            if not ev_clean:
                                return
                            entry = self._entry(box, 0, ev_clean)
                            trace_local = {
                                "user_id": user_id,
                                "metric": metric,
                                "trace_id": len(traces_merged),
                                "box_ids": [bid],
                                "entries": [entry],
                            }
                            traces_merged.append(trace_local)

                        if len(unmatched_events) == 1:
                            _create_trace_with_events(unmatched_events)
                        else:
                            init_res = self._llm_init_chain(unmatched_events, user_id) or {}
                            primary_chain = init_res.get("primary_chain") or []
                            secondary_chains = init_res.get("secondary_chains") or []
                            isolated_events = init_res.get("isolated_events") or []

                            chains_to_create: List[List[str]] = []
                            if primary_chain:
                                chains_to_create.append(primary_chain)
                            for ch in secondary_chains:
                                if ch:
                                    chains_to_create.append(ch)
                            if not chains_to_create and isolated_events:
                                chains_to_create.append(isolated_events)

                            for chain_events in chains_to_create:
                                _create_trace_with_events(chain_events)

                            if chains_to_create and isolated_events:
                                used_events = {e for chain in chains_to_create for e in chain}
                                remaining_iso = [e for e in isolated_events if e not in used_events]
                                for iso in remaining_iso:
                                    _create_trace_with_events([iso])

                # derive entries_text
                for t in traces_merged:
                    texts = []
                    for entry in t["entries"]:
                        if entry["events"]:
                            texts.append(f"{entry['start_time']}: {entry['events'][0]}")
                            for e in entry["events"][1:]:
                                texts.append(e)
                    t["entries_text"] = " ".join(texts)

                sample_traces.extend(traces_merged)

            store.flush()

            if sample_traces:
                os.makedirs(os.path.dirname(mx.Config.TIME_TRACE_FILE), exist_ok=True)
                with open(mx.Config.TIME_TRACE_FILE, "a", encoding="utf-8") as f_out:
                    for t in sample_traces:
                        f_out.write(json.dumps(t, ensure_ascii=False) + "\n")
                mx.logger.info("✅ Trace saved for sample %s (%d traces)", user_id, len(sample_traces))

        mx.logger.info("✅ Trace linking completed. Output -> %s", mx.Config.TIME_TRACE_FILE)

        llm_stats = mx.TokenAnalyzer.get_stage_stats("trace")
        avg_total_tokens_per_box = llm_stats.get("total", 0) / max(total_boxes, 1)
        mx.logger.info(
            "ℹ️ Trace LLM stats | calls=%s prompt=%s completion=%s total=%s avg_total_tokens_per_box=%.3f",
            llm_stats.get("calls", 0),
            llm_stats.get("prompt", 0),
            llm_stats.get("completion", 0),
            llm_stats.get("total", 0),
            avg_total_tokens_per_box,
        )

        summary = {
            "run_id": mx.Config.RUN_ID,
            "timestamp": datetime.now().isoformat(),
            "total_boxes": total_boxes,
            "llm_calls": llm_stats.get("calls", 0),
            "llm_prompt_tokens": llm_stats.get("prompt", 0),
            "llm_completion_tokens": llm_stats.get("completion", 0),
            "llm_total_tokens": llm_stats.get("total", 0),
            "avg_total_tokens_per_box": round(avg_total_tokens_per_box, 3),
        }
        os.makedirs(os.path.dirname(mx.Config.TRACE_STATS_FILE), exist_ok=True)
        with open(mx.Config.TRACE_STATS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")