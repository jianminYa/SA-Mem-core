from __future__ import annotations

import csv
import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import tiktoken


def _mx():
    """
    Lazy import to avoid circular imports:
    memblock_extractor imports generate_impl, but generate_impl must not import memblock_extractor at import-time.
    """
    import memblock_extractor as mx  # local import by design

    return mx


class AnswerGenerator:
    """统一的生成与评估模块，可选问题重排。"""

    def __init__(
        self,
        worker: Any,
        answer_topn: int | List[int] | None = None,
        text_modes: List[str] | None = None,
        stage_label: str = "gen",
    ):
        mx = _mx()
        self.worker = worker
        if isinstance(answer_topn, list):
            self.answer_topn_list = answer_topn
        else:
            self.answer_topn_list = [answer_topn or mx.Config.ANSWER_TOP_N or mx.Config.TOP_K_RETRIEVE]
        self.text_modes = text_modes or mx.Config.GEN_TEXT_MODES
        self.trace_metrics = mx.Config.TRACE_METRICS
        self.encoding = tiktoken.encoding_for_model(mx.Config.LLM_MODEL)

        self.box_map: Dict[Any, Dict[int, str]] = {}
        self.qa_map: Dict[Any, List[Dict[str, Any]]] = {}
        self.boxes_by_user: Dict[Any, List[Dict[str, Any]]] = {}
        self.trace_map: Dict[Any, Dict[str, List[Dict[str, Any]]]] = {}

        self.content_totals: Dict[Any, int] = defaultdict(int)

        self.aggregate: Dict[Tuple[str, str, str, str, int], Dict[str, float]] = defaultdict(
            lambda: {"f1_sum": 0.0, "bleu_sum": 0.0, "ctx_tokens_sum": 0.0, "count": 0}
        )
        self.aggregate_by_category: Dict[Tuple[str, str, str, str, int, str], Dict[str, float]] = defaultdict(
            lambda: {"f1_sum": 0.0, "bleu_sum": 0.0, "ctx_tokens_sum": 0.0, "count": 0}
        )

        self.conv_ctx_total: Dict[Tuple[str, Any], Dict[str, float]] = defaultdict(lambda: {"tokens": 0.0, "count": 0.0})
        self.conv_ctx_by_mode: Dict[Tuple[str, Any, str], Dict[str, float]] = defaultdict(lambda: {"tokens": 0.0, "count": 0.0})

        self.stage_label = stage_label

    @staticmethod
    def _tokens(text: str) -> List[str]:
        cleaned = re.sub(r"[^A-Za-z0-9]+", " ", str(text or "").lower())
        return [t for t in cleaned.split() if t]

    @classmethod
    def _f1(cls, pred: str, gold: Any) -> float:
        pred_tokens = cls._tokens(pred)
        gold_list = gold if isinstance(gold, list) else [gold]
        best = 0.0
        for g in gold_list:
            gold_tokens = cls._tokens(g)
            if not gold_tokens or not pred_tokens:
                overlap = 0
            else:
                overlap = 0
                gold_counts: Dict[str, int] = {}
                for t in gold_tokens:
                    gold_counts[t] = gold_counts.get(t, 0) + 1
                for t in pred_tokens:
                    if t in gold_counts and gold_counts[t] > 0:
                        overlap += 1
                        gold_counts[t] -= 1
            if overlap == 0:
                f1 = 0.0
            else:
                precision = overlap / len(pred_tokens)
                recall = overlap / len(gold_tokens)
                f1 = 2 * precision * recall / (precision + recall)
            best = max(best, f1)
        return best

    @classmethod
    def _bleu(cls, pred: str, gold: Any) -> float:
        import nltk
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

        pred = str(pred)
        refs = [str(g) for g in gold] if isinstance(gold, list) else [str(gold)]
        pred_tokens = nltk.word_tokenize(pred.lower())
        refs_tokens = [nltk.word_tokenize(r.lower()) for r in refs]
        smooth = SmoothingFunction().method1
        try:
            return sentence_bleu(refs_tokens, pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smooth)
        except Exception:
            return 0.0

    def _load_boxes(self):
        mx = _mx()
        if not os.path.exists(mx.Config.FINAL_CONTENT_FILE):
            return
        with open(mx.Config.FINAL_CONTENT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                b = json.loads(line)
                raw_sid = b.get("user_id")
                bid = mx._get_block_id(b)
                if raw_sid is None:
                    continue
                sid = str(raw_sid)
                # Get content_text (original conversation text)
                content_text = b.get("features", {}).get("content_text", "")

                # Store content_text for "content" mode (Membox style)
                # This will be used when mode == "content" in _generate_for_ranking
                if not hasattr(self, 'box_map_content'):
                    self.box_map_content = {}
                self.box_map_content.setdefault(sid, {})[bid] = content_text

                # Build event-based text for "event" mode (original SA-Mem style)
                events = b.get("events", [])
                topic_kw = b.get("features", {}).get("topic_kw_text", "")

                # Format events with temporal metadata
                event_descriptions = []
                for e in events:
                    desc = e.get("description", "")
                    if not desc:
                        continue

                    # Extract temporal metadata
                    time_meta = e.get("time_metadata", {})
                    start_time = time_meta.get("startTime")
                    end_time = time_meta.get("endTime")

                    # Format with temporal prefix
                    if start_time and end_time and start_time != end_time:
                        # Range: [2022-01-01 to 2022-12-31]
                        time_prefix = f"[{start_time} to {end_time}]"
                    elif start_time:
                        # Single date: [2023-05-07]
                        time_prefix = f"[{start_time}]"
                    else:
                        # No temporal metadata, use block-level fallback
                        block_time = b.get("temporal_index", {}).get("block_event_start_time")
                        time_prefix = f"[{block_time}]" if block_time else ""

                    # Combine time prefix with description
                    if time_prefix:
                        event_descriptions.append(f"{time_prefix} {desc}")
                    else:
                        event_descriptions.append(desc)

                # Combine topic keywords and event descriptions for "event" mode
                text_parts = []
                if topic_kw:
                    text_parts.append(f"Topics: {topic_kw}")
                if event_descriptions:
                    text_parts.append("Events:\n" + "\n".join(f"- {desc}" for desc in event_descriptions))

                event_text = "\n\n".join(text_parts) if text_parts else ""

                # Store event-based text for "event" mode
                if not hasattr(self, 'box_map_event'):
                    self.box_map_event = {}
                self.box_map_event.setdefault(sid, {})[bid] = event_text

                # Keep box_map for backward compatibility (defaults to content mode)
                self.box_map.setdefault(sid, {})[bid] = content_text

                self.boxes_by_user.setdefault(sid, []).append({"block_id": bid, "coverage": b.get("coverage", {})})

                # Use content_text for token counting (more accurate for content mode)
                self.content_totals[sid] += len(self.encoding.encode(content_text))

    def _load_qa(self):
        """
        Load QA pairs for LoCoMo with Membox-aligned indexing:
        - user_id is conversation index as string: "0", "1", "2", ...
        - qa_idx is local index within each conversation
        - category filtering is done in run(), not here
        """
        mx = _mx()
        if not os.path.exists(mx.Config.RAW_DATA_FILE):
            return

        with open(mx.Config.RAW_DATA_FILE, "r", encoding="utf-8") as f:
            raw_list = json.load(f)[: mx.Config.LIMIT_CONVERSATIONS]

        for conv_idx, data in enumerate(raw_list):
            user_id = str(conv_idx)
            qa_list = data.get("qa", []) or []
            self.qa_map[user_id] = qa_list

            valid_count = sum(1 for qa in qa_list if qa.get("category") != 5)
            mx.logger.info(
                f"ℹ️ Loaded {len(qa_list)} QA pairs for user_id={user_id} "
                f"(conversation {conv_idx}, non-cat5={valid_count})"
            )

    def _load_traces(self):
        mx = _mx()
        traces: Dict[Any, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
        if os.path.exists(mx.Config.TIME_TRACE_FILE):
            with open(mx.Config.TIME_TRACE_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    t = json.loads(line)
                    raw_sid = t.get("user_id")
                    metric = t.get("metric")
                    if raw_sid is None or metric is None:
                        continue
                    sid = str(raw_sid)
                    traces[sid][metric].append(t)
        self.trace_map = traces

    def _trace_events_for_box(self, sid: Any, bid: int, metric: str) -> List[str]:
        """Return raw events lines for the FIRST trace that contains bid.

        Note: events lines kept as-is (already include `[start, end] desc`).
        Outer entry.start_time is intentionally dropped to avoid double timestamping.
        """
        traces = self.trace_map.get(sid, {}).get(metric, [])
        for tr in traces:
            if bid not in tr.get("box_ids", []):
                continue
            events_texts: List[str] = []
            for entry in tr.get("entries", []):
                evs = entry.get("events") or []
                for ev in evs:
                    ev_clean = str(ev).strip()
                    if ev_clean:
                        events_texts.append(ev_clean)
            return events_texts
        return []

    def _trace_for_box(self, sid: Any, bid: int, metric: str) -> Dict[str, Any] | None:
        """Return the FIRST trace dict that contains bid (or None)."""
        traces = self.trace_map.get(sid, {}).get(metric, [])
        for tr in traces:
            if bid in tr.get("box_ids", []):
                return tr
        return None

    # Pre-compiled regex for parsing inner [start, end] prefix in event text
    _INNER_TIME_RE = re.compile(r"^\[([^\],]+),\s*([^\]]+)\]\s*(.*)$")

    @classmethod
    def _format_event_v3b(cls, raw_event: str) -> str:
        """Reformat one event line to v3b style: drop entry.start_time prefix and
        rewrite inner ``[start, end] desc`` as ``start: desc`` or ``start..end: desc``.

        Falls back to the raw event when parsing fails (no inner brackets).
        """
        ev = (raw_event or "").strip()
        if not ev:
            return ""
        m = cls._INNER_TIME_RE.match(ev)
        if not m:
            return ev
        s = m.group(1).strip()
        e = m.group(2).strip()
        desc = m.group(3).strip()
        if not desc:
            return ev
        # Treat missing/Unknown end as a point-in-time, not a range.
        if not e or e.lower() == "unknown":
            e = s
        if s == e:
            return f"{s}: {desc}"
        return f"{s}..{e}: {desc}"

    def _trace_events_lines(self, trace: Dict[str, Any]) -> List[str]:
        """Flatten a trace's events into v3b-formatted lines, dedup-preserving order.

        v3b format: drop outer ``entry.start_time`` prefix, keep inner event time as
        ``start: desc`` or ``start..end: desc``. This preserves precise event time
        (often more accurate than block-level start_time) while removing the
        outer-prefix duplication observed in the legacy A2 format.
        """
        seen = set()
        out: List[str] = []
        for entry in trace.get("entries", []):
            for ev in entry.get("events") or []:
                line = self._format_event_v3b(str(ev))
                if not line or line in seen:
                    continue
                seen.add(line)
                out.append(line)
        return out

    def _build_trace_contexts(self, sid: Any, top_ids: List[int], trace_metric: str, mode: str) -> List[str]:
        """Build LLM context for trace modes with three optimizations:
        1) trace-id level dedup (a trace shared by multiple top_ids is only emitted once)
        2) per-trace events truncated to TRACE_EVENT_TOPK (default 50)
        3) drop outer entry.start_time (events already contain `[start, end]`)
        """
        mx = _mx()
        topk = int(getattr(mx.Config, "TRACE_EVENT_TOPK", 50) or 50)

        # 1) Collect unique traces hit by top_ids, preserving the order of first hit
        seen_trace_ids = set()
        ordered_traces: List[Dict[str, Any]] = []
        block_to_trace: Dict[int, int] = {}  # bid -> trace_id (for content+trace mapping)
        for bid in top_ids:
            tr = self._trace_for_box(sid, bid, trace_metric)
            if tr is None:
                continue
            tid = tr.get("trace_id")
            block_to_trace[bid] = tid
            if tid in seen_trace_ids:
                continue
            seen_trace_ids.add(tid)
            ordered_traces.append(tr)

        # 2) Build trace event blocks (truncated to topk)
        trace_blocks: List[str] = []
        for tr in ordered_traces:
            ev_lines = self._trace_events_lines(tr)
            if not ev_lines:
                continue
            if topk > 0 and len(ev_lines) > topk:
                ev_lines = ev_lines[:topk]
            tid = tr.get("trace_id")
            trace_blocks.append(f"Trace {tid}:\n" + "\n".join(ev_lines))

        contexts: List[str] = []
        if mode == "content_trace_event":
            # Emit each top block's content. Then attach all hit traces ONCE at the end.
            for bid in top_ids:
                content = self.box_map.get(sid, {}).get(bid)
                if content:
                    contexts.append(content)
            if trace_blocks:
                contexts.append("Events:\n" + "\n\n".join(trace_blocks))
        elif mode == "trace_event":
            if trace_blocks:
                contexts.append("Events:\n" + "\n\n".join(trace_blocks))

        return contexts

    def _format_graph_context(self, graph_payload: Dict[str, Any], top_ids: List[int]) -> str:
        if not graph_payload or not graph_payload.get("graph_enabled"):
            return ""
        parts: List[str] = []

        seed_ids = graph_payload.get("seed_event_ids") or []
        if seed_ids:
            parts.append("Seed events (from retrieved blocks):")
            parts.extend([f"- {eid}" for eid in seed_ids if str(eid).strip()])

        expanded_events = graph_payload.get("expanded_events") or []
        if expanded_events:
            parts.append("Expanded events (1-hop neighbors):")
            event_lines = []
            for ev in expanded_events:
                desc = str(ev.get("event_description") or ev.get("event_id") or "").strip()
                if desc:
                    event_lines.append(f"- {desc}")
            if event_lines:
                parts.extend(event_lines)

        similar_edges = graph_payload.get("similar_events") or []
        if similar_edges:
            parts.append("Similar events:")
            edge_lines = []
            for edge in similar_edges:
                fe = str(edge.get("from_event_id") or "").strip()
                te = str(edge.get("to_event_id") or "").strip()
                score = edge.get("score")
                if fe and te:
                    edge_lines.append(f"- {fe} -> {te} (score={score})")
            if edge_lines:
                parts.extend(edge_lines)

        relations = graph_payload.get("relations") or []
        if relations:
            parts.append("Relations:")
            rel_lines = []
            for rel in relations:
                src = str(rel.get("source") or "").strip()
                rel_type = str(rel.get("relationship") or "").strip()
                dst = str(rel.get("destination") or "").strip()
                if src and rel_type and dst:
                    rel_lines.append(f"- {src} -- {rel_type} -- {dst}")
            if rel_lines:
                parts.extend(rel_lines)

        return "\n".join(parts)

    def _log_token_counts(self, context_text: str, question: str) -> Dict[str, Any]:
        mx = _mx()
        return {
            "memories_tokens": len(self.encoding.encode(context_text)),
            "question_tokens": len(self.encoding.encode(question)),
            "prompt_tokens_est": len(self.encoding.encode(mx.Config.PROMPT_QA_ANSWER.format(memories=context_text, question=question))),
        }

    def _record_metrics(
        self,
        ranking_strategy: str,
        metric: str,
        trace_metric: str | None,
        mode: str,
        topn: int,
        f1: float,
        bleu: float,
        ctx_tokens: int,
        sid: Any,
        category: Any,
    ):
        key = (ranking_strategy, metric, trace_metric or "", mode, topn)
        agg = self.aggregate[key]
        agg["f1_sum"] += f1
        agg["bleu_sum"] += bleu
        agg["ctx_tokens_sum"] += ctx_tokens
        agg["count"] += 1

        cat_label = "uncategorized" if category is None else str(category)
        cat_key = (ranking_strategy, metric, trace_metric or "", mode, topn, cat_label)
        cat_agg = self.aggregate_by_category[cat_key]
        cat_agg["f1_sum"] += f1
        cat_agg["bleu_sum"] += bleu
        cat_agg["ctx_tokens_sum"] += ctx_tokens
        cat_agg["count"] += 1

        conv_key = (ranking_strategy, sid)
        conv_stat_total = self.conv_ctx_total[conv_key]
        conv_stat_total["tokens"] += ctx_tokens
        conv_stat_total["count"] += 1

        conv_mode_key = (ranking_strategy, sid, mode)
        conv_stat_mode = self.conv_ctx_by_mode[conv_mode_key]
        conv_stat_mode["tokens"] += ctx_tokens
        conv_stat_mode["count"] += 1

    def _write_summary(self):
        mx = _mx()
        if not self.aggregate:
            return

        records = []
        for (ranking_strategy, metric, trace_metric, mode, topn), v in self.aggregate.items():
            count = max(v.get("count", 0), 1)
            records.append(
                {
                    "run_id": mx.Config.RUN_ID,
                    "stage": self.stage_label,
                    "ranking_strategy": ranking_strategy,
                    "metric": metric,
                    "trace_metric": trace_metric,
                    "text_mode": mode,
                    "topn": topn,
                    "avg_f1": round(v.get("f1_sum", 0) / count, 4),
                    "avg_bleu": round(v.get("bleu_sum", 0) / count, 4),
                    "avg_context_tokens": round(v.get("ctx_tokens_sum", 0) / count, 2),
                    "count": v.get("count", 0),
                }
            )

        for (ranking_strategy, metric, trace_metric, mode, topn, category), v in self.aggregate_by_category.items():
            count = max(v.get("count", 0), 1)
            records.append(
                {
                    "run_id": mx.Config.RUN_ID,
                    "stage": self.stage_label,
                    "ranking_strategy": ranking_strategy,
                    "metric": metric,
                    "trace_metric": trace_metric,
                    "text_mode": mode,
                    "topn": topn,
                    "category": category,
                    "avg_f1": round(v.get("f1_sum", 0) / count, 4),
                    "avg_bleu": round(v.get("bleu_sum", 0) / count, 4),
                    "avg_context_tokens": round(v.get("ctx_tokens_sum", 0) / count, 2),
                    "count": v.get("count", 0),
                    "type": "category_metrics",
                }
            )

        for (ranking_strategy, sid), stat in self.conv_ctx_total.items():
            count = max(stat.get("count", 0), 1)
            avg_ctx = stat.get("tokens", 0) / count
            content_total = self.content_totals.get(sid, 1)
            records.append(
                {
                    "run_id": mx.Config.RUN_ID,
                    "stage": self.stage_label,
                    "ranking_strategy": ranking_strategy,
                    "user_id": sid,
                    "avg_context_tokens": round(avg_ctx, 2),
                    "content_tokens_total": content_total,
                    "avg_context_ratio_over_content": round(avg_ctx / max(content_total, 1), 4),
                    "count": stat.get("count", 0),
                    "type": "conversation_context_usage",
                }
            )

        for (ranking_strategy, sid, mode), stat in self.conv_ctx_by_mode.items():
            content_total = self.content_totals.get(sid, 1)
            count = max(stat.get("count", 0), 1)
            avg_ctx = stat.get("tokens", 0) / count
            records.append(
                {
                    "run_id": mx.Config.RUN_ID,
                    "stage": self.stage_label,
                    "ranking_strategy": ranking_strategy,
                    "user_id": sid,
                    "text_mode": mode,
                    "avg_context_tokens": round(avg_ctx, 2),
                    "content_tokens_total": content_total,
                    "avg_context_ratio_over_content": round(avg_ctx / max(content_total, 1), 4),
                    "count": stat.get("count", 0),
                    "type": "conversation_context_usage_by_mode",
                }
            )

        os.makedirs(os.path.dirname(mx.Config.GEN_SUMMARY_FILE), exist_ok=True)
        with open(mx.Config.GEN_SUMMARY_FILE, "a", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def _generate_for_ranking(
        self,
        *,
        ranking_strategy: str,
        metric: str,
        trace_metric: str | None,
        mode: str,
        top_ids: List[int],
        sid: Any,
        qid: int,
        question: str,
        gold: Any,
        targets: List[int],
        category: Any,
        writer: csv.writer,
        out_jsonl: str,
        topn: int,
        graph_payload: Dict[str, Any] | None = None,
    ):
        mx = _mx()

        # Select appropriate box_map based on mode
        if mode == "content":
            # Membox style: Use only original conversation text (content_text)
            box_map = getattr(self, 'box_map_content', self.box_map)
            contexts = [box_map.get(sid, {}).get(bid) for bid in top_ids if box_map.get(sid, {}).get(bid)]
        elif mode == "event":
            # SA-Mem style: Use events with temporal metadata
            box_map = getattr(self, 'box_map_event', {})
            contexts = [box_map.get(sid, {}).get(bid) for bid in top_ids if box_map.get(sid, {}).get(bid)]
        else:
            # Trace modes (content_trace_event, trace_event)
            contexts = self._build_trace_contexts(sid, top_ids, trace_metric or "content_event_topic_kw", mode)

        if not contexts:
            return

        context_text = "\n\n".join(contexts)
        use_graph = bool(getattr(mx.Config, "USE_GRAPH_CONTEXT", False))
        allowed_categories = getattr(mx.Config, "GRAPH_CONTEXT_CATEGORIES", None)
        if use_graph and allowed_categories is not None:
            try:
                cat_int = int(category) if category is not None else None
            except Exception:
                cat_int = None
            use_graph = bool(cat_int is not None and cat_int in set(allowed_categories))

        graph_text = self._format_graph_context(graph_payload or {}, top_ids) if use_graph else ""
        if graph_text:
            context_text = f"{context_text}\n\nGraph Context:\n{graph_text}"
        user_prompt = mx.Config.PROMPT_QA_ANSWER.format(memories=context_text, question=question)
        note = f"S{sid}_QA_{qid}_{ranking_strategy}_{metric}_top{topn}_{trace_metric or 'content'}_{mode}"
        token_info = self._log_token_counts(context_text, question)

        ans = self.worker.chat_completion(
            user_prompt,
            note=note,
            extra={**token_info, "stage": f"{self.stage_label}:{ranking_strategy}"},
        )

        f1 = self._f1(ans, gold)
        bleu = self._bleu(ans, gold)
        ctx_tokens = int(token_info.get("memories_tokens", 0) or 0)

        writer.writerow(
            [
                sid,
                qid,
                ranking_strategy,
                question,
                gold,
                ans,
                f"{f1:.4f}",
                f"{bleu:.4f}",
                metric,
                trace_metric or "",
                mode,
                topn,
                top_ids,
                targets,
                category,
                ctx_tokens,
            ]
        )

        mx.TraceLogger.log(
            out_jsonl,
            {
                "user_id": sid,
                "qa_idx": qid,
                "ranking_strategy": ranking_strategy,
                "question": question,
                "gold": gold,
                "pred": ans,
                "f1": f1,
                "bleu": bleu,
                "metric": metric,
                "trace_metric": trace_metric,
                "text_mode": mode,
                "topn": topn,
                "topk": top_ids,
                "target_boxes": targets,
                "category": category,
                "context_tokens": ctx_tokens,
                "graph_context_used": bool(graph_text),
                "graph_context_categories": sorted(list(allowed_categories)) if allowed_categories is not None else None,
                "graph_payload_enabled": bool((graph_payload or {}).get("graph_enabled")),
            },
        )

        self._record_metrics(ranking_strategy, metric, trace_metric, mode, topn, f1, bleu, ctx_tokens, sid, category)

    def run(self, retrieval_jsonl: str, base_out_jsonl: str, base_out_csv: str):
        mx = _mx()
        if not os.path.exists(retrieval_jsonl):
            mx.logger.error("❌ Retrieval result not found: %s", retrieval_jsonl)
            return

        self._load_boxes()
        self._load_qa()
        self._load_traces()

        mx.logger.info("ℹ️ Generation text_modes=%s answer_topn=%s", self.text_modes, self.answer_topn_list)

        csv_base_exists = os.path.exists(base_out_csv)
        os.makedirs(os.path.dirname(base_out_csv), exist_ok=True)
        csv_base_file = open(base_out_csv, "a", newline="", encoding="utf-8")
        base_writer = csv.writer(csv_base_file)
        if not csv_base_exists:
            base_writer.writerow(
                [
                    "User_ID",
                    "QA_ID",
                    "Ranking_Strategy",
                    "Question",
                    "Gold",
                    "Pred",
                    "F1",
                    "BLEU",
                    "Metric",
                    "Trace_Metric",
                    "Text_Mode",
                    "TopN",
                    "TopIDs",
                    "Targets",
                    "Category",
                    "Context_Tokens",
                ]
            )

        with open(retrieval_jsonl, "r", encoding="utf-8") as f:
            entries = [json.loads(line) for line in f if line.strip()]

        for ent in entries:
            sid_raw = ent.get("user_id")
            qid_raw = ent.get("qa_idx")
            if sid_raw is None or qid_raw is None:
                continue
            sid = str(sid_raw)
            try:
                qid = int(qid_raw)
            except Exception:
                continue
            qa_list = self.qa_map.get(sid, [])
            if qid<0 or qid >= len(qa_list):
                continue
            qa = qa_list[qid]
            question = qa.get("question", "")
            gold = qa.get("answer", "")
            category = qa.get("category")
            if category == 5:
                continue

            targets = mx.evidence_to_targets(qa.get("evidence"), self.boxes_by_user.get(sid, []))

            rankings = ent.get("rankings", {}) or {}
            base_rank = rankings.get("content_event_topic_kw", []) or []
            if not base_rank:
                continue

            ranking_sets = [("baseline", base_rank, base_writer, base_out_jsonl)]
            graph_payload = ent.get("graph") or {}

            for ranking_strategy, ranking_list, writer_obj, out_path in ranking_sets:
                if not writer_obj or not out_path:
                    continue

                for topn in self.answer_topn_list:
                    top_ids = ranking_list[: topn]
                    if not top_ids:
                        continue

                    # Content mode: Membox style (only original conversation text)
                    if "content" in self.text_modes:
                        self._generate_for_ranking(
                            ranking_strategy=ranking_strategy,
                            metric="content_event_topic_kw",
                            trace_metric=None,
                            mode="content",
                            top_ids=top_ids,
                            sid=sid,
                            qid=qid,
                            question=question,
                            gold=gold,
                            targets=targets,
                            category=category,
                            writer=writer_obj,
                            out_jsonl=out_path,
                            topn=topn,
                            graph_payload=graph_payload,
                        )

                    # Event mode: SA-Mem style (events with temporal metadata)
                    if "event" in self.text_modes:
                        self._generate_for_ranking(
                            ranking_strategy=ranking_strategy,
                            metric="content_event_topic_kw",
                            trace_metric=None,
                            mode="event",
                            top_ids=top_ids,
                            sid=sid,
                            qid=qid,
                            question=question,
                            gold=gold,
                            targets=targets,
                            category=category,
                            writer=writer_obj,
                            out_jsonl=out_path,
                            topn=topn,
                            graph_payload=graph_payload,
                        )

                    if "content_trace_event" in self.text_modes or "trace_event" in self.text_modes:
                        for trace_metric in self.trace_metrics:
                            for mode in [m for m in self.text_modes if m in ("content_trace_event", "trace_event")]:
                                self._generate_for_ranking(
                                    ranking_strategy=ranking_strategy,
                                    metric="content_event_topic_kw",
                                    trace_metric=trace_metric,
                                    mode=mode,
                                    top_ids=top_ids,
                                    sid=sid,
                                    qid=qid,
                                    question=question,
                                    gold=gold,
                                    targets=targets,
                                    category=category,
                                    writer=writer_obj,
                                    out_jsonl=out_path,
                                    topn=topn,
                                    graph_payload=graph_payload,
                                )

        csv_base_file.close()
        self._write_summary()
        mx.logger.info("✅ Generation complete")