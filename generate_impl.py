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
    @staticmethod
    def _infer_retrieval_source(retrieval_jsonl: str) -> str:
        name = os.path.splitext(os.path.basename(retrieval_jsonl))[0].lower()
        if name.startswith("retrieval_"):
            # retrieval_baseline / retrieval_enhanced / retrieval_full ...
            return name[len("retrieval_"):]
        if "baseline" in name:
            return "baseline"
        if "enhanced" in name:
            return "enhanced"
        return "unknown"
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
                sid = b.get("user_id")
                bid = mx._get_block_id(b)
                if sid is None:
                    continue

                # Build text from events since content_text doesn't exist
                events = b.get("events", [])
                event_descriptions = [e.get("description", "") for e in events if e.get("description")]
                topic_kw = b.get("features", {}).get("topic_kw_text", "")

                # Combine topic keywords and event descriptions
                text_parts = []
                if topic_kw:
                    text_parts.append(f"Topics: {topic_kw}")
                if event_descriptions:
                    text_parts.append("Events:\n" + "\n".join(f"- {desc}" for desc in event_descriptions))

                text = "\n\n".join(text_parts) if text_parts else ""

                self.box_map.setdefault(sid, {})[bid] = text
                self.boxes_by_user.setdefault(sid, []).append({"block_id": bid, "coverage": b.get("coverage", {})})
                self.content_totals[sid] += len(self.encoding.encode(text))

    def _load_qa(self):
        mx = _mx()
        if not os.path.exists(mx.Config.RAW_DATA_FILE):
            return
        # NOTE: current code assumes RAW_DATA_FILE is a JSON list. Keep behavior unchanged.
        with open(mx.Config.RAW_DATA_FILE, "r", encoding="utf-8") as f:
            raw_list = json.load(f)[: mx.Config.LIMIT_CONVERSATIONS]
        for data in raw_list:
            user_id = data.get("user_id")
            # For LoCoMo data: assign default user_id based on filename if missing
            if user_id is None:
                # Extract filename from RAW_DATA_FILE (e.g., "locomo10" from "locomo10.json")
                filename = os.path.basename(mx.Config.RAW_DATA_FILE)
                user_id = os.path.splitext(filename)[0]  # Remove .json extension

            if user_id is not None:
                self.qa_map[user_id] = data.get("qa", [])

    def _load_traces(self):
        mx = _mx()
        traces: Dict[Any, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
        if os.path.exists(mx.Config.TIME_TRACE_FILE):
            with open(mx.Config.TIME_TRACE_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    t = json.loads(line)
                    sid = t.get("user_id")
                    metric = t.get("metric")
                    if sid is None or metric is None:
                        continue
                    traces[sid][metric].append(t)
        self.trace_map = traces

    def _trace_events_for_box(self, sid: Any, bid: int, metric: str) -> List[str]:
        traces = self.trace_map.get(sid, {}).get(metric, [])
        for tr in traces:
            if bid not in tr.get("box_ids", []):
                continue
            events_texts: List[str] = []
            for entry in tr.get("entries", []):
                evs = entry.get("events") or []
                if not evs:
                    continue
                ts = str(entry.get("start_time", "Unknown"))
                for ev in evs:
                    ev_clean = str(ev).strip()
                    if ev_clean:
                        events_texts.append(f"{ts}: {ev_clean}")
            return events_texts
        return []

    def _build_trace_contexts(self, sid: Any, top_ids: List[int], trace_metric: str, mode: str) -> List[str]:
        contexts: List[str] = []
        seen_events = set()
        for bid in top_ids:
            events = self._trace_events_for_box(sid, bid, trace_metric)
            if not events:
                continue
            events_text = "\n".join(events)
            if events_text in seen_events:
                continue
            seen_events.add(events_text)
            if mode == "content_trace_event":
                content = self.box_map.get(sid, {}).get(bid)
                if not content:
                    continue
                contexts.append(f"{content}\nEvents:\n{events_text}")
            elif mode == "trace_event":
                contexts.append(f"Events:\n{events_text}")
        return contexts

    def _format_graph_context(self, graph_payload: Dict[str, Any], top_ids: List[int]) -> str:
        if not graph_payload or not graph_payload.get("graph_enabled"):
            return ""
        block_graph = graph_payload.get("block_graph") or {}
        parts: List[str] = []
        for bid in top_ids:
            block_data = block_graph.get(str(bid)) or {}
            if not block_data:
                continue
            block_parts: List[str] = [f"Block {bid}:"]
            events = block_data.get("events") or []
            if events:
                event_lines = []
                for ev in events:
                    desc = str(ev.get("event_description") or ev.get("event_id") or "").strip()
                    if desc:
                        event_lines.append(f"- {desc}")
                if event_lines:
                    block_parts.append("Events:")
                    block_parts.extend(event_lines)
            similar_edges = block_data.get("similar_events") or []
            if similar_edges:
                edge_lines = []
                for edge in similar_edges:
                    fe = str(edge.get("from_event_id") or "").strip()
                    te = str(edge.get("to_event_id") or "").strip()
                    score = edge.get("score")
                    if fe and te:
                        edge_lines.append(f"- {fe} -> {te} (score={score})")
                if edge_lines:
                    block_parts.append("Similar events:")
                    block_parts.extend(edge_lines)
            relations = block_data.get("relations") or []
            if relations:
                rel_lines = []
                for rel in relations:
                    src = str(rel.get("source") or "").strip()
                    rel_type = str(rel.get("relationship") or "").strip()
                    dst = str(rel.get("destination") or "").strip()
                    if src and rel_type and dst:
                        rel_lines.append(f"- {src} -- {rel_type} -- {dst}")
                if rel_lines:
                    block_parts.append("Relations:")
                    block_parts.extend(rel_lines)
            parts.append("\n".join(block_parts))
        return "\n\n".join(parts)

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

        if mode == "content":
            contexts = [self.box_map.get(sid, {}).get(bid) for bid in top_ids if self.box_map.get(sid, {}).get(bid)]
        else:
            contexts = self._build_trace_contexts(sid, top_ids, trace_metric or "content_event_topic_kw", mode)
        if not contexts:
            return

        context_text = "\n\n".join(contexts)
        graph_text = self._format_graph_context(graph_payload or {}, top_ids)
        if graph_text:
            context_text = f"{context_text}\n\nGraph Context:\n{graph_text}"
        user_prompt = mx.Config.PROMPT_QA_ANSWER.format(memories=context_text, question=question)
        note = f"S{sid}_QA_{qid}_{ranking_strategy}_{metric}_top{topn}_{trace_metric or 'content'}_{mode}"
        token_info = self._log_token_counts(context_text, question)

        ans = self.worker.chat_completion(
            user_prompt,
            note=note,
            extra={
                **token_info,
                "stage": f"{self.stage_label}:{ranking_strategy}",
                "retrieval_source": ranking_strategy,
            },
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
            },
        )

        self._record_metrics(ranking_strategy, metric, trace_metric, mode, topn, f1, bleu, ctx_tokens, sid, category)

    def run(self, retrieval_jsonl: str, base_out_jsonl: str, base_out_csv: str, retrieval_source: str = "auto"):
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
        source = (retrieval_source or "auto").strip().lower()
        if source == "auto":
            source = self._infer_retrieval_source(retrieval_jsonl)
        for ent in entries:
            sid = ent.get("user_id")
            qid = ent.get("qa_idx")
            if sid is None or qid is None:
                continue

            qa_list = self.qa_map.get(sid, [])
            if qid >= len(qa_list):
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

            graph_payload = ent.get("graph") or {}

            ranking_sets = [(source, base_rank, base_writer, base_out_jsonl)]

            for ranking_strategy, ranking_list, writer_obj, out_path in ranking_sets:
                if not writer_obj or not out_path:
                    continue

                for topn in self.answer_topn_list:
                    top_ids = ranking_list[: topn]
                    if not top_ids:
                        continue

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