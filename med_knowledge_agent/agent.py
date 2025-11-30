# med_knowledge_agent/agent.py

from __future__ import annotations

import json
import logging
from typing import Any, AsyncGenerator, Dict, Optional
from datetime import datetime, timedelta
from pathlib import Path
import urllib.parse
import urllib.request
from xml.etree import ElementTree as ET

from google.adk.agents import BaseAgent, LlmAgent
from google.adk.events import Event
from google.adk.agents.invocation_context import InvocationContext

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Persistent Memory Store (JSON file)
# -----------------------------------------------------------------------------

MEMORY_DB_PATH = Path(__file__).parent / "memory_db.json"


def _load_memory_db() -> Dict[str, Any]:
    """Load persistent study memory from JSON file."""
    if not MEMORY_DB_PATH.exists():
        return {}
    try:
        with open(MEMORY_DB_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
    except Exception as e:
        logger.warning("Failed to load memory_db.json: %s", e)
    return {}


def _save_memory_db(db: Dict[str, Any]) -> None:
    """Save persistent study memory to JSON file."""
    try:
        with open(MEMORY_DB_PATH, "w", encoding="utf-8") as f:
            json.dump(db, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.warning("Failed to save memory_db.json: %s", e)


def _compute_next_review(difficulty: str) -> str:
    """Simple spaced review schedule based on difficulty."""
    difficulty = (difficulty or "medium").lower()
    if difficulty == "hard":
        days = 1
    elif difficulty == "easy":
        days = 4
    else:
        days = 2
    next_date = (datetime.utcnow() + timedelta(days=days)).date()
    return next_date.isoformat()


# -----------------------------------------------------------------------------
# External Tool: PubMed Search via E-utilities
# -----------------------------------------------------------------------------

NCBI_EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def pubmed_search(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Search PubMed for a query and fetch up to max_results articles.

    Returns:
        {
          "query": str,
          "results": [
            {
              "pmid": str,
              "title": str,
              "abstract": str,
              "authors": [str],
              "pubdate": str,
              "journal": str,
            },
            ...
          ]
        }
    """
    try:
        # 1) ESearch: get PMIDs
        esearch_params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "xml",
        }
        esearch_url = f"{NCBI_EUTILS_BASE}/esearch.fcgi?" + urllib.parse.urlencode(
            esearch_params
        )
        with urllib.request.urlopen(esearch_url, timeout=10) as resp:
            esearch_xml = resp.read().decode("utf-8")

        esearch_root = ET.fromstring(esearch_xml)
        id_list = [id_elem.text for id_elem in esearch_root.findall(".//Id")]

        if not id_list:
            return {"query": query, "results": []}

        # 2) EFetch: get article details
        efetch_params = {
            "db": "pubmed",
            "id": ",".join(id_list),
            "retmode": "xml",
        }
        efetch_url = f"{NCBI_EUTILS_BASE}/efetch.fcgi?" + urllib.parse.urlencode(
            efetch_params
        )
        with urllib.request.urlopen(efetch_url, timeout=10) as resp:
            efetch_xml = resp.read().decode("utf-8")

        efetch_root = ET.fromstring(efetch_xml)
        articles: list[Dict[str, Any]] = []

        for article in efetch_root.findall(".//PubmedArticle"):
            pmid = article.findtext(".//PMID") or ""
            title = article.findtext(".//ArticleTitle") or ""

            # Abstract may have multiple AbstractText nodes; join them.
            abstract_parts = []
            for at in article.findall(".//Abstract/AbstractText"):
                if at.text:
                    abstract_parts.append(at.text.strip())
            abstract = "\n".join(abstract_parts)

            journal = article.findtext(".//Journal/Title") or ""
            pub_year = article.findtext(".//PubDate/Year")
            medline_date = article.findtext(".//PubDate/MedlineDate")
            pubdate = pub_year or medline_date or ""

            authors: list[str] = []
            for a in article.findall(".//Author"):
                lastname = a.findtext("LastName") or ""
                firstname = a.findtext("ForeName") or ""
                full_name = f"{firstname} {lastname}".strip()
                if full_name:
                    authors.append(full_name)

            articles.append(
                {
                    "pmid": pmid,
                    "title": title,
                    "abstract": abstract,
                    "authors": authors,
                    "pubdate": pubdate,
                    "journal": journal,
                }
            )

        return {"query": query, "results": articles}
    except Exception as e:
        logger.warning("PubMed search failed for query %r: %s", query, e)
        return {"query": query, "results": [], "error": str(e)}


# -----------------------------------------------------------------------------
# Mock Tools (you can still use these or replace later)
# -----------------------------------------------------------------------------

def search_medical_literature(query: str, max_results: int = 3) -> Dict[str, Any]:
    """Simple mock literature search (kept for non-PubMed flows if desired)."""
    return {
        "query": query,
        "results": [
            f"Placeholder snippet {i+1} for '{query}'"
            for i in range(max_results)
        ],
        "note": "Replace with PubMed or guideline search API in production.",
    }


def save_study_plan(user_id: str, plan_markdown: str) -> Dict[str, Any]:
    """Mock persistence of a study plan."""
    logger.info(
        "[save_study_plan] user_id=%s, length=%d",
        user_id,
        len(plan_markdown),
    )
    return {
        "status": "saved",
        "user_id": user_id,
        "length": len(plan_markdown),
    }


# -----------------------------------------------------------------------------
# LLM Sub-Agents
# -----------------------------------------------------------------------------

MODEL = "gemini-2.0-flash"

# 1. Intent classifier --------------------------------------------------------

intent_classifier_agent = LlmAgent(
    model=MODEL,
    name="intent_classifier",
    description="Classify user intent for MedGuide.",
    instruction="""
Classify the user request into exactly ONE of:

- "study_concept"        # explain / teach a concept
- "create_quiz"          # user explicitly wants a quiz
- "make_study_plan"      # user wants a study plan
- "mixed_tutor"          # explanation + quiz and/or plan
- "clinical_question"    # real-world clinical reasoning question
- "study_history"        # user asks what they studied so far / review history
- "study_recommendation" # user asks what to study next / what is due
- "pubmed_search"        # user wants research papers, RCTs, trials, PubMed
- "chitchat"             # hi/hello/thanks/no/ok/etc.
- "other"                # anything else

Also extract "topic" (short string) or null.

Return ONLY JSON in this shape (no extra text, no backticks):

{
  "intent": "...",
  "topic": "..." or null
}
""",
    output_key="intent_json",
)

# 2. Literature search agent (simple / non-PubMed) ----------------------------

literature_search_agent = LlmAgent(
    model=MODEL,
    name="literature_searcher",
    description="Builds a search query and calls search_medical_literature (mock).",
    instruction="""
You are a research assistant for a MEDICAL LEARNING system.

Steps:
1. Read the user question and topic.
2. Build a concise, high-yield search query.
3. Call the `search_medical_literature` tool with that query.
4. Summarize the snippets in bullets.
5. End with: "Always verify with current primary literature and guidelines."

Do NOT invent citations or guideline numbers.
""",
    tools=[search_medical_literature],
    output_key="literature_results",
)

# 2b. PubMed agent (real evidence retrieval) ----------------------------------

pubmed_agent = LlmAgent(
    model=MODEL,
    name="pubmed_agent",
    description="Searches PubMed and summarizes key articles (educational use only).",
    instruction="""
You are a PubMed research assistant for a medical learner.

You have access to a tool `pubmed_search(query: str, max_results: int)` that
returns a list of articles with: pmid, title, abstract, authors, year, journal.

When the user asks for:
- papers
- RCTs
- trials
- evidence
- "PubMed" / "search PubMed"
you should:

1. Read the user question and derive a concise PubMed query.
2. Call `pubmed_search` with that query (max_results=5 is usually enough).
3. From the returned articles, create a structured summary:

   - Heading: "Key PubMed Articles"
   - For each article (3–5 max):
       - Year, Journal, PMID
       - 1–2 line summary of what was studied and the main outcome.
   - A short bullet list of overall "Evidence takeaways"

4. Finish with: "This is an educational summary of PubMed abstracts only and is
   NOT medical advice or a substitute for clinical judgment or full-text review."

Do NOT fabricate PMIDs or results; only summarize what you have from the tool.
""",
    tools=[pubmed_search],
    output_key="pubmed_summary",
)

# 3. Guideline explainer ------------------------------------------------------

guideline_explainer_agent = LlmAgent(
    model=MODEL,
    name="guideline_explainer",
    description="Gives a high-level guideline overview (educational only).",
    instruction="""
Provide a high-level overview of how major guidelines (e.g. ADA, ESC, ACC, KDIGO)
generally approach this topic.

Sections:
- Title
- Key points (bullets)
- Clinical caveats (bullets)
- Final line: "Educational only – not medical advice."
""",
    output_key="guideline_summary",
)

# 4. Concept explainer --------------------------------------------------------

concept_explainer_agent = LlmAgent(
    model=MODEL,
    name="concept_explainer",
    description="Explains mechanisms / pathophysiology clearly.",
    instruction="""
Explain the requested concept in this structure:

- Overview
- Mechanism / Pathophysiology
- Clinical relevance
- Exam pearls / pitfalls
- 3–5 take-home points

Be concise, structured, and educational.
Avoid giving patient-specific advice.
""",
    output_key="concept_explanation",
)

# 5. Quiz generator -----------------------------------------------------------

quiz_generator_agent = LlmAgent(
    model=MODEL,
    name="quiz_generator",
    description="Generates questions IF (and only if) user wants quiz.",
    instruction="""
If user intent is NOT "create_quiz" or "mixed_tutor", simply output: "OK."

Otherwise, generate:
- 5 short-answer questions
- 5 MCQs (A–E) with answer + 1-line explanation
- 10 flashcards ("Front: ... / Back: ...")

Use any available context in state (concept_explanation, guideline_summary, etc.).
""",
    output_key="quiz_pack",
)

# 6. Study plan builder -------------------------------------------------------

study_plan_builder_agent = LlmAgent(
    model=MODEL,
    name="study_plan_builder",
    description="Builds a short multi-day study plan.",
    instruction="""
ONLY run when intent is "make_study_plan" or "mixed_tutor".

Use:
- The topic from intent
- Any concept_explanation or quiz_pack that exists in state

Steps:
1. Build a realistic 3–7 day plan (30–60 min/day).
2. Each day: list topics, resources (e.g. "review today's explanation"), and tasks.
3. At the end, call save_study_plan("demo-user", plan_markdown).

Remember: This is just a suggested study plan; learners should adapt it to their
own schedule and curriculum.
""",
    tools=[save_study_plan],
    output_key="study_plan",
)

# 7. Smalltalk agent ----------------------------------------------------------

smalltalk_agent = LlmAgent(
    model=MODEL,
    name="smalltalk_agent",
    description="Handles greetings, 'no', 'thanks', etc.",
    instruction="""
You ONLY run when intent is "chitchat" or "other".

- If it's a greeting, reply with a short friendly hello and say what you can do
  (explain, quiz, study plans, PubMed summaries).
- If it's "no", "thanks", "ok", "bye", reply briefly and politely.
- Do NOT start teaching or generating quizzes/plans here.

Keep it 1–2 short sentences.
""",
    output_key="smalltalk_response",
)

# 8. Study history agent (uses persistent memory) -----------------------------

study_history_agent = LlmAgent(
    model=MODEL,
    name="study_history_agent",
    description="Summarizes what the user has studied (from memory_db.json).",
    instruction="""
You are a study history summarizer for a medical learning companion.

You receive a list of topics with:
- times studied
- last studied timestamps
- difficulty levels
- next review dates

Your tasks:
- Briefly summarize what they've studied so far.
- Reflect on coverage (e.g. systems: cardio, endocrine, renal).
- Mention which topics are due or overdue for review.
- Encourage them and suggest they continue with recommended areas.

Keep it short, structured and encouraging.
""",
    output_key="study_history_summary",
)

# 9. Study recommendation agent ----------------------------------------------

study_recommender_agent = LlmAgent(
    model=MODEL,
    name="study_recommender_agent",
    description="Recommends what to study next based on memory_db.json.",
    instruction="""
You are a medical study coach.

You receive:
- A list of topics with next_review dates and difficulty.
- A list of topics that are due or overdue today.

Your tasks:
1. List which topics are due/overdue and should be reviewed today.
2. Suggest 1–3 new logical topics to add (e.g. related mechanisms, same system).
3. Make it practical and not overwhelming.

Keep it to a few bullet points.
""",
    output_key="study_recommendations",
)

# -----------------------------------------------------------------------------
# ROOT ROUTER – Custom orchestration agent with persistent memory + PubMed
# -----------------------------------------------------------------------------

class MedGuideRouter(BaseAgent):
    """
    Root orchestrator:
    - Classifies intent.
    - Routes to:
      * smalltalk for greetings/other
      * explanation only for 'study_concept' / 'clinical_question'
      * quiz only when asked
      * study plan only when asked
      * full pipeline for 'mixed_tutor'
      * study_history for reviewing what has been studied (persistent JSON)
      * study_recommendation for recommending what to study next
      * pubmed_search for evidence / RCT / paper lists
    - Maintains a persistent JSON memory DB:
      memory_db.json with, per topic:
        - times_studied
        - last_studied
        - difficulty
        - next_review
        - notes (list)
    """

    intent_classifier: LlmAgent
    literature_searcher: LlmAgent
    pubmed_agent: LlmAgent
    guideline_explainer: LlmAgent
    concept_explainer: LlmAgent
    quiz_generator: LlmAgent
    study_plan_builder: LlmAgent
    smalltalk_agent: LlmAgent
    study_history_agent: LlmAgent
    study_recommender_agent: LlmAgent

    model_config = {"arbitrary_types_allowed": True}

    def __init__(
        self,
        name: str,
        intent_classifier: LlmAgent,
        literature_searcher: LlmAgent,
        pubmed_agent: LlmAgent,
        guideline_explainer: LlmAgent,
        concept_explainer: LlmAgent,
        quiz_generator: LlmAgent,
        study_plan_builder: LlmAgent,
        smalltalk_agent: LlmAgent,
        study_history_agent: LlmAgent,
        study_recommender_agent: LlmAgent,
    ) -> None:
        super().__init__(
            name=name,
            description="MedGuide Router – adaptive medical learning orchestrator.",
            sub_agents=[
                intent_classifier,
                literature_searcher,
                pubmed_agent,
                guideline_explainer,
                concept_explainer,
                quiz_generator,
                study_plan_builder,
                smalltalk_agent,
                study_history_agent,
                study_recommender_agent,
            ],
            intent_classifier=intent_classifier,
            literature_searcher=literature_searcher,
            pubmed_agent=pubmed_agent,
            guideline_explainer=guideline_explainer,
            concept_explainer=concept_explainer,
            quiz_generator=quiz_generator,
            study_plan_builder=study_plan_builder,
            smalltalk_agent=smalltalk_agent,
            study_history_agent=study_history_agent,
            study_recommender_agent=study_recommender_agent,
        )

    # 🔥 Helper: robustly extract JSON from model output -----------------------
    @staticmethod
    def _parse_intent_json(raw: Any) -> Dict[str, Any]:
        """
        Robustly extract JSON even if the LLM wraps it in ```json ... ``` fences.
        """
        intent = "other"
        topic: Optional[str] = None

        if not isinstance(raw, str):
            return {"intent": intent, "topic": topic}

        text = raw.strip()

        # Try to locate JSON object by { ... }
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {"intent": intent, "topic": topic}

        json_str = text[start : end + 1]
        try:
            parsed = json.loads(json_str)
            intent = parsed.get("intent", "other") or "other"
            topic = parsed.get("topic")
        except Exception as e:
            logger.warning("Failed to parse intent_json: %r (%s)", raw, e)

        return {"intent": intent, "topic": topic}

    # 🔁 Helper: update persistent memory for studied topics -------------------
    @staticmethod
    def _update_persistent_memory(topic: Optional[str], intent: str) -> None:
        """
        Update memory_db.json with topic studied, difficulty, etc.
        """
        if not topic:
            return

        norm_topic = str(topic).strip()
        if not norm_topic:
            return

        db = _load_memory_db()
        now = datetime.utcnow()
        now_str = now.strftime("%Y-%m-%d %H:%M")

        rec = db.get(norm_topic, {})

        times_studied = int(rec.get("times_studied", 0)) + 1

        # crude difficulty heuristic: clinical_question = hard, study_concept = medium, others easy
        if intent == "clinical_question":
            difficulty = "hard"
        elif intent == "study_concept":
            difficulty = rec.get("difficulty", "medium")
        else:
            difficulty = rec.get("difficulty", "easy")

        next_review = _compute_next_review(difficulty)

        notes = rec.get("notes", [])
        notes.append(f"studied via {intent} on {now_str}")

        db[norm_topic] = {
            "topic": norm_topic,
            "times_studied": times_studied,
            "last_studied": now_str,
            "difficulty": difficulty,
            "next_review": next_review,
            "notes": notes,
        }

        _save_memory_db(db)

    # Main ADK orchestration hook ---------------------------------------------
    async def _run_async_impl(
        self,
        ctx: InvocationContext,
    ) -> AsyncGenerator[Event, None]:

        # 1) Run intent classifier
        async for event in self.intent_classifier.run_async(ctx):
            yield event

        raw_intent = ctx.session.state.get("intent_json")
        parsed = self._parse_intent_json(raw_intent)
        intent = parsed["intent"]
        topic = parsed["topic"]

        logger.info("[MedGuideRouter] intent=%s topic=%s", intent, topic)

        # 2) Study history path: user asking what they've studied so far
        if intent == "study_history":
            db = _load_memory_db()
            if not db:
                history_prompt = (
                    "The user has no recorded study history yet in memory_db.json. "
                    "Explain that nothing has been stored yet and encourage them to "
                    "start by choosing a topic."
                )
            else:
                lines = []
                for t, rec in db.items():
                    times = rec.get("times_studied", 1)
                    last = rec.get("last_studied", "unknown time")
                    diff = rec.get("difficulty", "medium")
                    nr = rec.get("next_review", "unscheduled")
                    lines.append(
                        f"- {t}: studied {times} time(s), last at {last}, "
                        f"difficulty={diff}, next_review={nr}"
                    )
                history_str = "\n".join(lines)
                history_prompt = (
                    "Here is the user's persistent study history from memory_db.json:\n"
                    f"{history_str}\n\n"
                    "Summarize their progress, mention which topics are strong vs. early, "
                    "and encourage them."
                )

            new_ctx = ctx.copy(update={"user_input": history_prompt})
            async for event in self.study_history_agent.run_async(new_ctx):
                yield event
            return

        # 3) Study recommendation path: what to study next
        if intent == "study_recommendation":
            db = _load_memory_db()
            if not db:
                rec_prompt = (
                    "The user has no stored study history. "
                    "They are a medical learner. "
                    "Suggest 3–5 high-yield starter topics (with a sentence each) "
                    "for general medicine revision."
                )
            else:
                today = datetime.utcnow().date()
                due_lines = []
                all_lines = []
                for t, rec in db.items():
                    nr_str = rec.get("next_review")
                    diff = rec.get("difficulty", "medium")
                    times = rec.get("times_studied", 1)
                    try:
                        nr_date = datetime.strptime(nr_str, "%Y-%m-%d").date()
                    except Exception:
                        nr_date = None

                    all_lines.append(
                        f"- {t}: times_studied={times}, difficulty={diff}, next_review={nr_str}"
                    )

                    if nr_date is not None and nr_date <= today:
                        due_lines.append(
                            f"- {t}: due_for_review (next_review={nr_str}, difficulty={diff})"
                        )

                if not due_lines:
                    due_text = "No topics are strictly due or overdue today."
                else:
                    due_text = "Topics due/overdue today:\n" + "\n".join(due_lines)

                all_text = "\n".join(all_lines)

                rec_prompt = (
                    "Here is the user's current memory of studied topics:\n"
                    f"{all_text}\n\n"
                    f"{due_text}\n\n"
                    "Based on this, suggest:\n"
                    "1) Which topics they should review today (if any),\n"
                    "2) 1–3 new logical topics to study next.\n"
                )

            new_ctx = ctx.copy(update={"user_input": rec_prompt})
            async for event in self.study_recommender_agent.run_async(new_ctx):
                yield event
            return

        # 4) PubMed evidence path
        if intent == "pubmed_search":
            # Treat the whole user query as the research question; pubmed_agent
            # will derive the exact search string.
            async for event in self.pubmed_agent.run_async(ctx):
                yield event

            # Optionally log this as a "study" event in memory under a generic topic
            if topic:
                self._update_persistent_memory(topic, intent)
            else:
                self._update_persistent_memory("PubMed search", intent)
            return

        # 5) Smalltalk path: just greetings/thanks/etc.
        if intent in ("chitchat", "other"):
            async for event in self.smalltalk_agent.run_async(ctx):
                yield event
            return

        # 6) Shared info-gathering for all learning intents
        if intent in (
            "study_concept",
            "clinical_question",
            "create_quiz",
            "make_study_plan",
            "mixed_tutor",
        ):
            async for event in self.literature_searcher.run_async(ctx):
                yield event

            async for event in self.guideline_explainer.run_async(ctx):
                yield event

        # 7) Concept explanation (not needed for pure quiz)
        if intent in (
            "study_concept",
            "clinical_question",
            "make_study_plan",
            "mixed_tutor",
        ):
            async for event in self.concept_explainer.run_async(ctx):
                yield event

            # ✅ Update persistent memory after explanation
            self._update_persistent_memory(topic, intent)

        # 8) Quiz (only if user wants quiz or mixed tutor)
        if intent in ("create_quiz", "mixed_tutor"):
            async for event in self.quiz_generator.run_async(ctx):
                yield event

        # 9) Study plan (only if requested or mixed tutor)
        if intent in ("make_study_plan", "mixed_tutor"):
            async for event in self.study_plan_builder.run_async(ctx):
                yield event


# -----------------------------------------------------------------------------
# Root agent ADK will load with `adk run med_knowledge_agent`
# -----------------------------------------------------------------------------

root_agent = MedGuideRouter(
    name="medguide_router",
    intent_classifier=intent_classifier_agent,
    literature_searcher=literature_search_agent,
    pubmed_agent=pubmed_agent,
    guideline_explainer=guideline_explainer_agent,
    concept_explainer=concept_explainer_agent,
    quiz_generator=quiz_generator_agent,
    study_plan_builder=study_plan_builder_agent,
    smalltalk_agent=smalltalk_agent,
    study_history_agent=study_history_agent,
    study_recommender_agent=study_recommender_agent,
)
