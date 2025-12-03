"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                MEDGUIDE AI                                   ║
║                    AI-Powered Medical Learning Companion                     ║
║                                                                              ║
║  A production-grade multi-agent system with:                                 ║
║  • Explicit mode definitions & state machine                                 ║
║  • Formal input/output contracts                                             ║
║  • Safety boundaries & refusal patterns                                      ║
║  • Structured observability & tracing                                        ║
║  • Adaptive learning with spaced repetition                                  ║
║  • Multi-agent coordination protocols                                        ║
║                                                                              ║
║  Author: Ashhad                                                              ║
║  Competition: Google ADK Agents Intensive - Agents for Good                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
import uuid
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Callable, TypeVar, Generic
from xml.etree import ElementTree as ET

from google.adk.agents import BaseAgent, LlmAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai import types


# =============================================================================
# SECTION 1: CONFIGURATION & CONSTANTS
# =============================================================================

class Config:
    """Centralized configuration with explicit defaults."""
    
    # Model
    MODEL: str = "gemini-2.0-flash"
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent
    MEMORY_PATH: Path = BASE_DIR / "learner_memory.json"
    LOG_PATH: Path = BASE_DIR / "agent_logs.jsonl"
    
    # API Settings
    PUBMED_BASE_URL: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    API_TIMEOUT: int = 15
    MAX_SEARCH_RESULTS: int = 10
    
    # Learning System
    SPACED_REPETITION_INTERVALS: Dict[str, int] = {
        "new": 1,
        "learning": 3,
        "review": 7,
        "mastered": 14
    }
    
    # Safety
    MAX_INPUT_LENGTH: int = 5000
    MAX_RESPONSE_LENGTH: int = 4000


# =============================================================================
# SECTION 2: TYPE SYSTEM & CONTRACTS
# =============================================================================

class AgentMode(Enum):
    """
    Explicit agent operation modes.
    Each mode has defined input expectations and output contracts.
    """
    IDLE = auto()           # Waiting for input
    EXPLAIN = auto()        # Medical concept explanation
    QUIZ = auto()           # Assessment generation
    SEARCH = auto()         # Literature retrieval
    PLAN = auto()           # Study planning
    HISTORY = auto()        # Progress review
    CLARIFY = auto()        # Seeking clarification
    REFUSE = auto()         # Safety refusal


class TopicDifficulty(Enum):
    """Learner difficulty levels for adaptive learning."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class SafetyFlag(Enum):
    """Safety classification for inputs."""
    SAFE = auto()
    NEEDS_DISCLAIMER = auto()
    REFUSE_DIAGNOSIS = auto()
    REFUSE_TREATMENT = auto()
    REFUSE_EMERGENCY = auto()
    AMBIGUOUS = auto()


@dataclass
class AgentInput:
    """
    Formal input contract for the agent.
    All inputs must be validated against this contract.
    """
    text: str
    session_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    context: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # Input validation
        if len(self.text) > Config.MAX_INPUT_LENGTH:
            raise ValueError(f"Input exceeds maximum length of {Config.MAX_INPUT_LENGTH}")
        self.text = self.text.strip()
    
    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context
        }


@dataclass
class AgentOutput:
    """
    Formal output contract for the agent.
    All responses must conform to this contract.
    """
    content: str
    mode: AgentMode
    confidence: float
    topic: Optional[str] = None
    sources: List[Dict] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    safety_applied: bool = False
    
    def to_dict(self) -> Dict:
        return {
            "content": self.content[:Config.MAX_RESPONSE_LENGTH],
            "mode": self.mode.name,
            "confidence": self.confidence,
            "topic": self.topic,
            "sources": self.sources,
            "metadata": self.metadata,
            "safety_applied": self.safety_applied
        }


@dataclass
class ToolInvocation:
    """Contract for tool invocations with explicit pre/post conditions."""
    tool_name: str
    parameters: Dict[str, Any]
    preconditions: List[str]
    expected_output_type: str
    timeout: int = Config.API_TIMEOUT
    
    def validate_preconditions(self, context: Dict) -> bool:
        """Verify all preconditions are met before invocation."""
        return all(context.get(p) for p in self.preconditions)


@dataclass  
class LearnerState:
    """
    Learner profile with explicit state management.
    Follows data minimization principles.
    """
    user_id: str
    topics_studied: Dict[str, Dict] = field(default_factory=dict)
    current_difficulty: TopicDifficulty = TopicDifficulty.BEGINNER
    session_count: int = 0
    last_active: Optional[str] = None
    
    # Explicit: What we DON'T store
    # - No personal health information
    # - No diagnostic history
    # - No treatment preferences
    
    def to_dict(self) -> Dict:
        return {
            "user_id": self.user_id,
            "topics_studied": self.topics_studied,
            "current_difficulty": self.current_difficulty.value,
            "session_count": self.session_count,
            "last_active": self.last_active
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "LearnerState":
        return cls(
            user_id=data.get("user_id", "default"),
            topics_studied=data.get("topics_studied", {}),
            current_difficulty=TopicDifficulty(data.get("current_difficulty", "beginner")),
            session_count=data.get("session_count", 0),
            last_active=data.get("last_active")
        )


# =============================================================================
# SECTION 3: OBSERVABILITY & LOGGING
# =============================================================================

@dataclass
class LogEntry:
    """Structured log entry for observability."""
    timestamp: str
    correlation_id: str
    event_type: str
    component: str
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    duration_ms: Optional[float] = None
    error: Optional[str] = None


class ObservabilityLayer:
    """
    Structured logging and metrics for agent behavior tracing.
    Enables auditing and debugging of agent decisions.
    """
    
    def __init__(self, log_path: Path = Config.LOG_PATH):
        self.log_path = log_path
        self._correlation_id: Optional[str] = None
        self._start_time: Optional[float] = None
        self._metrics: Dict[str, List[float]] = {}
    
    def start_trace(self) -> str:
        """Begin a new request trace."""
        self._correlation_id = str(uuid.uuid4())[:8]
        self._start_time = time.perf_counter()
        return self._correlation_id
    
    def log(
        self,
        event_type: str,
        component: str,
        message: str,
        data: Optional[Dict] = None,
        error: Optional[str] = None
    ):
        """Write structured log entry."""
        duration = None
        if self._start_time:
            duration = (time.perf_counter() - self._start_time) * 1000
        
        entry = LogEntry(
            timestamp=datetime.utcnow().isoformat(),
            correlation_id=self._correlation_id or "no-trace",
            event_type=event_type,
            component=component,
            message=message,
            data=data or {},
            duration_ms=duration,
            error=error
        )
        
        # Console output
        level = "ERROR" if error else "INFO"
        print(f"{entry.timestamp} | {level:5} | {entry.correlation_id} | [{component}] {message}")
        
        # Persist to file
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(asdict(entry)) + "\n")
        except IOError:
            pass
    
    def log_mode_transition(self, from_mode: AgentMode, to_mode: AgentMode, reason: str):
        """Log agent mode transitions for auditability."""
        self.log(
            event_type="MODE_TRANSITION",
            component="StateMachine",
            message=f"{from_mode.name} → {to_mode.name}",
            data={"reason": reason}
        )
    
    def log_tool_invocation(self, tool: str, params: Dict, success: bool, result_count: int = 0):
        """Log tool invocations with results."""
        self.log(
            event_type="TOOL_CALL",
            component="ToolLayer",
            message=f"Tool: {tool}",
            data={"params": params, "success": success, "result_count": result_count}
        )
    
    def log_safety_check(self, flag: SafetyFlag, input_text: str):
        """Log safety evaluations."""
        self.log(
            event_type="SAFETY_CHECK",
            component="SafetyLayer",
            message=f"Flag: {flag.name}",
            data={"input_preview": input_text[:100]}
        )
    
    def record_metric(self, name: str, value: float):
        """Record a metric for analysis."""
        if name not in self._metrics:
            self._metrics[name] = []
        self._metrics[name].append(value)


# Global observability instance
obs = ObservabilityLayer()


# =============================================================================
# SECTION 4: SAFETY LAYER
# =============================================================================

class SafetyBoundary:
    """
    Explicit safety boundaries for medical AI.
    
    HARD BOUNDARIES (Always refuse):
    - Diagnosis requests
    - Treatment recommendations
    - Emergency medical advice
    - Drug dosing
    - Personal health decisions
    
    SOFT BOUNDARIES (Add disclaimer):
    - Symptom discussions
    - Drug information
    - Procedure descriptions
    """
    
    # Patterns that trigger HARD refusal
    REFUSE_PATTERNS = [
        # Diagnosis requests
        r"\b(do i have|diagnose|what('s| is) wrong with me|is it|could it be)\b.*\b(cancer|disease|condition|infection|disorder)\b",
        r"\bdiagnos(e|is)\b",
        r"\bwhat('s| is) (wrong|causing)\b.*\b(my|me)\b",
        
        # Treatment requests
        r"\b(should i take|what (medication|drug|medicine)|prescribe|how much .* should i)\b",
        r"\btreat(ment)? for my\b",
        r"\b(dosage|dose) for\b",
        
        # Emergency indicators
        r"\b(chest pain|can't breathe|overdose|suicide|heart attack|stroke)\b.*\b(right now|currently|having)\b",
        r"\bemergency\b",
        
        # Personal health decisions
        r"\bshould i (stop|start|continue|change)\b.*\b(medication|treatment|therapy)\b",
    ]
    
    # Patterns requiring disclaimer
    DISCLAIMER_PATTERNS = [
        r"\bsymptoms? of\b",
        r"\bside effects?\b",
        r"\b(drug|medication) interaction\b",
        r"\bprognosis\b",
    ]
    
    DISCLAIMER_TEXT = (
        "\n\n---\n"
        "⚕️ *Educational information only. Not medical advice. "
        "Consult a healthcare provider for personal health decisions.*"
    )
    
    REFUSAL_RESPONSES = {
        SafetyFlag.REFUSE_DIAGNOSIS: (
            "I can't provide diagnoses. I'm designed for medical education, not clinical assessment. "
            "For health concerns, please consult a healthcare provider who can properly evaluate you."
        ),
        SafetyFlag.REFUSE_TREATMENT: (
            "I can't recommend treatments or medications. Treatment decisions require "
            "a healthcare provider who knows your complete medical history. "
            "I can explain how treatments work educationally if that helps."
        ),
        SafetyFlag.REFUSE_EMERGENCY: (
            "This sounds like it could be a medical emergency. Please contact emergency services "
            "(911 in the US) or go to your nearest emergency room immediately. "
            "I'm not equipped to handle urgent medical situations."
        ),
    }
    
    @classmethod
    def evaluate(cls, text: str) -> SafetyFlag:
        """Evaluate input against safety boundaries."""
        text_lower = text.lower()
        
        # Check hard boundaries
        for pattern in cls.REFUSE_PATTERNS:
            if re.search(pattern, text_lower):
                if "emergency" in text_lower or any(w in text_lower for w in ["chest pain", "can't breathe", "overdose"]):
                    return SafetyFlag.REFUSE_EMERGENCY
                if any(w in text_lower for w in ["diagnose", "do i have", "what's wrong"]):
                    return SafetyFlag.REFUSE_DIAGNOSIS
                if any(w in text_lower for w in ["should i take", "prescribe", "dosage"]):
                    return SafetyFlag.REFUSE_TREATMENT
        
        # Check soft boundaries
        for pattern in cls.DISCLAIMER_PATTERNS:
            if re.search(pattern, text_lower):
                return SafetyFlag.NEEDS_DISCLAIMER
        
        return SafetyFlag.SAFE
    
    @classmethod
    def get_refusal_response(cls, flag: SafetyFlag) -> Optional[str]:
        """Get appropriate refusal response for a safety flag."""
        return cls.REFUSAL_RESPONSES.get(flag)
    
    @classmethod
    def apply_disclaimer(cls, response: str) -> str:
        """Append safety disclaimer to response."""
        return response + cls.DISCLAIMER_TEXT


# =============================================================================
# SECTION 5: INTENT CLASSIFICATION (Deterministic + LLM)
# =============================================================================

class IntentClassifier:
    """
    Hybrid intent classification with deterministic rules + LLM fallback.
    Provides predictable behavior for common patterns.
    """
    
    # Deterministic rules (checked first, in order)
    RULES = [
        # Greetings - highest priority
        (AgentMode.IDLE, r"^(hi|hello|hey|good morning|good afternoon|good evening)[\s!.,]*$"),
        (AgentMode.IDLE, r"^(thanks|thank you|thx|bye|goodbye|see you)[\s!.,]*$"),
        
        # History/Progress
        (AgentMode.HISTORY, r"\b(my progress|what have i (studied|learned)|show (my )?history|study history)\b"),
        (AgentMode.HISTORY, r"\b(recommend|what should i study|what next|suggest)\b"),
        
        # PubMed Search - explicit research requests
        (AgentMode.SEARCH, r"\b(find|search|get|show|give)\b.*\b(paper|papers|article|articles|research|studies|literature|pubmed)\b"),
        (AgentMode.SEARCH, r"\b(pubmed|research|literature)\b.*\b(on|about|for)\b"),
        
        # Quiz - explicit quiz requests
        (AgentMode.QUIZ, r"\b(quiz|test)\s+(me|my knowledge)\b"),
        (AgentMode.QUIZ, r"\b(practice questions|flashcards)\b.*\b(on|about|for)\b"),
        
        # Study Plan - explicit planning requests
        (AgentMode.PLAN, r"\b(study plan|learning plan|study schedule)\b"),
        (AgentMode.PLAN, r"\b(create|make|build)\b.*\b(plan|schedule)\b.*\b(for|to study)\b"),
    ]
    
    @classmethod
    def classify(cls, text: str) -> tuple[AgentMode, Optional[str], float]:
        """
        Classify intent with confidence score.
        
        Returns:
            (mode, topic, confidence)
        """
        text_lower = text.lower().strip()
        
        # Try deterministic rules first
        for mode, pattern in cls.RULES:
            if re.search(pattern, text_lower):
                topic = cls._extract_topic(text_lower, mode)
                obs.log("INTENT_CLASSIFIED", "IntentClassifier", 
                       f"Deterministic: {mode.name}", {"pattern": pattern[:50]})
                return (mode, topic, 0.95)
        
        # Default to EXPLAIN for medical questions
        # (LLM will handle the actual response)
        topic = cls._extract_topic(text_lower, AgentMode.EXPLAIN)
        return (AgentMode.EXPLAIN, topic, 0.80)
    
    @classmethod
    def _extract_topic(cls, text: str, mode: AgentMode) -> Optional[str]:
        """Extract the medical topic from input."""
        # Remove common prefixes
        prefixes = [
            r"^(what is|what are|what causes|explain|tell me about|how does|why do|describe)\s+",
            r"^(find|search|get|show|give)\s+(me\s+)?(papers?|articles?|research|studies)\s+(on|about|for)\s+",
            r"^(quiz|test)\s+(me\s+)?(on|about)\s+",
            r"^(create|make|build)\s+(a\s+)?(study\s+)?plan\s+(for|on|about)\s+",
        ]
        
        topic = text
        for prefix in prefixes:
            topic = re.sub(prefix, "", topic, flags=re.IGNORECASE)
        
        # Clean up
        topic = topic.strip().rstrip("?.!")
        
        return topic if topic and len(topic) > 2 else None


# =============================================================================
# SECTION 6: MEMORY SYSTEM (Adaptive Learning)
# =============================================================================

class MemoryManager:
    """
    Adaptive learning memory with spaced repetition.
    
    State Management Rules:
    - Store only learning progress, not personal data
    - Clear session data after inactivity
    - Never persist health-related queries
    """
    
    def __init__(self, path: Path = Config.MEMORY_PATH):
        self.path = path
        self._cache: Optional[LearnerState] = None
        self._ensure_storage()
    
    def _ensure_storage(self):
        if not self.path.exists():
            self.path.write_text("{}")
    
    def _load(self) -> Dict:
        try:
            return json.loads(self.path.read_text())
        except (json.JSONDecodeError, IOError):
            return {}
    
    def _save(self, data: Dict):
        self.path.write_text(json.dumps(data, indent=2))
    
    def get_learner(self, user_id: str = "default") -> LearnerState:
        """Get or create learner state."""
        data = self._load()
        if user_id in data:
            return LearnerState.from_dict(data[user_id])
        return LearnerState(user_id=user_id)
    
    def record_study(
        self,
        topic: str,
        activity: str,
        performance: Optional[float] = None,
        user_id: str = "default"
    ):
        """
        Record a study session with spaced repetition update.
        
        Args:
            topic: Medical topic studied
            activity: Type of activity (explain, quiz, search)
            performance: Optional score (0-1) for adaptive difficulty
            user_id: Learner identifier
        """
        if not topic:
            return
        
        data = self._load()
        learner = self.get_learner(user_id)
        
        topic_key = topic.lower().strip()
        now = datetime.utcnow()
        
        if topic_key not in learner.topics_studied:
            learner.topics_studied[topic_key] = {
                "first_seen": now.isoformat(),
                "study_count": 0,
                "last_studied": None,
                "next_review": None,
                "retention_stage": "new",
                "activities": []
            }
        
        topic_data = learner.topics_studied[topic_key]
        topic_data["study_count"] += 1
        topic_data["last_studied"] = now.isoformat()
        topic_data["activities"].append({
            "type": activity,
            "timestamp": now.isoformat(),
            "performance": performance
        })
        
        # Update spaced repetition schedule
        topic_data = self._update_retention(topic_data, performance)
        
        # Update difficulty based on performance history
        if performance is not None:
            learner.current_difficulty = self._adjust_difficulty(learner, performance)
        
        learner.session_count += 1
        learner.last_active = now.isoformat()
        
        data[user_id] = learner.to_dict()
        self._save(data)
        
        obs.log("MEMORY_UPDATE", "MemoryManager",
               f"Recorded: {topic_key}", {"activity": activity, "stage": topic_data["retention_stage"]})
    
    def _update_retention(self, topic_data: Dict, performance: Optional[float]) -> Dict:
        """Update retention stage based on spaced repetition algorithm."""
        current_stage = topic_data.get("retention_stage", "new")
        
        # Stage progression
        stages = ["new", "learning", "review", "mastered"]
        current_idx = stages.index(current_stage)
        
        if performance is not None:
            if performance >= 0.8 and current_idx < len(stages) - 1:
                current_stage = stages[current_idx + 1]
            elif performance < 0.5 and current_idx > 0:
                current_stage = stages[current_idx - 1]
        else:
            # Default progression for non-quiz activities
            if current_idx < 2:
                current_stage = stages[min(current_idx + 1, 2)]
        
        topic_data["retention_stage"] = current_stage
        
        # Calculate next review date
        interval_days = Config.SPACED_REPETITION_INTERVALS.get(current_stage, 3)
        next_review = datetime.utcnow() + timedelta(days=interval_days)
        topic_data["next_review"] = next_review.date().isoformat()
        
        return topic_data
    
    def _adjust_difficulty(self, learner: LearnerState, recent_performance: float) -> TopicDifficulty:
        """Adjust difficulty based on cumulative performance."""
        # Get average performance from recent activities
        all_perfs = []
        for topic_data in learner.topics_studied.values():
            for activity in topic_data.get("activities", [])[-10:]:
                if activity.get("performance") is not None:
                    all_perfs.append(activity["performance"])
        
        if len(all_perfs) < 3:
            return learner.current_difficulty
        
        avg_perf = sum(all_perfs) / len(all_perfs)
        
        if avg_perf >= 0.8 and learner.current_difficulty != TopicDifficulty.ADVANCED:
            return TopicDifficulty(["beginner", "intermediate", "advanced"][
                min(["beginner", "intermediate", "advanced"].index(learner.current_difficulty.value) + 1, 2)
            ])
        elif avg_perf < 0.5 and learner.current_difficulty != TopicDifficulty.BEGINNER:
            return TopicDifficulty(["beginner", "intermediate", "advanced"][
                max(["beginner", "intermediate", "advanced"].index(learner.current_difficulty.value) - 1, 0)
            ])
        
        return learner.current_difficulty
    
    def get_due_topics(self, user_id: str = "default") -> List[str]:
        """Get topics due for spaced repetition review."""
        learner = self.get_learner(user_id)
        today = datetime.utcnow().date().isoformat()
        
        due = []
        for topic, data in learner.topics_studied.items():
            next_review = data.get("next_review")
            if next_review and next_review <= today:
                due.append(topic)
        
        return due
    
    def get_progress_summary(self, user_id: str = "default") -> str:
        """Generate human-readable progress summary."""
        learner = self.get_learner(user_id)
        
        if not learner.topics_studied:
            return "You haven't studied any topics yet. Ask me about any medical concept to get started!"
        
        total_topics = len(learner.topics_studied)
        total_sessions = sum(t.get("study_count", 0) for t in learner.topics_studied.values())
        due_topics = self.get_due_topics(user_id)
        
        # Get topics by retention stage
        stages = {"new": [], "learning": [], "review": [], "mastered": []}
        for topic, data in learner.topics_studied.items():
            stage = data.get("retention_stage", "new")
            stages[stage].append(topic)
        
        lines = [
            f"**Topics Studied:** {total_topics}",
            f"**Total Sessions:** {total_sessions}",
            f"**Current Level:** {learner.current_difficulty.value.title()}",
            ""
        ]
        
        if stages["mastered"]:
            lines.append(f"✅ **Mastered:** {', '.join(stages['mastered'][:3])}")
        if stages["review"]:
            lines.append(f"📚 **In Review:** {', '.join(stages['review'][:3])}")
        if stages["learning"]:
            lines.append(f"📖 **Learning:** {', '.join(stages['learning'][:3])}")
        
        if due_topics:
            lines.append(f"\n📅 **Due for Review:** {', '.join(due_topics[:5])}")
        
        return "\n".join(lines)
    
    def get_recommendations(self, user_id: str = "default") -> str:
        """Generate personalized study recommendations."""
        learner = self.get_learner(user_id)
        due = self.get_due_topics(user_id)
        
        lines = []
        
        if due:
            lines.append(f"**Review these topics:** {', '.join(due[:3])}")
        
        # Find weak areas (topics with low performance)
        weak = []
        for topic, data in learner.topics_studied.items():
            perfs = [a.get("performance", 1) for a in data.get("activities", []) if a.get("performance")]
            if perfs and sum(perfs)/len(perfs) < 0.6:
                weak.append(topic)
        
        if weak:
            lines.append(f"**Need more practice:** {', '.join(weak[:3])}")
        
        # Suggest progression based on level
        suggestions = {
            TopicDifficulty.BEGINNER: ["anatomy basics", "vital signs", "common symptoms", "basic pharmacology"],
            TopicDifficulty.INTERMEDIATE: ["pathophysiology", "clinical reasoning", "differential diagnosis basics"],
            TopicDifficulty.ADVANCED: ["complex cases", "treatment protocols", "research interpretation"]
        }
        
        lines.append(f"**Try next:** {', '.join(suggestions[learner.current_difficulty][:2])}")
        
        return "\n".join(lines) if lines else "Keep exploring! Ask about any medical topic."


# Global memory instance
memory = MemoryManager()


# =============================================================================
# SECTION 7: TOOLS (Structured Invocation)
# =============================================================================

class ToolRegistry:
    """
    Structured tool management with explicit invocation contracts.
    """
    
    @staticmethod
    def search_pubmed(query: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Search PubMed for medical literature.
        
        PRECONDITIONS:
        - Query must be non-empty medical term
        - Max results between 1-10
        
        POSTCONDITIONS:
        - Returns dict with query, count, articles[]
        - Each article has: pmid, title, abstract, authors, journal, year, url
        - On error: returns dict with error field
        """
        # Validate preconditions
        if not query or len(query.strip()) < 2:
            return {"query": query, "count": 0, "articles": [], "error": "Invalid query"}
        
        max_results = max(1, min(max_results, Config.MAX_SEARCH_RESULTS))
        
        obs.log("TOOL_START", "PubMed", f"Searching: {query[:50]}")
        start_time = time.perf_counter()
        
        try:
            # Step 1: Search for PMIDs
            search_params = urllib.parse.urlencode({
                "db": "pubmed",
                "term": query,
                "retmax": max_results,
                "retmode": "xml",
                "sort": "relevance"
            })
            search_url = f"{Config.PUBMED_BASE_URL}/esearch.fcgi?{search_params}"
            
            with urllib.request.urlopen(search_url, timeout=Config.API_TIMEOUT) as resp:
                search_xml = resp.read().decode()
            
            root = ET.fromstring(search_xml)
            pmids = [e.text for e in root.findall(".//Id") if e.text]
            
            if not pmids:
                obs.log_tool_invocation("pubmed_search", {"query": query}, True, 0)
                return {"query": query, "count": 0, "articles": []}
            
            # Step 2: Fetch article details
            fetch_params = urllib.parse.urlencode({
                "db": "pubmed",
                "id": ",".join(pmids),
                "retmode": "xml"
            })
            fetch_url = f"{Config.PUBMED_BASE_URL}/efetch.fcgi?{fetch_params}"
            
            with urllib.request.urlopen(fetch_url, timeout=Config.API_TIMEOUT) as resp:
                fetch_xml = resp.read().decode()
            
            root = ET.fromstring(fetch_xml)
            articles = []
            
            for article in root.findall(".//PubmedArticle"):
                pmid = article.findtext(".//PMID") or ""
                title = article.findtext(".//ArticleTitle") or "Untitled"
                
                # Extract abstract (handle structured abstracts)
                abstract_parts = []
                for ab in article.findall(".//AbstractText"):
                    label = ab.get("Label", "")
                    text = ab.text or ""
                    if label:
                        abstract_parts.append(f"{label}: {text}")
                    else:
                        abstract_parts.append(text)
                abstract = " ".join(abstract_parts)[:800] or "No abstract available"
                
                # Extract authors (first 3)
                authors = []
                for auth in article.findall(".//Author")[:3]:
                    last = auth.findtext("LastName") or ""
                    if last:
                        authors.append(last)
                
                journal = article.findtext(".//Journal/Title") or ""
                year = article.findtext(".//PubDate/Year") or article.findtext(".//PubDate/MedlineDate") or ""
                
                articles.append({
                    "pmid": pmid,
                    "title": title,
                    "abstract": abstract,
                    "authors": authors,
                    "journal": journal,
                    "year": year[:4] if year else "",
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                })
            
            duration = (time.perf_counter() - start_time) * 1000
            obs.log_tool_invocation("pubmed_search", {"query": query}, True, len(articles))
            obs.record_metric("pubmed_search_ms", duration)
            
            return {"query": query, "count": len(articles), "articles": articles}
            
        except Exception as e:
            obs.log_tool_invocation("pubmed_search", {"query": query}, False)
            obs.log("TOOL_ERROR", "PubMed", str(e), error=str(e))
            return {"query": query, "count": 0, "articles": [], "error": str(e)}
    
    @staticmethod
    def save_study_plan(content: str, topic: str = "Medical") -> Dict[str, Any]:
        """Save generated study plan to file."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"study_plan_{timestamp}.md"
        filepath = Config.BASE_DIR / filename
        
        try:
            header = f"# Study Plan: {topic}\n*Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}*\n\n"
            filepath.write_text(header + content)
            obs.log_tool_invocation("save_plan", {"topic": topic}, True)
            return {"saved": True, "filename": filename, "path": str(filepath)}
        except IOError as e:
            obs.log_tool_invocation("save_plan", {"topic": topic}, False)
            return {"saved": False, "error": str(e)}


# =============================================================================
# SECTION 8: AGENT PROMPTS (Mode-Specific)
# =============================================================================

class AgentPrompts:
    """
    Mode-specific prompts with explicit behavioral contracts.
    """
    
    SYSTEM = """You are MedGuide, an AI medical education tutor.

CORE IDENTITY:
- Expert medical educator, not a clinician
- Warm, clear, encouraging teaching style
- Evidence-based, accurate information
- Adaptive to learner's level

ABSOLUTE CONSTRAINTS:
- NEVER diagnose conditions
- NEVER recommend specific treatments
- NEVER provide dosing information
- NEVER give emergency medical advice
- ALWAYS clarify this is educational

When uncertain, acknowledge limitations honestly."""

    EXPLAIN = """MODE: Medical Concept Explanation

Your task is to explain the medical topic clearly and thoroughly.

STRUCTURE (use naturally, not rigidly):
1. Start with a clear definition/overview
2. Explain the key mechanisms or concepts
3. Discuss clinical relevance (why it matters)
4. Highlight important points to remember
5. Offer to go deeper or test understanding

ADAPT TO LEARNER:
- Beginner: Focus on fundamentals, use analogies
- Intermediate: Include pathophysiology, clinical correlations
- Advanced: Discuss nuances, recent developments, edge cases

Be thorough but engaging. Explain terminology when first used."""

    QUIZ = """MODE: Assessment Generation

Create an educational quiz on the topic.

REQUIREMENTS:
- 3-4 multiple choice questions
- Use clinical vignettes where appropriate
- Include answer explanations (hidden until revealed)
- Add 3-5 rapid-fire flashcard Q&As

QUALITY STANDARDS:
- Questions should test understanding, not just recall
- Include common misconceptions as distractors
- Explanations should teach, not just reveal answers
- Match difficulty to learner level

Format answers with <details> tags for reveal."""

    PLAN = """MODE: Study Plan Generation

Create a practical, achievable study plan.

REQUIREMENTS:
- 5-7 day structure
- 30-60 minutes per day
- Mix of activities: reading, practice, review
- Clear daily objectives
- Progress checkpoints

PRINCIPLES:
- Start with foundations, build complexity
- Include active recall opportunities
- Space out review sessions
- Make it feel achievable, not overwhelming"""

    SEARCH = """MODE: Literature Search

You have access to the search_pubmed tool. Use it to find relevant research.

PROTOCOL:
1. ALWAYS call search_pubmed first with a focused query
2. Wait for results before writing
3. Synthesize findings from returned articles
4. Cite properly: Author et al., Year, Journal (PMID: xxx)

OUTPUT STRUCTURE:
- State what you searched
- Summarize key findings (2-3 most relevant papers)
- Synthesize: what does the evidence show?
- Note limitations or gaps

CRITICAL: Never fabricate citations. Only cite returned results."""

    CHAT = """MODE: Conversational Response

Respond naturally and briefly to greetings or simple interactions.

EXAMPLES:
- "Hi" → "Hey! What would you like to learn about today?"
- "Thanks" → "You're welcome! Any other questions?"
- "Bye" → "Take care! Good luck with your studies."

Keep it warm and natural. Don't give menus or lists."""

    HISTORY = """MODE: Progress Summary

Present the learner's study progress in a clear, encouraging way.

Include:
- Topics studied and mastery levels
- Items due for review
- Personalized recommendations

Be encouraging about progress made."""


# =============================================================================
# SECTION 9: SPECIALIZED AGENTS
# =============================================================================

# Create agents at module level for Pydantic compatibility
explain_agent = LlmAgent(
    model=Config.MODEL,
    name="explainer",
    description="Medical concept explanation",
    instruction=f"{AgentPrompts.SYSTEM}\n\n{AgentPrompts.EXPLAIN}",
    output_key="explanation"
)

quiz_agent = LlmAgent(
    model=Config.MODEL,
    name="quiz_master",
    description="Assessment generation",
    instruction=f"{AgentPrompts.SYSTEM}\n\n{AgentPrompts.QUIZ}",
    output_key="quiz"
)

plan_agent = LlmAgent(
    model=Config.MODEL,
    name="planner",
    description="Study plan creation",
    instruction=f"{AgentPrompts.SYSTEM}\n\n{AgentPrompts.PLAN}",
    tools=[ToolRegistry.save_study_plan],
    output_key="plan"
)

search_agent = LlmAgent(
    model=Config.MODEL,
    name="researcher",
    description="Literature search",
    instruction=f"{AgentPrompts.SYSTEM}\n\n{AgentPrompts.SEARCH}",
    tools=[ToolRegistry.search_pubmed],
    output_key="research"
)

chat_agent = LlmAgent(
    model=Config.MODEL,
    name="assistant",
    description="Conversational responses",
    instruction=f"{AgentPrompts.SYSTEM}\n\n{AgentPrompts.CHAT}",
    output_key="response"
)

router_agent = LlmAgent(
    model=Config.MODEL,
    name="router",
    description="Intent classification",
    instruction="""Classify the user's intent. Return ONLY JSON:
{"intent": "explain|quiz|plan|search|history|chat", "topic": "extracted topic or null"}

Rules:
- Medical questions → "explain"
- "quiz me on X" → "quiz"
- "study plan for X" → "plan"  
- "find papers/research on X" → "search"
- "my progress/history" → "history"
- Greetings only → "chat"

Default to "explain" for medical content.""",
    output_key="classification"
)

ALL_AGENTS = [explain_agent, quiz_agent, plan_agent, search_agent, chat_agent, router_agent]


# =============================================================================
# SECTION 10: ORCHESTRATOR (State Machine)
# =============================================================================

class MedGuide(BaseAgent):
    """
    Production-grade medical education orchestrator.
    
    Implements:
    - Explicit mode state machine
    - Safety-first processing pipeline
    - Deterministic routing with LLM fallback
    - Observability at every step
    - Adaptive learning integration
    """
    
    model_config = {"arbitrary_types_allowed": True}
    
    def __init__(self):
        super().__init__(
            name="medguide",
            description="AI-Powered Medical Learning Companion",
            sub_agents=ALL_AGENTS
        )
        self._current_mode = AgentMode.IDLE
        obs.log("INIT", "Orchestrator", "MedGuide initialized", 
               {"agents": len(ALL_AGENTS), "mode": self._current_mode.name})
    
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        """
        Main orchestration pipeline with explicit state transitions.
        
        Pipeline:
        1. Start trace → correlation ID
        2. Extract input → validate contract
        3. Safety check → boundaries
        4. Classify intent → deterministic + LLM
        5. Route to mode → execute agent
        6. Update memory → learning record
        7. Apply safety → disclaimer if needed
        """
        
        # ═══════════════════════════════════════════════════════════════════
        # STEP 1: Initialize Trace
        # ═══════════════════════════════════════════════════════════════════
        correlation_id = obs.start_trace()
        obs.log("REQUEST_START", "Orchestrator", "Processing request")
        
        # ═══════════════════════════════════════════════════════════════════
        # STEP 2: Extract & Validate Input
        # ═══════════════════════════════════════════════════════════════════
        user_text = self._extract_input(ctx)
        
        if not user_text:
            obs.log("INPUT_EMPTY", "Orchestrator", "No input received")
            yield self._create_event("I didn't catch that. What would you like to learn about?")
            return
        
        obs.log("INPUT_RECEIVED", "Orchestrator", f"Input: {user_text[:100]}...")
        
        # ═══════════════════════════════════════════════════════════════════
        # STEP 3: Safety Evaluation
        # ═══════════════════════════════════════════════════════════════════
        safety_flag = SafetyBoundary.evaluate(user_text)
        obs.log_safety_check(safety_flag, user_text)
        
        # Handle hard refusals
        if safety_flag in (SafetyFlag.REFUSE_DIAGNOSIS, SafetyFlag.REFUSE_TREATMENT, SafetyFlag.REFUSE_EMERGENCY):
            self._transition_mode(AgentMode.REFUSE, f"Safety: {safety_flag.name}")
            refusal = SafetyBoundary.get_refusal_response(safety_flag)
            yield self._create_event(refusal)
            return
        
        # ═══════════════════════════════════════════════════════════════════
        # STEP 4: Intent Classification
        # ═══════════════════════════════════════════════════════════════════
        mode, topic, confidence = IntentClassifier.classify(user_text)
        
        # For low confidence, use LLM router
        if confidence < 0.85 and mode == AgentMode.EXPLAIN:
            mode, topic = await self._llm_classify(ctx, user_text)
        
        obs.log("INTENT_CLASSIFIED", "Orchestrator", 
               f"Mode: {mode.name}, Topic: {topic}, Confidence: {confidence:.2f}")
        
        # ═══════════════════════════════════════════════════════════════════
        # STEP 5: Mode Execution
        # ═══════════════════════════════════════════════════════════════════
        self._transition_mode(mode, "Intent classification")
        
        needs_disclaimer = (safety_flag == SafetyFlag.NEEDS_DISCLAIMER)
        
        # Route to appropriate handler
        if mode == AgentMode.IDLE:
            # Greetings/chat
            async for event in chat_agent.run_async(ctx):
                yield event
        
        elif mode == AgentMode.HISTORY:
            # Direct response from memory
            summary = memory.get_progress_summary()
            recommendations = memory.get_recommendations()
            response = f"## Your Learning Progress\n\n{summary}\n\n## Recommendations\n\n{recommendations}"
            yield self._create_event(response)
        
        elif mode == AgentMode.SEARCH:
            # Literature search
            async for event in search_agent.run_async(ctx):
                if needs_disclaimer and hasattr(event, 'content'):
                    event = self._apply_disclaimer_to_event(event)
                yield event
            if topic:
                memory.record_study(topic, "search")
        
        elif mode == AgentMode.QUIZ:
            # Quiz generation
            async for event in quiz_agent.run_async(ctx):
                yield event
            if topic:
                memory.record_study(topic, "quiz")
        
        elif mode == AgentMode.PLAN:
            # Study planning
            async for event in plan_agent.run_async(ctx):
                yield event
            if topic:
                memory.record_study(topic, "plan")
        
        else:
            # Default: Explanation
            async for event in explain_agent.run_async(ctx):
                if needs_disclaimer and hasattr(event, 'content'):
                    event = self._apply_disclaimer_to_event(event)
                yield event
            if topic:
                memory.record_study(topic, "explain")
        
        # ═══════════════════════════════════════════════════════════════════
        # STEP 6: Finalize
        # ═══════════════════════════════════════════════════════════════════
        self._transition_mode(AgentMode.IDLE, "Request complete")
        obs.log("REQUEST_COMPLETE", "Orchestrator", "Done")
    
    def _extract_input(self, ctx: InvocationContext) -> str:
        """Safely extract user input from context."""
        try:
            if hasattr(ctx, 'user_content') and ctx.user_content:
                content = ctx.user_content
                if hasattr(content, 'parts'):
                    for part in content.parts:
                        if hasattr(part, 'text') and part.text:
                            return str(part.text).strip()
                return str(content).strip()
        except Exception as e:
            obs.log("INPUT_ERROR", "Orchestrator", f"Failed to extract input: {e}", error=str(e))
        return ""
    
    async def _llm_classify(self, ctx: InvocationContext, text: str) -> tuple[AgentMode, Optional[str]]:
        """Use LLM for complex intent classification."""
        try:
            async for _ in router_agent.run_async(ctx):
                pass
            
            result = ctx.session.state.get("classification", "")
            return self._parse_classification(result)
        except Exception as e:
            obs.log("CLASSIFY_ERROR", "Orchestrator", str(e), error=str(e))
            return AgentMode.EXPLAIN, None
    
    def _parse_classification(self, result: str) -> tuple[AgentMode, Optional[str]]:
        """Parse LLM classification result."""
        try:
            text = str(result).strip()
            # Find JSON in response
            match = re.search(r'\{[^{}]+\}', text)
            if match:
                data = json.loads(match.group())
                
                mode_map = {
                    "explain": AgentMode.EXPLAIN,
                    "quiz": AgentMode.QUIZ,
                    "plan": AgentMode.PLAN,
                    "search": AgentMode.SEARCH,
                    "history": AgentMode.HISTORY,
                    "chat": AgentMode.IDLE
                }
                
                intent = data.get("intent", "explain").lower()
                mode = mode_map.get(intent, AgentMode.EXPLAIN)
                topic = data.get("topic")
                
                return mode, topic
        except Exception:
            pass
        
        return AgentMode.EXPLAIN, None
    
    def _transition_mode(self, new_mode: AgentMode, reason: str):
        """Explicit mode transition with logging."""
        old_mode = self._current_mode
        self._current_mode = new_mode
        obs.log_mode_transition(old_mode, new_mode, reason)
    
    def _create_event(self, text: str) -> Event:
        """Create a response event."""
        return Event(
            author="medguide",
            content=types.Content(parts=[types.Part(text=text)])
        )
    
    def _apply_disclaimer_to_event(self, event: Event) -> Event:
        """Apply safety disclaimer to an event."""
        if hasattr(event, 'content') and hasattr(event.content, 'parts'):
            for part in event.content.parts:
                if hasattr(part, 'text') and part.text:
                    part.text = SafetyBoundary.apply_disclaimer(part.text)
        return event


# =============================================================================
# SECTION 11: EXPORTS
# =============================================================================

root_agent = MedGuide()

__all__ = ["root_agent", "MedGauide", "AgentMode", "SafetyBoundary", "MemoryManager"]
