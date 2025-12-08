from typing import List, Dict, Any
import re
import math

from app.utils.model_loader import build_classification_pipeline


class EmotionalAnalyzerService:
    def __init__(self):
        model_name = "emotion_model"
        self.pipeline = build_classification_pipeline(model_name)

        # Emotion type sensitivity adjustments
        self.emotion_scaling: Dict[str, float] = {
            "joy": 1.15,  # joy is often underdetected → boost
            "sadness": 1.00,
            "fear": 1.05,  # fear tends to appear more subtly → slight boost
            "anger": 0.85,  # anger is often overpredicted → reduce
            "disgust": 0.90,
            "surprise": 1.00,
            "neutral": 0.75,  # reduce neutral noise
        }

    # ---------------------------------------------------------
    # Phase splitting (detect emotional shifts inside sentences)
    # ---------------------------------------------------------
    def split_phases(self, text: str) -> List[str]:
        connectors = [
            " but ",
            " although ",
            " however ",
            " though ",
            " even though ",
            " because ",
            " so ",
            " yet ",
            " despite ",
            " nevertheless ",
            " still ",
        ]

        phases = [text]

        for c in connectors:
            new_phases = []
            for segment in phases:
                if c in segment.lower():
                    parts = re.split(c, segment, flags=re.IGNORECASE)
                    for p in parts:
                        p = p.strip()
                        if p:
                            new_phases.append(p)
                else:
                    new_phases.append(segment)
            phases = new_phases

        return phases

    # ---------------------------------------------------------
    # Base emotion analysis
    # ---------------------------------------------------------
    def analyze(self, text: str) -> List[Dict[str, Any]]:
        output = self.pipeline(text)
        first = output[0]
        scores = first if isinstance(first, list) else output

        # Apply emotion scaling adjustments
        adjusted = []
        for item in scores:
            label = item["label"]
            score = float(item["score"]) * self.emotion_scaling[label]
            adjusted.append({"label": label, "score": score})

        return sorted(adjusted, key=lambda x: x["score"], reverse=True)

    # ---------------------------------------------------------
    # Compute weight for a phase
    # ---------------------------------------------------------
    def compute_phase_weight(self, phase: str, index: int, total_phases: int) -> float:
        # Longer phases → more emotional content → higher weight
        length_weight = min(len(phase) / 60, 1.2)

        # Later phases often influence final emotional state → recency bonus
        position_weight = 1 + (index / max(total_phases, 1)) * 0.35

        # Emotional density: punctuation reflects intensity
        punctuation = phase.count("!") + phase.count("?")
        density_weight = 1 + (punctuation * 0.1)

        return length_weight * position_weight * density_weight

    # ---------------------------------------------------------
    # Prevent extreme words (death, suicide, trauma) from overpowering narrative
    # ---------------------------------------------------------
    def context_correction(self, label: str, score: float, phase: str) -> float:
        extreme_keywords = [
            "died",
            "death",
            "suicide",
            "kill",
            "trauma",
            "panic",
            "accident",
            "hurt",
            "hospital",
        ]

        if any(word in phase.lower() for word in extreme_keywords):
            if label in ["sadness", "fear", "anger"]:
                # reduce extreme emotional spike by 15%
                return score * 0.85

        return score

    # ---------------------------------------------------------
    # Emotional fusion algorithm
    # ---------------------------------------------------------
    def fuse_emotions(self, sentence_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        accumulator: Dict[str, float] = {
            "joy": 0.0,
            "sadness": 0.0,
            "fear": 0.0,
            "anger": 0.0,
            "disgust": 0.0,
            "surprise": 0.0,
            "neutral": 0.0,
        }

        for sentence in sentence_results:
            phases = sentence["phases"]
            total = len(phases)

            for i, phase_block in enumerate(phases):
                weight = self.compute_phase_weight(
                    phase_block["phase"], index=i, total_phases=total
                )

                for emotion in phase_block["emotions"]:
                    label = emotion["label"]
                    raw_score = float(emotion["score"])

                    corrected = self.context_correction(
                        label, raw_score, phase_block["phase"]
                    )

                    weighted_score = corrected * weight
                    accumulator[label] += weighted_score

        total_score = sum(accumulator.values())
        normalized = {k: v / total_score for k, v in accumulator.items()}

        primary = max(normalized, key=lambda k: normalized[k])
        primary_score = normalized[primary]

        secondary = [
            label
            for label, score in normalized.items()
            if label != primary and score >= primary_score * 0.40
        ]

        return {
            "primary": primary,
            "secondary": secondary,
            "composition": normalized,
        }

    # ---------------------------------------------------------
    # Full analysis engine
    # ---------------------------------------------------------
    def analyze_with_sentence_comparison(self, text: str) -> Dict[str, Any]:
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]

        sentence_results = []
        for sentence in sentences:
            phases = self.split_phases(sentence)

            phase_results = []
            for phase in phases:
                phase_results.append({"phase": phase, "emotions": self.analyze(phase)})

            sentence_results.append({"sentence": sentence, "phases": phase_results})

        global_result = self.analyze(text)
        fusion = self.fuse_emotions(sentence_results)

        return {
            "global_emotions": global_result,
            "sentence_emotions": sentence_results,
            "fusion": fusion,
        }


def get_emotion_service():
    return EmotionalAnalyzerService()
