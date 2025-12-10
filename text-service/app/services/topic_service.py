import keyword
from typing import Any, Dict, List
import numpy as np
from sentence_transformers import SentenceTransformer, util
import spacy
from spacy.cli.download import download
import torch
import yake


class TopicService:
    """Extract all subtopics"""

    def __init__(self) -> None:
        model = "en_core_web_md"
        try:
            self.nlp = spacy.load(model)
        except OSError:
            # self.logger(f"Downloaded model {model}")
            download(model)
            self.nlp = spacy.load(model)

        # Load embedding model (change implemented here)
        self.emb_model = SentenceTransformer("all-MiniLM-L6-v2")

    def extract_noun_phrases(self, text: str):
        """Extracts noun phrases and text structure"""
        doc = self.nlp(text)
        return [chunk.text.lower().strip() for chunk in doc.noun_chunks]

    def extract_keywords(self, text: str, max_kw=10):
        """Extract important keywords"""
        kw_extractor = yake.KeywordExtractor(lan="es", top=max_kw)
        keywords = kw_extractor.extract_keywords(text)
        return [kw for kw, score in keywords]

    def normalize_topic(self, topic: str):
        """
        Normalize a topic string by converting it into a clean and meaningful
        semantic representation.

        - Lemmatization: reduces words to their base form
        (e.g., "relajé" -> "relajación", "ansioso" -> "ansioso").

        - POS filtering: keeps only nouns and adjectives, which are the most
        informative parts of a topic. Verb-only phrases or irrelevant terms
        are discarded entirely (e.g., "sentí" -> "").

        - Lowercasing and whitespace cleanup.

        Returns:
            str: A normalized topic string containing only meaningful semantic units.
        """

        doc = self.nlp(topic)

        # Keep only lemmas from nouns and adjectives
        tokens = [
            token.lemma_.lower() for token in doc if token.pos_ in ["NOUN", "ADJ"]
        ]

        return " ".join(tokens).strip()

    def dedupe_topics(self, topics: list[str], threshold: float = 0.40) -> list[str]:
        """Remove semantically duplicate topics using embeddings"""
        if not topics:
            return []

        # Convert topics to vectors and generate a tensor
        embeddings = self.emb_model.encode(topics, convert_to_tensor=True)
        final_topics = []

        for i, topic in enumerate(topics):
            is_duplicate = False

            for j in range(i):
                sim = util.cos_sim(embeddings[i], embeddings[j]).item()

                if sim > threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                final_topics.append(topic)

        return final_topics

    def extract_candidates(self, text: str):
        """Combine keywords and noun phrases into a single list of candidates (bug fix implemented here)."""
        noun_phrases = self.extract_noun_phrases(text)
        keywords = self.extract_keywords(text)
        return list(set(noun_phrases + keywords))

    def extract_subtopics(self, text: str):
        """Full pipeline: extract, normalize, and dedupe topics (change implemented here)."""
        candidates = self.extract_candidates(text)

        # Normalize all candidates
        normalized = [self.normalize_topic(t) for t in candidates]
        normalized = [t for t in normalized if len(t) > 0]  # remove empty topics

        # Remove semantic duplicates
        final_topics = self.dedupe_topics(normalized)

        return final_topics

    # ----------------------------------------------------------
    # MATCHING SUBTOPICS -> GENERAL TOPICS (KEYWORDS EMBEDDINGS)
    # ----------------------------------------------------------
    def analyze_with_topics(
    self,
    text: str,
    general_topics: List[Dict[str, Any]],
    ) -> Dict[str, Any]:

        subtopics = self.extract_subtopics(text)

        if not subtopics:
            return {
                "subtopics": [],
                "topic_map": {},
                "primary_topic": None,
                "scores_by_topic": {},
                "scores_by_subtopic": {},
            }

        # Embeddings for subtopics
        sub_emb = self.emb_model.encode(subtopics, convert_to_tensor=True)

        # Prepare topic embeddings
        topic_names = []
        topic_emb_list = []

        for topic in general_topics:
            name = topic["topic"]
            keywords = topic.get("keywords", [])
            emb = self._embed_topic_keywords(keywords)
            if emb is not None:
                topic_names.append(name)
                topic_emb_list.append(emb)

        if not topic_emb_list:
            return {
                "subtopics": subtopics,
                "topic_map": {},
                "primary_topic": None,
                "scores_by_topic": {},
                "scores_by_subtopic": {},
            }

        topic_map = {t: [] for t in topic_names}
        scores_by_topic = {t: [] for t in topic_names}
        scores_by_subtopic = {}

        topic_emb_list = torch.stack(topic_emb_list, dim=0)
        for i, subtopic in enumerate(subtopics):
            sims = (
                util.cos_sim(sub_emb[i], topic_emb_list)[0]
                .cpu()
                .numpy()
            )

            best_idx = int(np.argmax(sims))
            best_topic = topic_names[best_idx]
            best_score = float(sims[best_idx])
            strength = self._strength_label(best_score)

            # Record information
            scores_by_subtopic[subtopic] = {
                "best_topic": best_topic,
                "score": best_score,
                "strength": strength,
            }

            topic_map[best_topic].append(
                {
                    "subtopic": subtopic,
                    "score": best_score,
                    "strength": strength,
                }
            )

            scores_by_topic[best_topic].append(best_score)

        # Remove empty topics
        topic_map = {t: subs for t, subs in topic_map.items() if subs}

        # Compute primary topic: average score
        primary_topic = None
        if topic_map:
            primary_topic = max(
                scores_by_topic.keys(),
                key=lambda t: sum(scores_by_topic[t]) / len(scores_by_topic[t])
                if scores_by_topic[t]
                else 0
            )

        return {
            "subtopics": subtopics,
            "topic_map": topic_map,
            "primary_topic": primary_topic,
            "scores_by_topic": scores_by_topic,
            "scores_by_subtopic": scores_by_subtopic,
        }

    def _strength_label(self, score: float) -> str:
        """Assigns a strength label based on similarity score."""
        if score >= 0.40:
            return "strong"
        if score >= 0.20:
            return "medium"
        if score >= 0.10:
            return "weak"
        return "very_weak"
    
    def _determine_primary_topic(self, filtered_map, scores):
        if not filtered_map:
            return None
        return max(filtered_map.keys(), key=lambda t: scores[t])

    def _filter_empty_topics(self, topic_map_items):
        return {t: subs for t, subs in topic_map_items if subs}

    def _best_idx_score_of_subtopics_with_topics(
        self,
        subtopic_embedding,
        topic_keywords_embeddings,
    ):
        sims = (
            util.cos_sim(subtopic_embedding, topic_keywords_embeddings)[0].cpu().numpy()
        )

        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])

        return best_score, best_idx

    def _embed_topic_keywords(self, keywords: List[str]):
        if not keywords:
            return None

        text = " ".join(keywords)
        return self.emb_model.encode(text, convert_to_tensor=True)


# TESTING
# from pprint import pprint

# topic = TopicService()

# text = "Hoy me sentí muy ansioso en clase pero luego en el gimnasio me relajé."

# general_topics = [
#     {"topic": "colegio", "keywords": ["colegio", "clase", "estudio", "profesor"]},
#     {"topic": "entrenar", "keywords": ["gimnasio", "ejercicio", "deporte", "entrenar"]},
# ]

# result = topic.analyze_with_topics(text, general_topics)

# pprint(result)
