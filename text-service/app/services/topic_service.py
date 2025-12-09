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
        try:
            self.nlp = spacy.load("es_core_news_md")
        except OSError:
            model = "es_core_news_md"
            # self.logger(f"Downloaded model {model}")
            download(model)
            self.nlp = spacy.load("es_core_news_md")

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
        similarity_threshold: float = 0.40,
    ):
        subtopics = self.extract_subtopics(text)

        if not subtopics:
            return {
                "subtopics": [],
                "topic_map": {},
                "primary_topic": None,
                "topic_scores": {},
                "scores_by_subtopic": {},
            }

        subtopic_embeddings = self.emb_model.encode(subtopics, convert_to_tensor=True)

        topic_keywords_embeddings = []
        topic_names = []

        for topic_data in general_topics:
            topic_name = topic_data["topic"]
            keywords = topic_data.get("keywords", [])

            emb = self._embed_topic_keywords(keywords)
            if emb is not None:
                topic_names.append(topic_name)
                topic_keywords_embeddings.append(emb)

        if not topic_keywords_embeddings:
            return {
                "subtopics": subtopics,
                "topic_map": {},
                "primary_topic": None,
                "topic_scores": {},
                "scores_by_subtopic": {},
            }

        topic_keywords_embeddings = torch.stack(topic_keywords_embeddings)

        topic_map = {t: [] for t in topic_names}
        topic_scores = {t: 0.0 for t in topic_names}

        # Added: store scores per subtopic for better analytics
        scores_by_subtopic = {}

        for i, subtopic in enumerate(subtopics):
            best_score, best_idx = self._best_idx_score_of_subtopics_with_topics(
                subtopic_embeddings[i],
                topic_keywords_embeddings,
            )

            # Added: save score for this subtopic even if threshold isn't passed
            scores_by_subtopic[subtopic] = {
                "best_topic": topic_names[best_idx],
                "score": best_score,
            }

            if best_score >= similarity_threshold:
                best_topic = topic_names[best_idx]
                topic_map[best_topic].append(subtopic)
                topic_scores[best_topic] = max(topic_scores[best_topic], best_score)

        filtered_map = self._filter_empty_topics(topic_map.items())
        primary_topic = self._determine_primary_topic(filtered_map, topic_scores)

        return {
            "subtopics": subtopics,
            "topic_map": filtered_map,
            "primary_topic": primary_topic,
            "topic_scores": topic_scores,  # Added
            "scores_by_subtopic": scores_by_subtopic,  # Added
        }

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
from pprint import pprint

topic = TopicService()

text = "Hoy me sentí muy ansioso en clase pero luego en el gimnasio me relajé."

general_topics = [
    {"topic": "colegio", "keywords": ["colegio", "clase", "estudio", "profesor"]},
    {"topic": "entrenar", "keywords": ["gimnasio", "ejercicio", "deporte", "entrenar"]},
]

result = topic.analyze_with_topics(text, general_topics)

pprint(result)
# TODO  DEVOLVER LOS SCORES TAMBIEN
