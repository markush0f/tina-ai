from typing import Any, Dict, List
import numpy as np
from sentence_transformers import SentenceTransformer, util
import spacy
from spacy.cli.download import download
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

    def extract_topics(self, text: str):
        """Full pipeline: extract, normalize, and dedupe topics (change implemented here)."""
        candidates = self.extract_candidates(text)

        # Normalize all candidates
        normalized = [self.normalize_topic(t) for t in candidates]
        normalized = [t for t in normalized if len(t) > 0]  # remove empty topics

        # Remove semantic duplicates
        final_topics = self.dedupe_topics(normalized)

        return final_topics

    def _match_subtopics_to_topics(
        self,
        subtopics: List[str],
        general_topics: List[Dict[str, List]],
    ) -> Dict[str, Any]:
        """
        For each subtopic, find the most similar general topic using embeddings.
        Returns a mapping topic -> [subtopics...] and the primary topic.
        """
        if not subtopics or not general_topics:
            return {
                "subtopics": subtopics,
                "topic_map": {t: [] for t in general_topics},
                "primary_topic": None,
            }

        # Embeddings of topics and subtopics
        topic_embeddings = self.emb_model.encode(general_topics, convert_to_tensor=True)
        subtopic_embeddings = self.emb_model.encode(subtopics, convert_to_tensor=True)

        # Accumulate which subtopics fall under each topic
        topic_map: Dict[str, List[str]] = {t: [] for t in general_topics}

        # Stores, for each topic, the greatest similarity it has had with any subtopic.
        topic_max_score: Dict[str, float] = {t: 0.0 for t in general_topics}

        # Assign each subtopic to a topic
        for i, sub in enumerate(subtopics):
            sims = util.cos_sim(subtopic_embeddings[i], topic_embeddings)[0]
            #
            sims_np = sims.cpu().numpy()
            print("Convertimos a numpy: ", sims_np)

            # Take the most similar index
            best_idx = int(np.argmax(sims_np))
            # Take the best topic for subtopic
            best_topic = general_topics[best_idx]
            # Take the best score of subtopic
            best_score = float(sims_np[best_idx])

            # Save subtopic inside topic selected
            topic_map[best_topic].append(sub)

            # Save the best score to that topic
            topic_max_score[best_topic] = max(topic_max_score[best_topic], best_score)

        # Filter topics without subtopics
        filtered_topic_map = {t: subs for t, subs in topic_map.items() if subs}
        print(f"Filtered topics: {filtered_topic_map}")
        # Calculate primary_topic only between the topics with subtopics
        if filtered_topic_map:
            primary_topic = max(
                filtered_topic_map.keys(),
                key = lambda t: topic_max_score[t]
            )
        primary_topic = None
        if any(topic_map.values()):
            primary_topic = max(topic_max_score, key=topic_max_score.get)

        return {
            "subtopics": subtopics,
            "topic_map": topic_map,
            "primary_topic": primary_topic,
        }

    def analyze_with_topics(
        self,
        text: str,
        general_topics: List[str],
    ) -> Dict[str, Any]:
        """
        Main entrypoint:
        - extract subtopics from text
        - match each subtopic to the closest general topic
        """
        subtopics = self.extract_topics(text)
        result = self._match_subtopics_to_topics(subtopics, general_topics)
        return result


# TESTING
topic = TopicService()
text = "Hoy me sentí muy ansioso en clase pero luego en el gimnasio me relajé."
# print(topic.extract_noun_phrases(text))
# print(topic.extract_keywords(text))
# candidates = topic.extract_candidates(text)
# dedupe = topic.dedupe_topics(candidates)
# print("DEDUPE: ", dedupe)
# topics = topic.extract_topics(text)
# print("Topics: ", topics)
analizer = topic.analyze_with_topics(text,["colegio", "entrenar"])
print(analizer)
# def show_model_location():
#     path = get_package_path("es_core_news_md")
#     print(f"Model installed at: {path}")


# show_model_location()
