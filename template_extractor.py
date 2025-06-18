from transformers import AutoModel
import json
import os
import pickle
import numpy as np
from functools import lru_cache
import argparse

class TemplateMatcher:
    """
    A class for matching query templates against a catalog of templates using semantic similarity.

    This class loads a template catalog, builds an index of template embeddings,
    and provides methods to search for the best matching template based on input queries.
    It uses a pre-trained sentence embedding model for semantic encoding and caching for efficiency.
    """
    def __init__(self,
                 model_name="jinaai/jina-embeddings-v3",
                 catalog_path='data/template_library.json',
                 cache_dir='.cache'):
        """
        Initializes the TemplateMatcher.

        Args:
            model_name (str): Name of the pre-trained sentence embedding model to use.
            catalog_path (str): Path to the JSON file containing the template catalog.
            cache_dir (str): Directory to store cached embeddings for faster loading.
        """
        self.model_name = model_name
        self.catalog = self._load_catalog(catalog_path)
        self.cache_dir = cache_dir
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

        self._build_index()
        self._save_cache()

    def _load_catalog(self, path):
        """
        Loads the template catalog from a JSON file.

        Args:
            path (str): Path to the catalog JSON file.

        Returns:
            dict: The loaded template catalog as a dictionary.
        """
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _get_cache_path(self, prefix):
        """
        Constructs the cache file path for a given level.

        Args:
            prefix (str): Prefix indicating the level of cache (e.g., 'chapter', 'section', 'method').

        Returns:
            str: The full path to the cache file.
        """
        return os.path.join(self.cache_dir, f"{self.model_name}_{prefix}.pkl")

    def _try_load_cache(self):
        """
        Attempts to load cached embeddings for all levels (chapter, section, method).

        Returns:
            bool: True if all levels of cache were successfully loaded, False otherwise.
        """
        try:
            return all([
                self._load_cache_level('chapter'),
                self._load_cache_level('section'),
                self._load_cache_level('method')
            ])
        except Exception: # Catch broad exception for loading failures, consider specific exception handling if possible
            return False

    def _load_cache_level(self, level):
        """
        Loads cached embeddings for a specific level.

        Args:
            level (str): The level of cache to load (e.g., 'chapter', 'section', 'method').

        Returns:
            bool: True if the cache for the level was successfully loaded, False otherwise.
        """
        cache_path = self._get_cache_path(level)
        if not os.path.exists(cache_path):
            return False

        with open(cache_path, 'rb') as f:
            data = pickle.load(f)

        # Dynamically set level index attribute
        setattr(self, f"{level}_index", data)
        return True

    def _save_cache(self):
        """
        Saves cached embeddings for all levels (chapter, section, method).
        """
        self._save_cache_level('chapter', self.chapter_index)
        self._save_cache_level('section', self.section_index)
        self._save_cache_level('method', self.method_index)

    def _save_cache_level(self, level, data):
        """
        Saves cached embeddings for a specific level.

        Args:
            level (str): The level of cache to save (e.g., 'chapter', 'section', 'method').
            data (dict): The embedding data to be saved.
        """
        cache_path = self._get_cache_path(level) # Define cache_path here for clarity
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _build_index(self):
        """
        Builds the full index of embeddings for chapters, sections, and methods from the catalog.
        """
        # Chapter index
        self.chapter_index = {
            chapter: self._encode(chapter)
            for chapter in self.catalog
        }

        # Section index
        self.section_index = {}
        for chapter in self.catalog:
            self.section_index[chapter] = {
                section: self._encode(section)
                for section in self.catalog[chapter]
            }

        # Method index
        self.method_index = {}
        for chapter in self.catalog:
            self.method_index[chapter] = {}
            for section in self.catalog[chapter]:
                self.method_index[chapter][section] = [
                    self._encode(item['template_name'])
                    for item in self.catalog[chapter][section]
                ]

    @lru_cache(maxsize=5000)
    def _encode(self, text):
        """
        Encodes text into embeddings using the pre-trained model with caching.

        Args:
            text (str): The text to encode.

        Returns:
            numpy.ndarray: The embedding for the input text.
        """
        return self.model.encode(text, task="text-matching")

    @staticmethod
    def _find_best_match(query_embedding, candidates):
        """
        Finds the best matching text from candidates based on embedding similarity.

        Args:
            query_embedding (numpy.ndarray): Embedding of the query text.
            candidates (dict): Dictionary of candidate texts and their embeddings.

        Returns:
            tuple: The best matching text and its maximum similarity score.
        """
        max_similarity = -1.0 # Initialize with float for comparison with similarity
        best_match = None
        for text, embedding in candidates.items():
            similarity = query_embedding @ embedding.T
            if similarity > max_similarity:
                max_similarity = similarity
                best_match = text
        return best_match, max_similarity

    def _find_top_matches(self, query_embedding, candidates, top_n=2):
        """
        Finds the top N best matching texts from candidates based on embedding similarity.

        Args:
            query_embedding (numpy.ndarray): Embedding of the query text.
            candidates (dict): Dictionary of candidate texts and their embeddings.
            top_n (int): The number of top matches to return.

        Returns:
            list: A list of tuples, each containing a text and its similarity score, sorted by similarity in descending order.
        """
        similarities = [
            (text, query_embedding @ embedding.T)
            for text, embedding in candidates.items()
        ]
        # Sort by similarity in descending order
        sorted_matches = sorted(similarities, key=lambda x: x[1], reverse=True)
        return sorted_matches[:top_n]

    def search_template(self, chapter_query, section_query, method_query):
        """
        Searches for the best matching template across three levels: chapter, section, and method.

        Args:
            chapter_query (str): Query text for chapter level.
            section_query (str): Query text for section level.
            method_query (str): Query text for method level.

        Returns:
            dict: A dictionary containing the best matching template information, including chapter, section, method details, and confidence score.
                  Returns None if no match is found.
        """
        # Chapter matching
        chapter, _ = self._find_best_match(
            self._encode(chapter_query),
            self.chapter_index
        )

        # Section matching
        section, _ = self._find_best_match(
            self._encode(section_query),
            self.section_index[chapter]
        )
        section_candidates = self._find_top_matches(
            self._encode(section_query),
            self.section_index[chapter],
            top_n=2
        )

        # Final candidates for methods
        final_candidates = []

        for section, section_score in section_candidates:
            # Get method candidates for the section
            method_embeddings = {
                item['template_name']: emb
                for item, emb in zip(
                    self.catalog[chapter][section],
                    self.method_index[chapter][section]
                )
            }

            # Get best matching method
            method, method_score = self._find_best_match(
                self._encode(method_query),
                method_embeddings
            )
            print(method) # Consider using logging instead of print for debugging
            # Calculate weighted total score (weights can be adjusted)
            total_score = 0.9 * method_score + 0.1 * section_score
            final_candidates.append((
                section,
                method,
                total_score,
                self.catalog[chapter][section] # Keep the catalog section for final result extraction
            ))

        # Select the candidate with the highest total score
        best_match = max(final_candidates, key=lambda x: x[2])

        # Locate the final result in the template
        for item in best_match[3]:
            if item['template_name'] == best_match[1]:
                return {
                    'chapter': chapter,
                    'section': best_match[0],
                    'method': item,
                    'confidence': best_match[2]
                }
        return None
