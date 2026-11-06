import json
from collections import defaultdict
from typing import List, Literal, Optional, Tuple, cast

import numpy as np
from haystack import Document, component
from pydantic import BaseModel
from rapidfuzz import fuzz
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

from config import config
from document_common.components.base.batched_operations import batch_function_calls
from document_common.components.base.embedder import DocumentEmbedder, EmbedderFactory
from document_common.components.base.assembly import StructuredAssembly
from document_common.schemas.entities import Entity
from document_common.utils.logger import Logger


@component
class DocumentKmeansDeduplicators:
    """
    A semantic deduplication component that clusters documents based on their embeddings
    using KMeans and selects the top-N most centered documents from each cluster.

    It uses an external embedding API (e.g., via Docker) through the `DocEmbedder`.

    Attributes:
        top_n (int): Number of most centered documents to return per cluster.
        n_clusters (int): Number of clusters for KMeans.
        embedder (DocEmbedder): Embedding service used to encode documents.
    """

    def __init__(self, top_n: int = 3, n_clusters: int = 5):
        self.top_n = top_n
        self.n_clusters = n_clusters
        self.embedder = DocumentEmbedder()

    @component.output_types(documents=List[List[Document]])
    def run(self, documents: List[Document]):
        docs_to_embed = [doc for doc in documents if not doc.embedding]
        docs_embedded = [doc for doc in documents if doc.embedding]

        self.embedder.run(docs_to_embed)["documents"]

        documents = docs_to_embed + docs_embedded

        embeddings = [doc.embedding for doc in documents]
        X = np.array(embeddings)

        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X)
        distances = cdist(X, kmeans.cluster_centers_)

        top_centered = defaultdict(list)
        # Step 1: Get distances from each point to its cluster centroid
        for idx, label in enumerate(cluster_labels):
            dist_to_centroid = distances[idx][label]
            top_centered[label].append((idx, dist_to_centroid))  # store (index, distance)

        # Step 3: Sort within each cluster by distance
        for label in top_centered:
            top_centered[label] = sorted(top_centered[label], key=lambda x: x[1])[:3]

        top_centered_docs = []
        # Step 4: Show results
        for label in sorted(top_centered):
            tmp = []
            for idx, dist in top_centered[label]:
                tmp.append(documents[idx])
            top_centered_docs.append(tmp)
        return {"documents": top_centered_docs}


class Rating(BaseModel):
    rating: int


class Ratings(BaseModel):
    ratings: List[Rating]

class CombinedScoringConfig(BaseModel):
    fuzzy_weight: float
    name_weight: float
    context_weight: float
    combined_threshold: float

class StepScoringConfig(BaseModel):
    name_embedding_threshold: float
    context_embedding_threshold: float
    fuzzy_threshold: float

class EntityDeduplicatorConfig(BaseModel):
    method: Literal["combined_scoring", "step_scoring"]
    prompt_name: str
    laj_prompt_name: str
    combined_scoring_config: CombinedScoringConfig
    step_scoring_config: StepScoringConfig

class EntityDeduplicator:
    def __init__(
        self,
        prompt_name: Optional[str] = None,
        method: Optional[str] = None,
        fuzzy_weight: Optional[float] = None,
        name_weight: Optional[float] = None,
        context_weight: Optional[float] = None,
        combined_threshold: Optional[float] = None,
        name_embedding_threshold: Optional[float] = None,
        context_embedding_threshold: Optional[float] = None,
        fuzzy_threshold: Optional[float] = None,
        fixed_inputs: Optional[dict[str, str]] = None,
        laj_prompt_name: Optional[str] = None,
    ):
        """
        Initialize the EntityDeduplicator with optional overrides.
        Explicit arguments take precedence over configuration values.
        """
        entity_config_dict = config.get_current_config().entity_extraction
        entity_config = EntityDeduplicatorConfig(**entity_config_dict)

        # Extract nested configs for clarity
        combined_cfg = entity_config.combined_scoring_config
        step_cfg = entity_config.step_scoring_config

        # overall params
        self.prompt_name: str = prompt_name or entity_config.prompt_name
        self.method: str = method or entity_config.method
        self.laj_prompt_name = laj_prompt_name or entity_config.laj_prompt_name
        
        # combined scoring params
        self.fuzzy_weight: float = fuzzy_weight or combined_cfg.fuzzy_weight
        self.name_weight: float = name_weight or combined_cfg.name_weight
        self.context_weight: float = context_weight or combined_cfg.context_weight
        self.combined_threshold: float = combined_threshold or combined_cfg.combined_threshold

        # step scoring params
        self.name_embedding_threshold: float = name_embedding_threshold or step_cfg.name_embedding_threshold
        self.context_embedding_threshold: float = context_embedding_threshold or step_cfg.context_embedding_threshold
        self.fuzzy_threshold: float = fuzzy_threshold or step_cfg.fuzzy_threshold

        self.fixed_inputs = fixed_inputs or {}
            
        if any(v is None for v in [
            self.fuzzy_weight,
            self.name_weight,
            self.context_weight,
            self.combined_threshold
        ]):
            raise ValueError(
                "In the Entity Deduplicator, all scoring weights and thresholds must be  set either as arguments or in the configuration (configs/entity_extraction)."
            )

        self.embedder_factory = EmbedderFactory()
        self.embedder = self.embedder_factory.get_embedder()
        self.generator = StructuredAssembly(
            prompt_name=self.prompt_name, base_model=Entity, fixed_inputs=fixed_inputs
        )
        self.logger = Logger().get_logger("entity_deduplicator")

    def deduplicate_documents(
        self,
        docs: List[Document],
    ) -> List[Entity]:
        """
        Extracts Entity objects from Document.content (which contains JSON-encoded Entity),
        then deduplicates them using the entity deduplication pipeline.

        Args:
            docs: List of Document objects where each Document.content is a JSON string
                  representing an Entity.

        Returns:
            List of deduplicated and refined Entity objects.
        """
        entities = self.documents_to_entities(docs)

        if self.method == "combined_scoring":
            refined_entities = self.deduplicate_with_combined_scoring(entities)
        elif self.method == "step_scoring":
            refined_entities = self.deduplicate_with_steps(entities)
        else: 
            raise ValueError("Deduplication method should be combined_scoring or step_scoring")
        
        if len(entities)>=1:
            dedup_rate = 1 - (len(refined_entities) / len(entities))
        else :
            dedup_rate = 0.0
        self.logger.debug(f"Deduplication rate : {dedup_rate:.2%} ({len(entities)} → {len(refined_entities)} entities)")

        return refined_entities

    def documents_to_entities(self, documents: List[Document]) -> List[Entity]:
        entities: List[Entity] = []

        for doc in documents:
            if not doc.content:
                continue  # Skip empty content

            try:
                entity_dict = json.loads(doc.content)

                entity = Entity(**entity_dict)
                entities.append(entity)

            except (json.JSONDecodeError, TypeError, ValueError):
                self.logger.warning(f"Doc {doc.id} failed validation into an Entity. It has been skipped.")
                continue
        return entities

    def deduplicate_with_combined_scoring(self, entities: List[Entity]) -> List[Entity]:
        """
        Deduplicate entities using combined similarity scoring with batch processing.

        Groups entities by combined fuzzy, name, and context similarity scores,
        then refines groups using batch LLM processing for improved performance.
        """
        if not entities:
            return []

        groups = self._group_by_combined_similarity(entities)

        self._log_groups(groups)

        new_refined_entities: List[Entity] = []
        refinement_tasks = []
        task_indices = []

        # Prepare groups for batch processing
        for idx, group in enumerate(groups):
            if len(group) <= 1:
                # Singleton groups don't need LLM refinement
                new_refined_entities.extend(group)
            else:
                # Groups requiring LLM refinement
                refinement_tasks.append(group)
                task_indices.append(idx)

        # Batch process entity refinements for performance
        if refinement_tasks:
            refined_entities = batch_function_calls(
                self._refine_entity_group,
                [[group] for group in refinement_tasks],
                [None for _ in refinement_tasks],
                batch_size=32,  # Optimal batch size for LLM processing
            )

            # Handle results and exceptions
            for result in refined_entities:
                if isinstance(result, Exception):
                    self.logger.warning(f"Failed to refine entity group: {result}")
                    continue
                new_refined_entities.append(result)

        return new_refined_entities

    def _group_by_combined_similarity(
        self,
        entities: List[Entity],
        fuzzy_weight: Optional[float] = None,
        name_weight: Optional[float] = None,
        context_weight: Optional[float] = None,
        combined_threshold: Optional[float] = None,
    ) -> List[List[Entity]]:
        """Group entities by combining fuzzy matching score and cosine similarity of name embeddings with weighted sum."""
        if not entities:
            return []

        fuzzy_weight = fuzzy_weight or self.fuzzy_weight
        name_weight = name_weight or self.name_weight
        context_weight = context_weight or self.context_weight
        combined_threshold = combined_threshold or self.combined_threshold

        if (fuzzy_weight+name_weight+context_weight) != 1:
            self.logger.warning("All weights combined are != 1.")

        # Precompute name embeddings for all entities
        name_embeddings = self.embedder.run([e.name for e in entities])['embeddings']
        context_embeddings = self.embedder.run([e.context for e in entities])['embeddings']
        remaining = list(range(len(entities)))
        groups: List[List[Entity]] = []

        self.logger.debug(f"{'NAME':20}| FUZZY  | NAME   | CONTEX | COMBINED")
        while remaining:
            idx = remaining.pop(0)
            group = [entities[idx]]

            self.logger.debug(f"{entities[idx].name:20}| 0.0000 | 0.0000 | 0.0000")

            to_remove: List[int] = []
            for other_idx in remaining:
                # Fuzzy matching score (normalized 0-1)
                fuzzy_score = fuzz.partial_ratio(entities[idx].name.lower(), entities[other_idx].name.lower()) / 100.0

                # Cosine similarity between embeddings
                name_embedding_sim = self._compute_cosine_similarity(name_embeddings[idx], name_embeddings[other_idx])

                # Cosine similarity between embeddings
                context_embedding_sim = self._compute_cosine_similarity(
                    context_embeddings[idx], context_embeddings[other_idx]
                )

                # Combined score with weights
                combined_score = (
                    (fuzzy_weight * fuzzy_score)
                    + (name_weight * name_embedding_sim)
                    + (context_weight * context_embedding_sim)
                )

                if combined_score >= combined_threshold:
                    group.append(entities[other_idx])
                    to_remove.append(other_idx)

                    self.logger.debug(
                        f"{entities[other_idx].name:20}| {fuzzy_score:6.3f} | {name_embedding_sim:6.3f} | {context_embedding_sim:6.3f} | {combined_score:6.3f}"
                    )

            # Remove matched entities from remaining
            for r in to_remove:
                remaining.remove(r)

            groups.append(group)

        return groups

    def deduplicate_with_steps(self, entities: List[Entity]) -> List[Entity]:
        """
        Deduplicate entities in three stages:
        1. Fuzzy match on name → group
        2. Semantic similarity on name embeddings → merge groups
        3. Semantic similarity on context embeddings → merge groups
        Returns: List of lists, each inner list is a group of deduplicated entities
        """
        if not entities:
            return []

        # Stage 1: Fuzzy matching on name
        groups = self._fuzzy_group_by_name(entities)

        self.logger.info("Fuzzy Grouping Results :")
        self._log_groups(groups)

        # Stage 2: Merge groups using name embeddings
        groups = self._merge_groups_by_embeddings(groups, attribute="name")

        self.logger.info("Grouping by 'name' embedding Results:")
        self._log_groups(groups)

        # Stage 3: Merge groups using context embeddings
        groups = self._merge_groups_by_embeddings(groups, attribute="context")

        self.logger.info("Grouping by 'context' embedding Results:")
        self._log_groups(groups)

        # Final Stage: groups are rewrote by the llm into one entity.
        new_refined_entities: List[Entity] = []
        for group in groups:
            if len(group) <= 1:
                new_refined_entities += group
            else:
                llm_answer = self.generator.run(inputs={"entities": group})
                entity: Entity = llm_answer["structured_reply"]  # type: ignore
                new_refined_entities.append(entity)

        return new_refined_entities

    def _refine_entity_group(self, group_list: List[List[Entity]]) -> Entity:
        """
        Refine a group of entities into a single entity using LLM assembly.

        Args:
            group_list: List containing a single group of entities to consolidate

        Returns:
            Single refined Entity object
        """
        group = group_list[0]  # Extract the group from the wrapper list
        llm_answer = self.generator.run(inputs={"entities": group})
        entity: Entity = llm_answer["structured_reply"]  # type: ignore
        return entity

    def _log_groups(self, groups: List[List[Entity]]):
        for group in groups:
            message = ""
            for e in group:
                message = message + f"{e.name}\n"
            self.logger.debug(message)

    def _fuzzy_group_by_name(self, entities: List[Entity]) -> List[List[Entity]]:
        """Group entities by fuzzy matching on name."""
        remaining = entities[:]
        groups: List[List[Entity]] = []

        while remaining:
            entity = remaining.pop(0)
            group = [entity]

            # Compare with all remaining entities
            to_remove: List[Entity] = []
            for other in remaining:
                score = fuzz.partial_ratio(entity.name.lower(), other.name.lower())
                if score >= self.fuzzy_threshold:
                    group.append(other)
                    to_remove.append(other)

            # Remove matched entities
            for r in to_remove:
                remaining.remove(r)

            groups.append(group)

        return groups

    def _compute_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two embedding vectors."""
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm == 0:
            return 0.0
        return float(np.dot(v1, v2) / norm)

    def _get_embedding(self, text: str) -> List[float]:
        return self.embedder._embed_text(text)

    def _get_mean_embedding_from_group(self, texts: List[str]) -> List[float]:
        embeddings = [self._get_embedding(text) for text in texts]

        return np.mean(embeddings, axis=0)

    def _merge_groups_by_embeddings(self, groups: List[List[Entity]], attribute: str = "name") -> List[List[Entity]]:
        """Merge groups if their name embeddings are semantically similar above threshold."""
        if len(groups) <= 1:
            return groups

        # Compute embeddings for each group's representative (first entity)
        group_embeddings: List[List[float]] = []
        for group in groups:
            if attribute == "name":
                emb = self._get_mean_embedding_from_group([g.name for g in group])  # Use mean vector of whole group
                # emb = self._get_embedding(group[0].name)  # Use first entity as representative
            elif attribute == "context":
                emb = self._get_mean_embedding_from_group([g.context for g in group])  # Use mean vector of whole group
                # emb = self._get_embedding(group[0].name)  # Use first entity as representative
            else:
                raise ValueError(f"The attribute '{attribute}' is not supported.")
            group_embeddings.append(emb)

        merged: List[List[Entity]] = []
        used = [False] * len(groups)

        for i in range(len(groups)):
            if used[i]:
                continue
            new_group = groups[i][:]
            used[i] = True

            for j in range(i + 1, len(groups)):
                if used[j]:
                    continue
                sim = self._compute_cosine_similarity(
                    group_embeddings[i], group_embeddings[j]
                )
                if attribute == "name":
                    if sim >= self.name_embedding_threshold:
                        new_group.extend(groups[j])
                        used[j] = True
                elif attribute == "context":
                    if sim >= self.context_embedding_threshold:
                        new_group.extend(groups[j])
                        used[j] = True

            merged.append(new_group)

        return merged

    def filter_by_LaJ(
            self, 
            entities: List[Entity], 
            subject: Optional[str] = None, 
            prompt_name: Optional[str] = None, 
            batch_size: int = 5,
            threshold: float = 3
        ) -> List[Entity]:
        """
        Filter entities by LLM-assigned ratings using Structuredassembly.

        Args:
            prompt_name (str): Name of the prompt template to use.
            entities (List[Entity]): List of entities to evaluate.
            subject (str): Subject or topic context for the evaluation.
            batch_size (int): Number of entities per batch.
            threshold (float): relevance threshold for the entities
        Returns:
            List[Entity]: Filtered list of entities that meet the rating threshold.
        """
        self.logger.debug("Starting the LaJ filtering.")
        prompt_name = prompt_name or self.prompt_name

        filtered_entities: List[Entity] = []

        # Instantiate the structured generator
        rating_generator = StructuredAssembly(
            prompt_name=self.laj_prompt_name,
            base_model=Ratings,
            fixed_inputs={"subject": subject}
        )

        # Iterate in batches
        for i in range(0, len(entities), batch_size):
            batch: List[Entity] = entities[i : i + batch_size]

            # Run assembly
            try:
                result = rating_generator.run(inputs={"entities": [e.model_dump_json(indent=2) for e in batch]})[
                    "structured_reply"
                ]
                ratings_model: Ratings = cast(Ratings, result)
            except Exception as e:
                self.logger.error(f"Rating assembly failed for batch {i // batch_size}: {e}")
                continue

            # Filter entities based on rating
            for entity, rating_obj in zip(batch, ratings_model.ratings):
                self.logger.debug(f"[LaJ] NAME : {entity.name}, CONTEXT: {entity.context}, RATING : {rating_obj.rating}")
                if rating_obj.rating >= threshold:  # threshold can be adjusted
                    filtered_entities.append(entity)

        self.logger.debug(f"[LaJ] Kept entities : {[e.name for e in filtered_entities]}")
        return filtered_entities
