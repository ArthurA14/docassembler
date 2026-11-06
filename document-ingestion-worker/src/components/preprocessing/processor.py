from typing import List, Set, Tuple, Union

import networkx as nx
from haystack import Document, Pipeline, component
from haystack.components.joiners.list_joiner import ListJoiner
from haystack.document_stores.errors import MissingDocumentError
from haystack.document_stores.types import DocumentStore
from document_common.components.base import (
    Writer,
)
from document_common.components.base.embedder import DocumentEmbedder
from document_common.utils.logger import Logger

from src.components.preprocessing.document_factories import (
    create_document_factory,
    get_document_types_for_factory,
)


@component
class Processor:
    """
    A Haystack component that generates sub-documents (chunks) of specific types from source documents.

    This component uses a `DocumentFactory` to create derived document types (e.g., summaries, key points,
    entity extractions) from existing documents. It handles duplicate detection by retrieving existing
    documents of the same type before generating new ones, and writes results to a document store.

    Haystack Inputs:
        - documents: List[Document] -- Source documents to process.
        - skip: bool -- If True, skips assembly and returns existing documents.
        - file_id: str -- File identifier used to query existing documents.

    Haystack Outputs:
        - documents: List[str] -- IDs of newly created documents (or existing if skip=True).
        - doc_types: List[str] -- List of document types generated.
    """

    def __init__(
        self,
        factory_type: str,
        document_store: DocumentStore,
        embed: bool = True,
        batch_size: int = 16,
    ):
        """
        Initialize the processor.

        Args:
            document_factory (DocumentFactory): Component responsible for generating new documents.
            document_store (DocumentStore): Store to check duplicates against and write results to.
            embed (bool, optional): Whether to embed generated documents before writing. Defaults to True.
            batch_size (int, optional): Number of input document IDs to process per batch. Defaults to 16.
        """
        self.logger = Logger().get_logger("processor")
        self.factory_type = factory_type

        self.document_factory = create_document_factory(name=factory_type)
        self.doc_types = get_document_types_for_factory(self.factory_type)
        self.pipeline = Pipeline()

        self.pipeline.add_component("document_factory", self.document_factory)

        if embed:
            self.pipeline.add_component("embedder", DocumentEmbedder())
        self.pipeline.add_component("writer", Writer(document_store=document_store))

        if embed:
            self.pipeline.connect("document_factory", "embedder")
            self.pipeline.connect("embedder", "writer")
        else:
            self.pipeline.connect("document_factory", "writer")

        self.batch_size = batch_size
        self.document_store = document_store

    def _get_document_if_exists(self, file_id: str) -> List[Document]:
        conditions = {
            "operator": "AND",
            "conditions": [
                {"field": "meta.file_id", "operator": "==", "value": file_id},
                {
                    "operator": "OR",
                    "conditions": [
                        {"field": "meta.type", "operator": "==", "value": dt}
                        for dt in self.doc_types
                    ],
                },
            ],
        }
        self.logger.info("Retrieving")
        new_documents = self.document_store.filter_documents(conditions)
        return new_documents

    @component.output_types(documents=List[Document], doc_types=List[str] | None)
    def run(
        self,
        documents: List[Document],
        skip: bool = False,
        file_id: str | None = None,
    ) -> dict[str, List[str | Document] | None]:
        if skip:
            return {
                "documents": self._get_document_if_exists(file_id),
                "doc_types": self.doc_types,
            }

        if not documents:
            self.logger.error(
                f"Last Processor failed, this processor ({self.doc_types}) relies on it."
            )
            return {"documents": [], "doc_types": None}

        self.logger.info(f"Generating sub chunks {self.doc_types}")

        try:
            new_documents = self.pipeline.run(
                {"document_factory": {"documents": documents}}
            )["writer"]["documents"]

            self.logger.info(f"Sub chunks {self.doc_types} done.")
        except Exception as e:
            docs = self._get_document_if_exists(
                file_id
            )  # We clean in case of partial assembly
            if docs:
                try :
                    self.document_store.delete_documents([d.id for d in docs])
                except MissingDocumentError as e:
                    self.logger.warning(f"While trying to clean the evaluation document store, a MissingDocumentError was raised : {e}")
                
            self.logger.error(
                f"Error while generating sub chunks {self.doc_types} : {e}", exc_info=True
            )
            return {"documents": [], "doc_types": None}
        else:
            return {"documents": new_documents, "doc_types": self.doc_types}

@component
class ProcessorManager:
    """
    Orchestrates multiple `Processor` components in a pipeline.

    Each `Processor` generates sub-documents (chunks) of a specific type.
    `ProcessorManager` ensures that:
      - Each `doc_type` is unique.
      - Processors that have no upstream dependencies receive the original documents.
      - Custom dependencies between processors can be defined via `connections`.
      - All outputs are collected and returned.

    Haystack Inputs:
        - documents: List[Document] -- Source documents to process.
        - file_id: str -- File identifier for querying existing documents.
        - processings_to_skip: List[str] -- List of processor names to skip.

    Haystack Outputs:
        - doc_types: List[str] -- List of all sub-chunk types successfully created.
        - processings: List[str] -- List of all successfully executed processor names.
    """

    def __init__(
        self,
        processings: List[str],
        connections: List[Tuple[str, str]],
        document_store: DocumentStore,
    ):
        """
        Initialize the ProcessorManager.

        Args:
            processors (List[Processor], optional): List of processors to orchestrate. Defaults to [].
            connections (List[tuple[str, str]], optional): Custom connections between processors,
                where each tuple (producer, consumer) means the output of producer feeds into consumer.
                Defaults to [].

        Raises:
            ValueError: If doc_type is not unique, or if connections reference non-existent processors,
                        or if a cycle is detected in the dependency graph.
        """
        self.logger = Logger().get_logger("processor_manager")

        self.processings = processings or []
        self.processors = [
            Processor(factory_type=p_type, document_store=document_store)
            for p_type in processings
        ]
        self.connections = connections or []

        self._validate_processor_setup(
            self.processings, self.processors, self.connections
        )

        self.pipeline = self._build_pipeline(
            self.processings, self.processors, self.connections
        )

    def _validate_processor_setup(
        self,
        processings: List[str],
        processors_obj: List[Processor],
        connections: List[Tuple[str, str]],
    ):
        # Validate unique doc_types (aka the name of the chunks in the DB)
        doc_types = [dt for p in processors_obj for dt in p.doc_types]
        if len(doc_types) != len(set(doc_types)):
            raise ValueError(
                "Each processor should produce unique 'doc_types'. Found duplicates: "
                + f"{[t for t in doc_types if doc_types.count(t) > 1]}"
            )

        # Validate that all connections refer to existing processors
        for producer, consumer in connections:
            if producer not in processings:
                raise ValueError(
                    f"Connection error: Producer '{producer}' does not exist."
                )
            if consumer not in processings:
                raise ValueError(
                    f"Connection error: Consumer '{consumer}' does not exist."
                )

        # Detect cycles in dependency graph using networkx
        self._validate_no_cycles(set(processings), connections)

    def _validate_no_cycles(
        self, processings: Set[str], connections: List[Tuple[str, str]]
    ):
        """
        Detect cycles in the dependency graph using directed graph analysis.

        Raises:
            ValueError: If a cycle is detected.
        """
        G = nx.DiGraph()
        G.add_nodes_from(processings)
        G.add_edges_from(connections)

        try:
            cycle = list(nx.find_cycle(G, orientation="original"))
            if cycle:
                # Extract source nodes of each edge in the cycle
                nodes_in_cycle = [edge[0] for edge in cycle]
                # Close the cycle by appending the first node
                first_node = cycle[0][0]
                cycle_str = " -> ".join(nodes_in_cycle + [first_node])
                raise ValueError(
                    f"While building the preprocessing pipeline, a circular dependency was detected in connections: {cycle_str}"
                )
        except nx.NetworkXNoCycle:
            pass  # No cycle, all good

    def _build_pipeline(
        self,
        processings: List[str],
        processors_obj: List[Processor],
        connections: List[Tuple[str, str]],
    ):
        pipeline = Pipeline()

        # Add input and output joiners
        pipeline.add_component("input", ListJoiner(list_type_=List[Document]))
        pipeline.add_component("output", ListJoiner(list_type_=List[Document]))

        # Add all processors to the pipeline
        consumers = {conn[1] for conn in connections}

        for name, proc in zip(processings, processors_obj):
            pipeline.add_component(name, proc)

            if name not in consumers:
                pipeline.connect("input", f"{name}.documents")

        # Connect custom dependencies
        for producer, consumer in connections:
            pipeline.connect(f"{producer}.documents", f"{consumer}.documents")

        # Connect all processors to output
        for name in processings:
            pipeline.connect(f"{name}.documents", "output")

        return pipeline

    def _normalize_documents(
        self, documents: Union[List[str], List["Document"]]
    ) -> List[str]:
        if all(isinstance(d, Document) for d in documents):
            return [doc.id for doc in documents]  # type: ignore
        elif all(isinstance(d, str) for d in documents):
            return documents  # type: ignore
        else:
            raise TypeError(
                "Input 'documents' must be either List[str] (IDs) or List[Document]. "
                + f"Got mixed types: {[type(d).__name__ for d in documents]}"
            )

    @component.output_types(doc_types=List[str], processors=List[str])
    def run(
        self, documents: List[Document], file_id: str, processings_to_skip: List[str] | None = None
    ) -> dict[str, List[str]]:
        """
        Run the orchestrated processors on the given documents.

        Args:
            documents: List of document IDs or Document objects to process.

        Returns:
            dict: {
                "doc_types": List[str], — Flat List of all sub chunk types successfully created.
                "processors": List[str], — Flat List of all successfull processors.
            }
        """
        processings_to_skip = processings_to_skip or []
        processing_not_valid = [
            p for p in processings_to_skip if p not in self.processings
        ]
        if processing_not_valid:
            self.logger.warning(
                f"You asked to skip processings that was not in the processings scheduled : {processing_not_valid}"
            )
        if processings_to_skip:
            self.logger.info(f"Some processing have already been applied to this file, skipping : {processings_to_skip}")
        
        # Run pipeline
        data = {}
        for p in self.processings:
            data[p] = {"file_id": file_id}
            if p in processings_to_skip:
                data[p]["skip"] = True
        data["input"] = {"values": documents}

        result = self.pipeline.run(data, include_outputs_from=self.processings)

        successful_doc_types = []
        successful_processings = []

        for p in self.processings:
            p_output = result.get(p, {})
            doc_types = p_output.get("doc_types")

            if doc_types:  # if we have doc_types
                successful_doc_types.extend(doc_types)
                successful_processings.append(p)

        return {"doc_types": successful_doc_types, "processings": successful_processings}
