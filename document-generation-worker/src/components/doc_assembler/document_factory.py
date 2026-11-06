from pathlib import Path
from typing import Dict, List, Optional, Set, Union

import yaml
from haystack import Pipeline, component
from haystack.components.joiners.list_joiner import ListJoiner
from haystack.document_stores.types import DocumentStore
from pydantic import ValidationError
from document_common.components.writers.document_writer import DocumentWriter
from document_common.schemas import File, Document, DocumentStatus, Section
from document_common.schemas.document_template import DocumentTemplate
from document_common.utils.logger import Logger

from src.components.doc_assembler.section_generators.base_registry import (
    SectionGenerator,
)


@component
class SectionGeneratorManager:
    """
    Orchestrates multiple `SectionGenerator` components into a pipeline.

    Each `SectionGenerator` produces a `Section`, and their execution order
    is determined by dependency constraints (`connections`). If a section
    requires input from multiple other sections, a `ListJoiner` is inserted
    to merge their outputs.

    This manager ensures:
      - All section generators are registered in the pipeline.
      - Dependencies between sections are enforced.
      - There is exactly one "entry point" (a section with no dependencies).
      - All final sections are collected and returned in order.

    Inputs:
        - none (the entry section is triggered internally with an empty list of sections)

    Outputs:
        - sections: List[Section] -- All generated sections, one per SectionGenerator.
    """

    def __init__(
        self,
        section_generators: List[SectionGenerator],
        connections: dict[str, list[str]],
    ):
        """
        Initialize the SectionGeneratorManager.

        Args:
            section_generators (List[SectionGenerator]): Section generators to orchestrate.
            connections (dict[str, list[str]]): Dependency graph, where each key is a section
                name that requires input from a list of other section names.
        """
        self.pipeline = Pipeline()

        self.section_names = [ptp.section.name for ptp in section_generators]

        for sg in section_generators:
            self.pipeline.add_component(sg.section.name, sg)

        for i, (receiver, senders) in enumerate(connections.items()):
            self.pipeline.add_component(
                f"joiner{i}", ListJoiner(list_type_=List[Section])
            )
            for sender in senders:
                self.pipeline.connect(sender, f"joiner{i}")
            self.pipeline.connect(f"joiner{i}", receiver)

        first = [name for name in self.section_names if name not in connections.keys()]

        if len(first) != 1:
            raise ValueError(
                "The template selected does not have a section that do not require another section to start so the assembly can't start from anywhere."
            )

        self.first = first[0]

    @component.output_types(sections=List[Section])
    def run(self) -> dict[str, List[Section]]:
        """
        Run the pipeline and generate all sections.

        Returns:
            dict: {"sections": List[Section]} - All generated sections.
        """
        result = self.pipeline.run(
            {self.first: {"sections": []}}, include_outputs_from=set(self.section_names)
        )

        sections = [result[p]["section"][0] for p in self.section_names]

        return {"sections": sections}

    # def show(self, server_url:str = config.get_current_config().others.mermaid_url):
    #     return self.pipeline.show(server_url=server_url)

    # def draw(self, path:str, server_url:str = config.get_current_config().others.mermaid_url):
    #     return self.pipeline.draw(path=Path(path), server_url=server_url)


class DocumentFactory:
    """
    Factory that builds and executes a document assembly pipeline based on a YAML (or string) template.

    Responsibilities:
      - Load and validate a document template.
      - Dynamically construct `SectionGenerator` objects for each section.
      - Wire generators together according to dependency rules (`using`).
      - Execute the pipeline via `SectionGeneratorManager`.
      - Update and persist the status of the document during execution.

    Typical workflow:
        1. Load a template (YAML file or string).
        2. Validate its configuration (unique section names, valid dependencies, no cycles).
        3. Build section generators + dependency graph.
        4. Run the pipeline asynchronously to generate all sections.
    """

    def __init__(
        self,
        template: Union[Path, str],
        document: Document,
        document_writer: DocumentWriter,
        document_store: DocumentStore,
        context: Optional[dict[str, str]] = None,
    ):
        """
        Initialize a documentFactory from a template.

        Args:
            template (Union[Path, str]): Path to a YAML template file or raw YAML string.
            document (document): document object (tracks job ID, status, etc.).
            document_writer (documentWriter): Persists updates to the document object.
            document_store (DocumentStore): Document store used by section generators.
            context (Optional[dict]): Extra static inputs available to section generators. Defaults to None.
        """
        self.logger = Logger().get_logger("document_factory")
        # We load the template (can be a file or a str already loaded)
        if isinstance(template, Path):
            with open(template, "r") as f:
                data = yaml.safe_load(f)
            document_template = DocumentTemplate(**data)
        elif type(template) is str:
            document_template = DocumentTemplate(**yaml.safe_load(template))
        else:
            raise ValueError("template must be of type Path or str")

        self.logger.info(f"Template {document_template.template_name} loaded")

        self._validate_document_template(document_template)

        # Now we dynamically build the rest of the pipeline based on a document configuration
        section_generators: List[SectionGenerator] = []
        connections: dict[str, List[str]] = {}
        self.critical_documents: List[str] = []
        self.optional_documents: List[str] = []
        for i, section in enumerate(document_template.document):
            section_content = section.section

            # Convert SectionContent to Section explicitly
            section_obj = Section(
                title=section_content.title,
                order=i,
                name=section_content.name,
                num_lines_min=section_content.num_lines.min,
                num_lines_max=section_content.num_lines.max,
                job_id=document.job_id,
            )

            try:
                section_generators.append(
                    SectionGenerator.from_code(
                        code=section_content.method,
                        prompt_name=section_content.prompt,
                        section=section_obj,
                        document_store=document_store,
                        document=document,
                        document_writer=document_writer,
                        context=context,
                        method_parameters=section_content.method_parameters,
                    )
                )
                needed_documents = SectionGenerator.get_document_requirements(code=section_content.method)
                self.critical_documents.extend(needed_documents["critical"])
                self.optional_documents.extend(needed_documents["optional"])
            except ValidationError as e:
                message = f"Error in while adding SectionGenerator '{section_content.method}', the method_parameters in your template are not compatible with the method you chose. Details : {e}"
                self.logger.error(message)
                raise ValueError(message)

            if section_content.using and len(section_content.using) > 0:
                connections[section_content.name] = section_content.using

        self.pgm = SectionGeneratorManager(
            section_generators=section_generators, connections=connections
        )

        self.document = document
        self.document_writer = document_writer

    def _validate_document_template(self, document_template: DocumentTemplate):
        """
        Validate the structure and constraints of a document template.

        Checks performed:
          1. All section names are unique.
          2. "using" references only point to existing sections.
          3. No section references itself.
          4. No circular dependencies between sections.
          5. At least one section has no dependencies (entry point).

        Args:
            document_template (documentTemplate): The parsed template configuration.

        Raises:
            ValueError: If validation fails with detailed error messages.
        """
        errors: List[str] = []

        # --- 1. Check unique names ---
        names: List[str] = [block.section.name for block in document_template.document]
        if len(names) != len(set(names)):
            duplicates = [name for name in set(names) if names.count(name) > 1]
            errors.append(f"Duplicate section 'name' values found: {duplicates}")

        # --- 2. Validate "using" references ---
        name_set: Set[str] = set(names)
        for block in document_template.document:
            sec = block.section
            if sec.using:
                for dep_name in sec.using:
                    if dep_name not in name_set:
                        errors.append(
                            f"Section '{sec.name}' uses '{dep_name}' but no section with that name exists."
                        )
                    elif dep_name == sec.name:
                        errors.append(f"Section '{sec.name}' cannot use itself.")

        # --- 3. Detect circular dependencies ---
        # Build graph: node = name, edges = dependencies (names)
        graph: Dict[str, List[str]] = {
            block.section.name: block.section.using or []
            for block in document_template.document
        }

        def has_cycle(node: str, visited: Set[str], stack: Set[str]) -> bool:
            visited.add(node)
            stack.add(node)
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor, visited, stack):
                        return True
                elif neighbor in stack:
                    return True
            stack.remove(node)
            return False

        visited_nodes: Set[str] = set()
        for node in graph:
            if node not in visited_nodes:
                if has_cycle(node, visited_nodes, set()):
                    errors.append("Circular dependency detected in 'using' references.")
                    break

        # --- 4. Ensure at least one section has no dependencies ---
        if all(len(block.section.using or []) > 0 for block in document_template.document):
            errors.append(
                "All sections depend on other sections; at least one must have no dependencies."
            )

        # --- document results ---
        if errors:
            self.logger.error("\n".join(errors))
            raise ValueError("\n".join(errors))
        else:
            self.logger.info(
                f"documentTemplate validation of {document_template.template_name} passed successfully."
            )

    def validate_document_availability(self, files: List[File]) -> tuple[bool, str]:
        critical_set = set(self.critical_documents) 
        optional_set = set(self.optional_documents) 

        message = ""
        validated = True

        for file in files:
            doc_types_set = set(file.doc_types or []) 

            missing = optional_set - doc_types_set
            if missing:
                message += f"File {file.core_data_id} misses optional doc type {missing}\n"

            missing = critical_set - doc_types_set
            if missing:
                message += f"File {file.core_data_id} misses critical doc type {missing}\n"
                validated = False

        if message:
            self.logger.warning(message)
        
        return validated, message
    
    @component.output_types(sections=Optional[List[Section]])
    async def run_async(self):
        """
        Run the document assembly pipeline asynchronously.

        Updates the document status throughout the process:
          - PROCESSING at start
          - FAILED if assembly errors
          - COMPLETED if successful

        Returns:
            dict: {"sections": Optional[List[Section]]} - Generated sections or None if failed.
        """
        self.document.status = DocumentStatus.PROCESSING
        self.document_writer.update_document(self.document)
        try:
            sections = self.pgm.run()
        except Exception as e:
            self.logger.error(f"document assembly failed due to error : {e}")
            self.document.status = DocumentStatus.FAILED
            self.document_writer.update_document(self.document)
            return {"sections": None}
        else:
            self.document.status = DocumentStatus.COMPLETED
            self.document_writer.update_document(self.document)
        return sections
