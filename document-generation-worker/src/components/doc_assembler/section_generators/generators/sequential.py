# Deprecated
#  from typing import Any, Dict, List, Optional, Union

# from haystack import Pipeline, component
# from haystack.document_stores.types import DocumentStore
# from pydantic import BaseModel

# from config import config
# from document_common.components.base import (
#     Retriever,
#     Structuredassembly,
#     Textassembly,
# )
# from document_common.components.base.embedder import TextEmbedder
# from src.components.doc_assembler.citation_generators import CitationExtractor
# from src.components.doc_assembler.models import document, Section, SectionStatus
# from src.components.doc_assembler.document_writer import documentWriter
# from src.components.doc_assembler.section_generators.base_registry import (
#     SectionGenerator,
# )
# from document_common.utils.logger import Logger


# class RecursiveassemblyConfig(BaseModel):
#     doc_type: str
#     paragraphs_count: int
#     auto_title: bool
#     titles: Optional[List[Optional[str]]] = None
#     top_k: int
#     citations: bool


# class documentT1Outline(BaseModel):
#     problematic: str
#     paragraphs: List[str]


# @SectionGenerator.register("sequential")
# @component
# class SequentialSectionGenerator(SectionGenerator):
#     """
#     Generates a section sequentially by first creating a structured outline and then generating
#     paragraphs one by one, using previous paragraphs as context to avoid repetition.

#     The generator can focus on a specific subject when provided in the context, or generate content
#     that covers the entire document corpus if no subject is specified.

#     This implementation uses a two-stage pipeline architecture:
#     1. Outline assembly: Creates a structured T1 outline with paragraph titles
#     2. Paragraph assembly: Generates each paragraph sequentially using the outline title and
#        previously generated paragraphs as context to ensure content diversity
#     """

#     def __init__(
#         self,
#         prompt_name: str,
#         section: Section,
#         document_store: DocumentStore,
#         method_parameters: Dict[str, Any],
#         document: Optional[document] = None,
#         document_writer: Optional[documentWriter] = None,
#         context: Optional[dict[str, str]] = None,
#         outline_prompt_name: str = "t1_outline",
#         default_language: str = "English",
#         section_heading_prefix: str = "### ",
#         extract_citations: Optional[bool] = None,
#     ):
#         """
#         Initialize the SequentialSectionGenerator.

#         Args:
#             prompt_name: Name of the prompt template to use for paragraph assembly
#             section: The section object that will be populated with generated content
#             document_store: The document store from which relevant documents will be retrieved
#             method_parameters: Configuration parameters for assembly
#             document: The parent document object (if available)
#             document_writer: The document writer for output formatting (if available)
#             context: Additional context for assembly, including optional 'subject' key
#             outline_prompt_name: Name of the prompt template for generating the T1 outline (default: "t1_outline")
#             default_language: Default language for assembly if not provided in context (default: "English")
#             section_heading_prefix: Markdown prefix for section headings (default: "### ")
#             extract_citations: Enable citation extraction for this section (None = read from config)
#         """
#         self.logger = Logger().get_logger("section_generator.sequential")

#         # Store prompt_name as an instance attribute for use in the class
#         self.prompt_name = prompt_name

#         # Same for outline_prompt_name | default_language | section_heading_prefix
#         self.outline_prompt_name = outline_prompt_name
#         self.default_language = default_language
#         self.section_heading_prefix = section_heading_prefix

#         # Initialize parent class with all parameters
#         SectionGenerator.__init__(
#             self,
#             prompt_name=prompt_name,
#             document_store=document_store,
#             section=section,
#             document=document,
#             document_writer=document_writer,
#             context=context,
#             method_parameters=method_parameters,
#         )

#         # Extract subject from context if provided
#         self.subject = context.get("subject") if context else None

#         # Load assembly configuration
#         self.assembly_config = RecursiveassemblyConfig(**method_parameters)
#         self.logger.debug(f"assembly configuration: {self.assembly_config}")

#         # Build the assembly pipelines
#         self.outline_assembly_pipeline = self._build_outline_assembly_pipeline()
#         self.paragraph_assembly_pipeline = self._build_paragraph_assembly_pipeline()
#         self.citation_assembly_pipeline = self._build_citation_assembly_pipeline(
#             extract_citations, config.get_current_config(), document_store
#         )

#     def _build_outline_assembly_pipeline(self) -> Pipeline:
#         """
#         Build the pipeline for generating the T1 outline.

#         The pipeline can be configured to:
#         - Use text embedding if a subject is provided to focus retrieval
#         - Retrieve top-k relevant documents
#         - Generate a structured outline using the t1_outline prompt

#         Returns:
#             Configured Pipeline object for outline assembly
#         """
#         pipeline = Pipeline()

#         # Add embedder component if subject is provided for focused retrieval
#         if self.subject:
#             pipeline.add_component("embedder", TextEmbedder())

#         # Add retriever component to fetch relevant documents
#         pipeline.add_component(
#             "retriever",
#             Retriever(
#                 doc_type=self.assembly_config.doc_type,
#                 document_store=self.document_store,
#                 top_k=self.assembly_config.top_k,
#             ),
#         )

#         # Add outline generator component
#         predefined_titles = self._get_predefined_titles()

#         pipeline.add_component(
#             "outline_generator",
#             Structuredassembly(
#                 prompt_name=self.outline_prompt_name,
#                 base_model=documentT1Outline,
#                 fixed_inputs=(self.context or {})
#                 | {
#                     "section_count": self.assembly_config.paragraphs_count,
#                     "subject": self.subject or "",
#                     "titles_already_decided": predefined_titles,
#                     "language": (self.context or {}).get("language", self.default_language),
#                 },
#             ),
#         )

#         # Connect components
#         if self.subject:
#             pipeline.connect("embedder", "retriever")
#         pipeline.connect("retriever", "outline_generator")

#         return pipeline

#     def _get_predefined_titles(self) -> list[str]:
#         """
#         Extract and preprocess predefined titles from configuration.

#         Skips empty titles and formats them as "i. title" for prompt context.

#         Returns:
#             List of formatted predefined titles
#         """
#         titles_already_decided = []
#         if not self.assembly_config.auto_title and self.assembly_config.titles:
#             for i, title in enumerate(self.assembly_config.titles):
#                 if title:  # Skip empty titles
#                     titles_already_decided.append(f"{i}. {title}")
#         return titles_already_decided

#     def _build_paragraph_assembly_pipeline(self) -> Pipeline:
#         """
#         Build the pipeline for generating individual paragraphs.

#         The pipeline:
#         - Embeds the paragraph title to guide retrieval
#         - Retrieves relevant documents based on the title
#         - Generates the paragraph content using the provided prompt

#         Returns:
#             Configured Pipeline object for paragraph assembly
#         """
#         pipeline = Pipeline()

#         pipeline.add_component("embedder", TextEmbedder())
#         pipeline.add_component(
#             "retriever",
#             Retriever(
#                 doc_type=self.assembly_config.doc_type,
#                 document_store=self.document_store,
#             ),
#         )
#         pipeline.add_component(
#             "section_generator",
#             Textassembly(
#                 prompt_name=self.prompt_name,
#                 fixed_inputs=(self.context or {}),
#             ),
#         )

#         pipeline.connect("embedder", "retriever")
#         pipeline.connect("retriever.documents", "section_generator.documents")

#         return pipeline

#     def _build_citation_assembly_pipeline(self, extract_citations, config, document_store):
#         # Initialize citation extraction if enabled
#         self.extract_citations = extract_citations
#         self.citation_extractor = None

#         if extract_citations is None:
#             # Read from config
#             citation_config = config.citations
#             self.extract_citations = citation_config.get("enable_in_documents", True)

#         if self.extract_citations:
#             try:
#                 self.citation_extractor = CitationExtractor(document_store=document_store)
#                 self.logger.info("Citation extraction enabled for this section")
#             except Exception as e:
#                 self.logger.warning(f"Failed to initialize citation extractor: {e}")
#                 self.citation_extractor = None
#         elif self.extract_citations:
#             self.logger.warning("Citation extraction requested but citations module not available")
#             self.extract_citations = False

#     @component.output_types(section=List[Section])
#     def run(self, sections: Union[List[Section], Section] = []) -> Dict[str, List[Section]]:
#         """
#         Generate section content by first creating an outline and then generating paragraphs based on that outline.

#         This method:
#         1. Creates a T1 outline using either a subject (if provided) or document-wide retrieval
#         2. Generates paragraphs sequentially, using previous paragraphs as context to avoid repetition
#         3. Extracts source IDs from retrieved documents for citation purposes
#         4. Formats the final content with markdown headings

#         Args:
#             sections: List of sections or single section to process (unused in this implementation)

#         Returns:
#             Dictionary with generated section content

#         Raises:
#             Exception: If assembly fails at any stage
#         """
#         self.logger.info(f"Starting assembly of section '{self.section.name}'")
#         self.section.status = SectionStatus.PROCESSING
#         self.update_document_section()

#         try:
#             # Step 1: Generate T1 outline
#             self.logger.info("Generating T1 outline...")
#             if self.subject:
#                 outline_result = self.outline_assembly_pipeline.run(
#                     {
#                         "embedder": {
#                             "text": self.subject,
#                         }
#                     },
#                     include_outputs_from={"outline_generator"},
#                 )
#             else:
#                 outline_result = self.outline_assembly_pipeline.run({"retriever": {}})

#             self.logger.debug(f"Retriever results for outline assembly: {outline_result}")
#             document_t1_outline: documentT1Outline = outline_result["outline_generator"]["structured_reply"]
#             self.logger.debug(f"Generated outline with {len(document_t1_outline.paragraphs)} paragraphs")
#             self.logger.info(f"Outline preview : {document_t1_outline.paragraphs}")

#             # Step 2: Generate paragraphs sequentially
#             generated_paragraph_texts = []
#             self.section.source_ids = []  # Reset source_ids

#             for outline_title in document_t1_outline.paragraphs:
#                 # Run the paragraph assembly pipeline
#                 section_result = self.paragraph_assembly_pipeline.run(
#                     {
#                         "embedder": {
#                             "text": outline_title,
#                         },
#                         "section_generator": {
#                             "inputs": (self.context or {})
#                             | {
#                                 "title": outline_title,
#                                 "previous_paragraphs": generated_paragraph_texts,
#                                 "min_length": self.section.num_lines_min,
#                                 "max_length": self.section.num_lines_max,
#                                 "language": (self.context or {}).get("language", self.default_language),
#                             }
#                         },
#                     },
#                     include_outputs_from={"retriever", "section_generator"},
#                 )
#                 self.logger.debug(f"Retriever output: {section_result['retriever']}")

#                 # Extract generated text
#                 paragraph_text = section_result["section_generator"]["llm_answer"]
#                 generated_paragraph_texts.append(paragraph_text)

#                 # Extract source IDs from retriever output
#                 retriever_output = section_result["retriever"]
#                 source_ids: list[str] = []

#                 self.logger.debug(f"Created section using documents:\n{retriever_output['documents']}")
#                 for doc in retriever_output["documents"]:
#                     # Traverse up the parent chain until we find the original document
#                     current_doc = doc
#                     while (
#                         hasattr(current_doc, "meta")
#                         and "type" in current_doc.meta
#                         and current_doc.meta["type"] != "original"
#                     ):
#                         if "parent_id" not in current_doc.meta:
#                             break
#                         parent_id = current_doc.meta["parent_id"]
#                         parent_docs = self.document_store.filter_documents(
#                             filters={"field": "id", "operator": "==", "value": parent_id}
#                         )
#                         if not parent_docs:
#                             break
#                         current_doc = parent_docs[0]
#                     source_ids.append(current_doc.id)
#                 self.section.source_ids.append(source_ids)

#                 self.logger.debug(f"Generated paragraph: {outline_title[:50]}...")

#             # Step 3: Store individual paragraph texts for citation extraction
#             # And compute cumulative offsets for each section to fix citation span positioning
#             self.section.sections = generated_paragraph_texts
#             self.section.section_offsets = []

#             current_offset = 0
#             for i, (title, paragraph) in enumerate(zip(document_t1_outline.paragraphs, generated_paragraph_texts)):
#                 # Calculate offset of this section in the full generated content
#                 self.section.section_offsets.append(current_offset)
#                 # Update current_offset for next section
#                 section_title_length = len(f"{title}\n")
#                 current_offset += section_title_length + len(paragraph)

#             # Step 4: Format final content with markdown headings
#             self.section.generated_content = self._format_content_with_headings(
#                 generated_paragraph_texts, document_t1_outline.paragraphs
#             )

#         except Exception as e:
#             self.section.status = SectionStatus.FAILED
#             self.update_document_section()
#             self.logger.error(
#                 f"Error while generating section '{self.section.name}': {e}",
#                 exc_info=True,
#             )
#             raise

#         finally:
#             self.section.status = SectionStatus.COMPLETED
#             self.update_document_section()
#             self.logger.info(
#                 f"assembly finished for section '{self.section.name}'\nContent: {self.section.generated_content}"
#             )

#             # Extract citations if enabled
#             self.extract_citations_from_sections()

#         return {"section": [self.section]}

#     def extract_citations_from_sections(self):
#         if self.extract_citations and self.citation_extractor:
#             self.logger.info(f"Extracting citations for section '{self.section.name}'")
#             try:
#                 self.section = self.citation_extractor.extract_for_section(
#                     section=self.section, request_id=f"{self.section.job_id}-{self.section.section_id}"
#                 )
#                 # Update section with citation data
#                 self.update_document_section()
#             except Exception as citation_error:
#                 self.logger.error(
#                     f"Citation extraction failed for section '{self.section.name}': {citation_error}", exc_info=True
#                 )
#                 # Don't fail the entire section assembly, just log the error
#                 self.section.citation_extraction_status = "failed"
#                 self.update_document_section()

#     def _format_content_with_headings(self, paragraphs: list[str], titles: list[str]) -> str:
#         """
#         Format the generated content with markdown headings.

#         Args:
#             paragraphs: List of generated paragraph texts
#             titles: List of corresponding outline titles

#         Returns:
#             Formatted content with markdown headings
#         """
#         formatted_content: list[str] = []
#         for paragraph, title in zip(paragraphs, titles):
#             formatted_content.append(f"{self.section_heading_prefix} {title}\n{paragraph}")

#         return "\n".join(formatted_content)
