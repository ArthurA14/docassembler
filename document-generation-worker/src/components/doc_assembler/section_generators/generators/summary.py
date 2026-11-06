
from typing import Any, Dict, List, Optional, Union

from haystack import Document, component
from haystack.document_stores.types import DocumentStore
from pydantic import BaseModel
from document_common.components.base import (
    Retriever,
    StructuredAssembly,
    TextAssembly,
)
from document_common.components.writers.document_writer import DocumentWriter
from document_common.schemas import Document as AssembledDocument, Section, SectionStatus
from document_common.utils.logger import Logger

from src.components.doc_assembler.section_generators.base_registry import (
    SectionGenerator,
)


class PersonSummary(BaseModel):
    person: str
    summary: str


class SummariesBaseModel(BaseModel):
    summaries: List[PersonSummary]
    overall_summary: str


class SummaryAssemblyConfig(BaseModel):
    aggregated_person_summary_prompt: str
    final_overall_summary_prompt: str


@SectionGenerator.register("summary", critical_documents=["original"])
@component
class SummarySectionGenerator(SectionGenerator):
    """
    Generates a summary section from the provided documents.
    Aggregates summaries per person and creates an overall summary.
    Updates section status in the document and stores the generated content.
    Uses retrieval, structured LLM assembly, and text aggregation pipelines.
    """

    def __init__(
        self,
        prompt_name: str,
        section: Section,
        document_store: DocumentStore,
        method_parameters: Dict[str, Any],
        document: Optional[AssembledDocument] = None,
        document_writer: Optional[DocumentWriter] = None,
        context: Optional[dict[str, str]] = None,
    ):
        super().__init__(
            prompt_name=prompt_name,
            section=section,
            document_store=document_store,
            document=document,
            document_writer=document_writer,
            context=context,
            method_parameters=method_parameters,
        )
        self.logger = Logger().get_logger("section_generator.summary")
        self.config = SummaryAssemblyConfig(**method_parameters)

        # Initialize assembly components
        self._initialize_generators(prompt_name, context)
        self._initialize_retriever()

    def _initialize_generators(
        self, prompt_name: str, context: Optional[Dict[str, str]]
    ) -> None:
        """Initialize structured and text assembly components."""
        self.structured_generator = StructuredAssembly(
            prompt_name=prompt_name, fixed_inputs=context, base_model=SummariesBaseModel
        )

        self.text_generator = TextAssembly(
            prompt_name=self.config.aggregated_person_summary_prompt,
            fixed_inputs=context,
        )

    def _initialize_retriever(self) -> None:
        """Initialize document retriever for original documents."""
        self.retriever = Retriever(
            document_store=self.document_store, doc_type="original"
        )

    @component.output_types(section=List[Section])
    def run(
        self, sections: Union[List[Section], Section] = []
    ) -> Dict[str, List[Section]]:  # type: ignore
        """
        Generate summary section from documents.

        Args:
            sections: Optional sections parameter

        Returns:
            Dictionary containing the generated section
        """
        if sections is None:
            sections = []

        self._update_section_status(SectionStatus.PROCESSING)
        self.logger.info(f"Starting assembly of section '{self.section.name}'")

        try:
            # Retrieve and process documents
            documents = self._retrieve_sorted_documents()
            batch_summaries = self._process_documents_in_batches(documents)

            # Generate aggregated summaries
            person_summaries = self._generate_person_summaries(batch_summaries)
            overall_summary = self._generate_overall_summary(batch_summaries)

            # Build final content
            self._build_section_content(overall_summary, person_summaries)

            self._update_section_status(SectionStatus.COMPLETED)
            self.logger.info(f"Successfully generated section '{self.section.name}'")

        except Exception as error:
            self._handle_assembly_error(error)
            raise

        return {"section": [self.section]}

    def _retrieve_sorted_documents(self) -> List[Document]:
        """Retrieve all documents and sort them by order metadata."""
        documents = self.retriever.run(top_k=-1)["documents"]
        return sorted(documents, key=lambda doc: doc.meta.get("order", 0))

    def _process_documents_in_batches(
        self, documents: List[Document]
    ) -> List[SummariesBaseModel]:
        """Process documents in batches to generate structured summaries."""
        BATCH_SIZE = 10
        batch_summaries: List[SummariesBaseModel] = []

        for batch_start in range(0, len(documents), BATCH_SIZE):
            batch_end = batch_start + BATCH_SIZE
            document_batch = documents[batch_start:batch_end]

            summary_result = self.structured_generator.run(documents=document_batch)
            batch_summaries.append(summary_result["structured_reply"])  # type: ignore

        return batch_summaries

    def _generate_person_summaries(
        self, batch_summaries: List[SummariesBaseModel]
    ) -> Dict[str, str]:
        """Generate aggregated summaries for each person from batch summaries."""
        person_summary_texts = self._collect_person_summary_texts(batch_summaries)
        person_final_summaries: Dict[str, str] = {}

        for person, summary_list in person_summary_texts.items():
            aggregated_summary = self.text_generator.run(
                inputs={"discussion": "\n".join(summary_list), "person": person},
                prompt_name=self.config.aggregated_person_summary_prompt,
            )["llm_answer"]

            person_final_summaries[person] = aggregated_summary  # type: ignore

        return person_final_summaries

    def _collect_person_summary_texts(
        self, batch_summaries: List[SummariesBaseModel]
    ) -> Dict[str, List[str]]:
        """Collect and group summary texts by person from batch summaries."""
        person_summary_texts: Dict[str, List[str]] = {}

        for batch_summary in batch_summaries:
            for individual_summary in batch_summary.summaries:
                person_name = individual_summary.person

                if person_name not in person_summary_texts:
                    person_summary_texts[person_name] = []

                person_summary_texts[person_name].append(individual_summary.summary)

        return person_summary_texts

    def _generate_overall_summary(self, batch_summaries: List[SummariesBaseModel]) -> str:
        """Generate final overall summary from batch overall summaries."""
        overall_summaries = [
            batch.overall_summary
            for batch in batch_summaries
            if hasattr(batch, "overall_summary")
        ]

        if not overall_summaries:
            return "No overall summary available."

        final_summary_result = self.text_generator.run(
            inputs={"overall_summaries": "\n".join(overall_summaries)},
            prompt_name=self.config.final_overall_summary_prompt,
        )

        return final_summary_result["llm_answer"]  # type: ignore

    def _build_section_content(
        self, overall_summary: str, person_summaries: Dict[str, str]
    ) -> None:
        """Build the final section content from summaries."""
        content_parts = [f"Overall Summary:\n{overall_summary}"]

        if person_summaries:
            content_parts.append("\nSummary of each person:")
            person_content = "\n".join(
                [f"{person}: {summary}" for person, summary in person_summaries.items()]
            )
            content_parts.append(person_content)

        self.section.generated_content = "\n".join(content_parts)

    def _update_section_status(self, status: SectionStatus) -> None:
        """Update section status and document."""
        self.section.status = status
        self.update_document_section()

    def _handle_assembly_error(self, error: Exception) -> None:
        """Handle assembly errors with proper logging and status update."""
        self._update_section_status(SectionStatus.FAILED)
        self.logger.error(
            f"Error generating section '{self.section.name}': {error}", exc_info=True
        )
