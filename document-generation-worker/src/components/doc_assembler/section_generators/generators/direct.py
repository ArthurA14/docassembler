from typing import Any, Dict, List, Optional, Union

from haystack import component
from haystack.document_stores.types import DocumentStore
from document_common.components.base import (
    TextAssembly,
)
from document_common.components.writers.document_writer import DocumentWriter
from document_common.schemas import Document, Section, SectionStatus
from document_common.utils.logger import Logger

from src.components.doc_assembler.section_generators.base_registry import (
    SectionGenerator,
)


@SectionGenerator.register("direct", critical_documents=[])
@component
class DirectSectionGenerator(SectionGenerator):
    """
    Generates a section by passing other sections' content to an LLM prompt.
    
    Used for sections that transform or synthesize content from upstream sections
    (e.g., introductions, conclusions, executive summaries).
    """

    def __init__(
        self,
        prompt_name: str,
        section: Section,
        document_store: DocumentStore,
        document: Optional[Document] = None,
        document_writer: Optional[DocumentWriter] = None,
        context: Optional[dict[str, str]] = None,
        method_parameters: Optional[Dict[str, Any]] = None,
    ):
        self.logger = Logger().get_logger("section_generator.direct")
        SectionGenerator.__init__(
            self,
            prompt_name=prompt_name,
            section=section,
            document_store=document_store,
            document=document,
            document_writer=document_writer,
            context=context,
            method_parameters=method_parameters,
        )

        self.generator = TextAssembly(prompt_name=prompt_name, fixed_inputs=context)

    @component.output_types(section=List[Section])
    def run(self, sections: Union[List[Section], Section] = []):
        self.logger.info(f"Starting assembly of section '{self.section.name}'")
        self.section.status = SectionStatus.PROCESSING
        self.update_document_section()

        if isinstance(sections, Section):
            sections = [sections]

        try:
            self.logger.info(
                f"Generating section '{self.section.name}' with content from {[section.name for section in sections]}"
            )
            variables = {section.name: self._get_full_generated_content(section) for section in sections}
            variables = variables | {"section": self.section}

            self.section.generated_content = self.generator.run(inputs=variables)[  # type: ignore
                "llm_answer"
            ]

        except Exception as e:
            self.section.status = SectionStatus.FAILED
            self.update_document_section()
            self.logger.error(
                f"Error while generating section '{self.section.name}' : {e}"
            )
            raise e
        else:
            self.section.status = SectionStatus.COMPLETED
            self.update_document_section()
            self.logger.info(
                f"assembly finished for section '{self.section.name}'\nContent: {self.section.generated_content}"
            )
        return {"section": [self.section]}

    def _get_full_generated_content(self, section: Section, level:int = 1) -> str:
        generated_content = ""
        if section.title:
            generated_content += f"{'#'*level} {section.title}\n"
        if section.generated_content:
            generated_content += f"{section.generated_content}\n"

        if section.sections:
            for sub_section in section.sections:
                generated_content += self._get_full_generated_content(sub_section, level=level+1)
        
        return generated_content
