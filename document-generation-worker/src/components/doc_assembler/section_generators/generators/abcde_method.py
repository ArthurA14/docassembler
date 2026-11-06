from typing import Any, Dict, List, Optional, Union

from haystack import component
from haystack.document_stores.types import DocumentStore
from pydantic import BaseModel
from document_common.components.base import (
    Retriever,
    TextAssembly,
)
from document_common.components.base.deduplicators import EntityDeduplicator
from document_common.components.writers.document_writer import DocumentWriter
from document_common.schemas import Document, Section, SectionStatus
from document_common.utils.logger import Logger

from src.components.doc_assembler.section_generators.base_registry import (
    SectionGenerator,
)


class AssemblyConfig(BaseModel):
    actors_prompt: str
    behavior_prompt: str
    content_prompt: str
    diffuse_prompt: str
    entities_prompt: str



@SectionGenerator.register("abcde_method", critical_documents=[])
@component
class ABCDEMethodSectionGenerator(SectionGenerator):
    """
    Generates a section using the ABCDE methodology: Actors, Behavior, Content, Diffusion, Entities.
    
    Retrieves tweets (or other documents) and entity extractions, then generates five subsections analyzing
    geopolitical actors, their behaviors, content produced, diffusion patterns, and named entities.
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
        
        self.parsed_params = AssemblyConfig(**method_parameters)

        self.retriever = Retriever(document_store=document_store)

        self.generator = TextAssembly(prompt_name=prompt_name, fixed_inputs=context)

        self.entity_dedup = EntityDeduplicator()

    @component.output_types(section=List[Section])
    def run(self, sections: Union[List[Section], Section] = []):
        self.logger.info(f"Starting assembly of section '{self.section.name}'")
        self.section.status = SectionStatus.PROCESSING
        self.update_document_section()

        try:
            tweets = self.retriever.run(top_k=-1, doc_type="original")["documents"]

            actor_section = self.generator.run(prompt_name=self.parsed_params.actors_prompt, documents=tweets)["llm_answer"]
            behavior_section = self.generator.run(prompt_name=self.parsed_params.behavior_prompt, documents=tweets)["llm_answer"]
            content_section = self.generator.run(prompt_name=self.parsed_params.content_prompt, documents=tweets)["llm_answer"]

            diffuse_section = self.generator.run(prompt_name=self.parsed_params.diffuse_prompt, inputs={"content":content_section})["llm_answer"]

            entities = self.retriever.run(top_k=-1, doc_type="entity_extraction")["documents"]
            refined_entities = self.entity_dedup.deduplicate_documents(docs=entities)
            refined_entities = self.entity_dedup.filter_by_LaJ(entities=refined_entities, subject="Ivory coast geopolitical situation")

            entities_section = self.generator.run(prompt_name=self.parsed_params.entities_prompt, inputs={"entities":refined_entities})["llm_answer"]

            self.section.generated_content = f"""# 1. Actors:
{actor_section}

# 2. Behavior
{behavior_section}

# 3. Content
{content_section}

# 4. Diffusion
{diffuse_section}

# 5. Entities
{entities_section}
"""
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
