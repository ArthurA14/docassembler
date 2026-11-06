from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, ValidationInfo, field_validator
from document_common.schemas import DocumentStatus, Section, SectionStatus
from document_common.schemas.api import DocRef
from document_common.schemas.citations import (
    CitationStack,
    SourceReference,
    SourceReferences,
)


class GenerateDocumentRequest(BaseModel):
    templateId: str = Field(
        ...,
        description=(
            "ID of the document template to apply. Must correspond to an existing template. "
            + "Use `GET /templates` to retrieve the list of valid template IDs."
            + "or found them in the configuration at config/configs/templates/document."
        ),
        examples=["t1", "t2"],
    )
    docs: List[DocRef] = Field(
        ..., description="List of document the document will be generate from."
    )
    context: Optional[Dict[str, str]] = Field(
        default_factory=dict,
        description="Additional contextual data to guide document assembly.",
        examples=[
            {
                "persona": "You are a high-ranking French military officer with decades of experience in strategic planning. You specialize in analyzing French geopolitical and economic interests worldwide, with a focus on defense, diplomacy, and trade security. Your documents must be precise and grounded in current international and regional dynamics, offering actionable insights for policymakers.",
                "subject": "France's strategic economic and military interests in West Africa amid rising regional competition and security challenges.",
            }
        ],
    )
    flush: bool = Field(
        default=False,
        description="If True, removes from the database all files and related processing results created during this operation.",
    )


class GenerateDocumentResponse(BaseModel):
    jobId: str = Field(
        ...,
        description="Identifier of the document assembly job.",
        examples=["a1b2c3d4-e5f6-7890-1234-567890abcdef"],
    )
    # taskId: Optional[str] = Field(
    #     default=None,
    #     description="Identifier of the Celery task id queueing job.",
    #     examples=["40b0c46c-68b9-4c64-81c4-c4f156754686"]
    # )


class DocumentResponse(BaseModel):
    """API response model for document with per-section citations and root-level sources.

    Refactored response structure that separates concerns:
    - sections: Content structure with per-section citations (clear relationship between content and sources)
    - sources: Root-level deduplicated mapping of source_id to SourceReference (single source of truth)
    - citations & references: Kept for backward compatibility (will be deprecated in future major version)

    Each section includes its own citations that reference sources by source_id in the root sources dict.
    Direct-assembly sections have citations=null. Sequential/recursive sections have populated citations.

    This structure allows API consumers to:
    - Access per-section citations directly within sections
    - Understand which assembly method was used (null citations = direct, populated = sequential)
    - Trace citations back to original source files via core_data_id, core_type, core_data_source
    - Cache and reference source documents without duplication

    Attributes:
        jobId (str): Unique identifier of the document assembly job
        status (documentStatus): Current status of the document (pending, processing, completed, failed)
        createdAt (datetime): Timestamp when the document was created
        fileIds (List[str]): IDs of source files used in document assembly
        sections (List[SectionResponse]): document sections with per-section citations
        sources (Optional[Dict[str, SourceReference]]): Root-level deduplicated sources with traceability fields
    """

    jobId: str = Field(
        ...,
        description="Unique identifier of the document assembly job",
        examples=["550e8400-e29b-41d4-a716-446655440000"],
    )
    # taskId: str | None = Field(
    #     default=None,
    #     description="Identifier of the Celery task id queueing job.",
    #     examples=["40b0c46c-68b9-4c64-81c4-c4f156754686"]
    # )
    status: DocumentStatus = Field(..., description="Current document status", examples=["completed"])
    createdAt: Optional[datetime] = Field(None, description="Timestamp when document was created")
    coreDataIds: List[str] = Field(
        ...,
        description="IDs of source files used in document assembly",
        examples=[["file-1", "file-2"]],
    )
    sections: List[SectionResponse] = Field(
        ..., description="document sections with per-section citations"
    )
    sources: Optional[Dict[str, SourceReference]] = Field(
        default=None,
        description="Root-level deduplicated sources with traceability fields (file_id)",
    )

    @staticmethod
    def _collect_all_sections(sections: List[SectionResponse]) -> List[SectionResponse]:
        """Recursively collect all sections including nested subsections.

        Args:
            sections: List of top-level SectionResponse objects

        Returns:
            Flat list of all sections including nested ones
        """
        all_sections = []
        for section in sections:
            all_sections.append(section)
            if section.sections:
                all_sections.extend(
                    DocumentResponse._collect_all_sections(section.sections)
                )
        return all_sections

    @field_validator("sources")
    @classmethod
    def validate_sources_completeness(
        cls, v: Optional[Dict[str, SourceReference]], info: ValidationInfo
    ) -> Optional[Dict[str, SourceReference]]:
        """Validate that all source_ids referenced in citations exist in sources dict.

        Recursively collects all sections (including nested subsections) to find
        citations and source references.

        Args:
            v: The sources dict
            info: Validation context containing other field values

        Returns:
            The validated sources dict

        Raises:
            ValueError: If any citation references a source_id not present in sources
        """
        # If no sources, validate and deduplicate from sections' references
        sections: List[SectionResponse] = info.data.get("sections", [])

        # Recursively collect ALL sections including nested subsections
        all_sections = cls._collect_all_sections(sections)

        new_source: Dict[str, SourceReference] = {}
        referenced_source_ids: set[str] = set()

        for section in all_sections:
            if section.citations:
                for citation_stack_group in section.citations:
                    if not citation_stack_group:
                        continue
                    for citation_stack in citation_stack_group:
                        # Add primary citation source_id
                        if citation_stack.primary_citation:
                            referenced_source_ids.add(
                                citation_stack.primary_citation.chunk_id
                            )

                        # Add supporting citation source_ids
                        if citation_stack.supporting_citations:
                            for supporting in citation_stack.supporting_citations:
                                referenced_source_ids.add(supporting.chunk_id)

            # Deduplicate references from section into root-level sources
            if section.references:
                for chunk_id, source_metadata in section.references.references.items():
                    # Skip if already deduplicated
                    if chunk_id not in new_source:
                        new_source[chunk_id] = SourceReference(
                            chunk_id=source_metadata.chunk_id,
                            file_id=source_metadata.file_id,
                            content=source_metadata.content,
                        )

        # Validate all referenced source_ids exist in sources
        missing_source_ids = referenced_source_ids - set(new_source.keys())

        if missing_source_ids:
            raise ValueError(
                f"Citations reference source_ids not found in sources dict: {missing_source_ids}. "
                + "All source_ids in citations must exist in root sources dict."
            )

        return new_source if new_source else v


class SectionResponse(BaseModel):
    """API response model for document section with per-section citations.

    This is the section representation used in API responses. Each section includes
    its own subset of citations that reference sources in the parent documentResponse.

    Attributes:
        title (str): The section title
        order (int): Position in the document (for ordering)
        template_section_id (str): Internal identifier used during processing
        generated_content (Optional[str]): The generated content text
        source_ids (List[List[str]]): IDs of source documents used (per-paragraph)
        sections (List[str]): Reserved for future extended structure
        job_id (Optional[str]): ID of the document assembly job
        status (SectionStatus): Current status of the section
        section_id (str): Unique identifier for this section
        citations (Optional[List[CitationStack]]): Per-section citations. None for direct-assembly sections,
            populated for sections using retrieved documents. Each citation references sources by source_id
            in the parent documentResponse.sources dict.
    """

    title: str = Field(..., description="The section title", examples=["Introduction"])
    order: int = Field(
        ..., description="Position of this section in the document", examples=[1]
    )
    template_section_id: str = Field(
        ..., description="Template section identifier", examples=["intro"]
    )
    generated_content: Optional[str] = Field(
        None,
        description="Generated content for this section",
        examples=["La guerre en Ukraine a cristallisé une mutation géostratégique sans précédent, transformant les équilibres militaires traditionnels en un réseau de tensions interconnectées où chaque région devient un amplificateur des rivalités mondiales..."],
    )
    source_ids: List[str] = Field(
        default_factory=list, description="Source document IDs used"
    )
    sections: Optional[List[SectionResponse]] = Field(
        None, description="Reserved for extended structure"
    )
    job_id: Optional[str] = Field(
        None,
        description="document assembly job ID",
        examples=["a1b2c3d4-e5f6-7890-1234-567890abcdef"],
    )
    status: SectionStatus = Field(
        default=SectionStatus.PENDING, description="Section processing status", examples=['completed']
    )
    section_id: str = Field(
        ...,
        description="Unique identifier for this section",
        examples=["a1b2c3d4-e5f6-7890-1234-567890abcdef"],
    )
    citations: Optional[List[List[CitationStack]]] = Field(
        None,
        description="Per-section citations (None for direct assembly, populated for sequential/recursive)",
    )
    references: Optional[SourceReferences] = Field(None, exclude=True)
    # TODO : Nested sections objects in key 'sections' :
    # - generated_content:
    # If the section has no children, it is not None, content is the string of the response
    # If the section has children, it is not None and content
    # -
    # mask unused elements from frontend (decorelate internal objects with data from datastore and external response API objects, e.i: order has no use in the frontend)

    @classmethod
    def from_section(cls, section: Section) -> "SectionResponse":
        """Create a SectionResponse from a Section model, including per-section citations.

        Args:
            section (Section): The internal Section model

        Returns:
            SectionResponse: API response model with per-section citations
        """
        section_responses = None
        if section.sections:
            section_responses = [
                SectionResponse.from_section(s) for s in section.sections
            ]

        return cls(
            title=section.title,
            order=section.order,
            template_section_id=section.name,
            generated_content=section.generated_content,
            source_ids=section.source_ids,
            sections=section_responses,
            job_id=section.job_id,
            status=section.status,
            section_id=section.section_id,
            citations=section.citations,
            references=section.source_references,
        )

SectionResponse.model_rebuild() # Helper to have correct swagger assembly


class ErrorResponse(BaseModel):
    error: str = Field(..., examples=["Missing required parameters."])
