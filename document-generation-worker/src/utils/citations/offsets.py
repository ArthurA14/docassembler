"""
Section offset calculation utilities for citation extraction.

Provides functions to calculate character offsets for sections/paragraphs
within formatted section content, enabling accurate citation span mapping.
"""

from typing import List

from document_common.utils.logger import Logger


def calculate_section_offsets(content: str, sections: List[str], num_sections: int) -> List[int]:
    """
    Calculate character offsets for each section in formatted content.

    The formatted content follows pattern: "### Title\\n<content>\\n### Title\\n<content>..."
    This function calculates offsets based on the actual structure of the formatted content.

    Args:
        content: Full formatted section content
        sections: List of paragraph content for each section
        num_sections: Expected number of sections

    Returns:
        List of character offsets (one per section)
    """
    logger = Logger().get_logger("section_offsets")

    assert num_sections == len(sections)

    if not content or num_sections <= 0:
        return [0] * max(1, num_sections)

    offsets: List[int] = []
    lines = content.split("\n")

    # Calculate offset for each section based on structure
    # Format: "### Title\\n<content>\\n### Title\\n<content>..."
    current_offset = 0

    for i, section in enumerate(sections):
        # Find the title line that precedes this section
        # We assume the title is the line before the section content
        if i == 0:
            # First section: find the first "###" line
            for j, line in enumerate(lines):
                if line.startswith("###"):
                    # The section content starts after the title line
                    # Calculate offset: sum of lengths of all lines before the section
                    current_offset = 0
                    for k in range(j + 1):  # Include the title line
                        current_offset += len(lines[k]) + 1  # +1 for newline
                    break
        else:
            # Subsequent sections: find the next "###" line after the previous section
            # We need to find where the previous section ended
            prev_section_end = current_offset + len(sections[i-1])
            # Find the next title line after the previous section
            for j, line in enumerate(lines):
                if line.startswith("###") and j > 0:  # Look for title lines after the first
                    # Calculate offset from start to this title line
                    current_offset = 0
                    for k in range(j + 1):  # Include the title line
                        current_offset += len(lines[k]) + 1  # +1 for newline
                    break

        # The section content starts after the title line
        offsets.append(current_offset)

        # Update current_offset for next iteration
        if i < len(sections) - 1:  # Not the last section
            current_offset += len(section) + 1  # +1 for newline after section

    return offsets

    # If we couldn't parse properly, use improved fallback
    if len(offsets) != num_sections:
        logger.warning(
            f"Offset calculation mismatch: found {len(offsets)} positions for "
            f"{num_sections} sections. Using improved fallback strategy."
        )

    return offsets


def adjust_span_offsets(offsets: List[int], section_offset: int) -> tuple[int, int]:
    """
    Adjust citation span offsets for paragraph-aware extraction.

    When extracting citations per paragraph, spans are relative to the paragraph.
    This function adjusts them to be relative to the full section content.

    Args:
        offsets: [start_offset, end_offset] relative to paragraph
        section_offset: Offset where this paragraph starts in full content

    Returns:
        Tuple of adjusted (start_offset, end_offset)
    """
    return (
        offsets[0] + section_offset,
        offsets[1] + section_offset,
    )
