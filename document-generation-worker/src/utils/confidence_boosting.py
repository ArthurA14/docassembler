from typing import List

from document_common.schemas.citations import Citation, SupportingCitation


def calculate_stack_confidence(
    primary_citation: Citation,
    supporting_citations: List[SupportingCitation],
    boost_factor: float = 0.05,
    max_boost: float = 0.25,
    quality_threshold: float = 0.7,
) -> float:
    """
    Calculate stack confidence with multi-source agreement boosting.

    Args:
        primary_citation: Primary citation with highest confidence
        supporting_citations: List of supporting citations
        boost_factor: Confidence boost per high-quality supporting citation
        max_boost: Maximum total boost allowed
        quality_threshold: Minimum confidence for a citation to contribute to boost

    Returns:
        Stack confidence score (0.0-1.0)
    """
    base_confidence = primary_citation.confidence

    if not supporting_citations:
        return base_confidence

    # Count high-quality supporting citations
    quality_supporters = [c for c in supporting_citations if c.confidence >= quality_threshold]

    # Calculate boost based on number of quality supporters
    boost = min(len(quality_supporters) * boost_factor, max_boost)

    # Apply boost and cap at 1.0
    stack_confidence = min(1.0, base_confidence + boost)

    return stack_confidence


def calculate_progressive_boost(
    primary_citation: Citation,
    supporting_citations: List[SupportingCitation],
    base_boost: float = 0.05,
    max_boost: float = 0.25,
) -> float:
    """
    Calculate stack confidence with progressive boosting.

    Progressive boosting gives higher weight to the first few supporting citations
    and diminishing returns for additional ones.

    Args:
        primary_citation: Primary citation
        supporting_citations: List of supporting citations
        base_boost: Base boost for first supporting citation
        max_boost: Maximum total boost

    Returns:
        Stack confidence score (0.0-1.0)
    """
    base_confidence = primary_citation.confidence

    if not supporting_citations:
        return base_confidence

    # Progressive boost: 5%, 4%, 3%, 2%, 1% for 1st, 2nd, 3rd, 4th, 5th
    boost = 0.0
    for idx in range(len(supporting_citations)):
        # Decrease boost factor for each additional citation
        factor = base_boost * max(0.2, 1.0 - (idx * 0.2))
        boost += factor

        if boost >= max_boost:
            boost = max_boost
            break

    stack_confidence = min(1.0, base_confidence + boost)
    return stack_confidence


def calculate_diversity_weighted_boost(
    primary_citation: Citation,
    supporting_citations: List[SupportingCitation],
    boost_factor: float = 0.05,
    max_boost: float = 0.25,
) -> float:
    """
    Calculate stack confidence with diversity-weighted boosting.

    Citations from diverse sources contribute more to confidence boost.

    Args:
        primary_citation: Primary citation
        supporting_citations: List of supporting citations
        boost_factor: Base boost per citation
        max_boost: Maximum total boost

    Returns:
        Stack confidence score (0.0-1.0)
    """
    base_confidence = primary_citation.confidence

    if not supporting_citations:
        return base_confidence

    # Count unique sources
    sources = {primary_citation.file_id}
    unique_supporters = []

    for citation in supporting_citations:
        if citation.file_id not in sources:
            sources.add(citation.file_id)
            unique_supporters.append(citation)

    # Boost only from unique sources
    boost = min(len(unique_supporters) * boost_factor, max_boost)
    stack_confidence = min(1.0, base_confidence + boost)

    return stack_confidence


def calculate_quality_weighted_boost(
    primary_citation: Citation,
    supporting_citations: List[SupportingCitation],
    boost_factor: float = 0.05,
    max_boost: float = 0.25,
) -> float:
    """
    Calculate stack confidence with quality-weighted boosting.

    Higher confidence supporting citations contribute more to the boost.

    Args:
        primary_citation: Primary citation
        supporting_citations: List of supporting citations
        boost_factor: Base boost per citation
        max_boost: Maximum total boost

    Returns:
        Stack confidence score (0.0-1.0)
    """
    base_confidence = primary_citation.confidence

    if not supporting_citations:
        return base_confidence

    # Weight boost by supporting citation confidence
    boost = 0.0
    for citation in supporting_citations:
        weighted_boost = boost_factor * citation.confidence
        boost += weighted_boost

        if boost >= max_boost:
            boost = max_boost
            break

    stack_confidence = min(1.0, base_confidence + boost)
    return stack_confidence


def assess_stack_quality(
    primary_citation: Citation,
    supporting_citations: List[SupportingCitation],
    min_confidence: float = 0.5,
    diversity_threshold: int = 2,
) -> dict:
    """
    Assess the quality of a citation stack.

    Args:
        primary_citation: Primary citation
        supporting_citations: List of supporting citations
        min_confidence: Minimum acceptable confidence
        diversity_threshold: Minimum number of unique sources for high quality

    Returns:
        Dict with quality assessment metrics
    """
    # Count unique sources
    sources = {primary_citation.file_id}
    for citation in supporting_citations:
        sources.add(citation.file_id)

    num_unique_sources = len(sources)

    # Calculate average confidence
    all_citations = [primary_citation] + supporting_citations
    avg_confidence = sum(c.confidence for c in all_citations) / len(all_citations)

    # Assess quality flags
    has_low_confidence = primary_citation.confidence < min_confidence
    lacks_diversity = num_unique_sources < diversity_threshold
    weak_support = len(supporting_citations) == 0

    # Overall quality score (0-1)
    quality_score = 0.0

    # Primary confidence contributes 50%
    quality_score += 0.5 * primary_citation.confidence

    # Source diversity contributes 25%
    diversity_score = min(1.0, num_unique_sources / 3)  # 3+ sources = full score
    quality_score += 0.25 * diversity_score

    # Supporting strength contributes 25%
    support_score = min(1.0, len(supporting_citations) / 3)  # 3+ supporters = full score
    quality_score += 0.25 * support_score

    return {
        "quality_score": quality_score,
        "num_sources": len(all_citations),
        "num_unique_sources": num_unique_sources,
        "avg_confidence": avg_confidence,
        "primary_confidence": primary_citation.confidence,
        "num_supporting": len(supporting_citations),
        "has_low_confidence": has_low_confidence,
        "lacks_diversity": lacks_diversity,
        "weak_support": weak_support,
        "is_high_quality": quality_score >= 0.75 and not has_low_confidence,
    }


def calculate_confidence_with_method(
    primary_citation: Citation,
    supporting_citations: List[SupportingCitation],
    method: str = "standard",
    boost_factor: float = 0.05,
    max_boost: float = 0.25,
    quality_threshold: float = 0.7,
) -> float:
    """
    Calculate stack confidence using specified method.

    Args:
        primary_citation: Primary citation
        supporting_citations: List of supporting citations
        method: Boosting method (standard, progressive, diversity, quality)
        boost_factor: Base boost factor
        max_boost: Maximum boost
        quality_threshold: Quality threshold for standard method

    Returns:
        Stack confidence score
    """
    method = method.lower()

    if method == "progressive":
        return calculate_progressive_boost(primary_citation, supporting_citations, boost_factor, max_boost)
    elif method == "diversity":
        return calculate_diversity_weighted_boost(primary_citation, supporting_citations, boost_factor, max_boost)
    elif method == "quality":
        return calculate_quality_weighted_boost(primary_citation, supporting_citations, boost_factor, max_boost)
    else:  # standard or default
        return calculate_stack_confidence(
            primary_citation, supporting_citations, boost_factor, max_boost, quality_threshold
        )
