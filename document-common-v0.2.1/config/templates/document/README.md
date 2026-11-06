# document assembly Methods Documentation

This document describes the different methods available for generating document sections in the YAML template system.

## Template Structure Overview

Each YAML template follows this basic structure:
```yaml
template_name: [template_identifier]
description: [template_description]
document:
  - section:
      method: [assembly_method]
      title: "[Section Title]"
      name: [section_identifier]
      prompt: [prompt_identifier]
      using: [dependencies]  # Optional: list of other sections this depends on
      num_lines:
        min: [minimum_lines]
        max: [maximum_lines]
      method_parameters: {}  # Optional: method-specific parameters
```

## assembly Methods

### 1. Direct Method

The `direct` method is the simplest assembly approach. It takes content from other sections and feeds them to an LLM with a specific prompt.

#### Configuration Parameters
- `method`: `direct`
- `prompt`: Identifier of the prompt to use for assembly
- `using`: List of other section names that this section will use
- `num_lines`: Minimum and maximum line count for the generated section

#### How It Works
1. The system collects content from all sections listed in the `using` parameter
2. This content is provided to the LLM along with the specified prompt
3. The LLM generates the section content based on the input

#### Example Use Cases
- Introduction sections that summarize the main content
- Conclusion sections that synthesize information from multiple sections
- Context sections that provide background based on main content

#### Example Configuration
```yaml
- section:
    method: direct
    title: "Introduction"
    name: intro
    prompt: t1_introduction
    using: [main, context]
    num_lines:
      min: 10
      max: 15
```

### 2. Narrative Method

The `narrative` method is more complex. It's designed for generating detailed, structured content with multiple sub-sections.

#### Configuration Parameters
- `method`: `narrative`
- `prompt`: Identifier of the prompt used to generate the main plan/outline
- `method_parameters`: Object containing narrative-specific parameters
- `num_lines`: Minimum and maximum line count for the generated section

#### Method Parameters
- `summaries_top_k`: Number of refined summaries chunks used to gain an overview of the subject IF a subject is given, if not we try to get ALL summaries available (default: 15)
- `originals_top_k`: Number of original chunks used to generate each sub-section (default: 8)
- `entities_top_k`: Number of entities retrieved for each sub-section (default: 10)
- `paragraphs_count`: Number of sub-sections to generate (default: 3)
- `auto_title`: Boolean - whether LLM should generate titles automatically (default: true)
- `sequential`: Boolean - assembly approach (see below)
- `restructure`: Boolean - assembly approach (see below)
- `prompt_paragraph`: Prompt identifier for non-sequential paragraph assembly
- `prompt_sequential_paragraph`: Prompt identifier for sequential paragraph assembly
- `prompt_restructure`: Prompt identifier for final restructuring
- `citations`: Boolean - whether to include citations (default: true)

#### assembly Approaches

The narrative method supports two different assembly approaches controlled by the `sequential` parameter:

- **Non-Sequential Mode** (`sequential: false`)
1. Each sub-section is generated independently
2. All generated sub-sections are collected
3. A final restructuring prompt reorganizes and optimizes the entire section

- **Sequential Mode** (`sequential: true`)
1. Sub-sections are generated one after another
2. Each assembly receives previously generated content as context
3. The LLM is asked to maintain consistency and avoid repetition
4. A final restructuring prompt reorganizes and optimizes the entire section

Based on vibe tests (true eval will be coming) we find that both seems to yield the same results.

- **Restructure Mode** (`restructure: true`)

The `restructure` parameter controls whether the system performs an additional LLM pass after all paragraphs have been generated.

When `restructure: true`, the previously generated paragraphs are:
1. Collected together as a single block of content
2. Passed through a dedicated restructuring prompt (`prompt_restructure`)
3. Rewritten to improve:
   - Global coherence
   - Logical flow between paragraphs
   - Redundancy removal

In short, restructure is a refinement pass that optimizes the section as a whole rather than treating paragraphs independently.
To note : The restructuring step is model-capacity sensitive.

- **Large models**
  - Generally improve clarity and structure
  - Respect paragraph counts and section boundaries
  - Produce more polished, publication-ready output

- **Small models**
  - Tend to:
    - Merge or split paragraphs incorrectly
    - Ignore requested paragraph counts
  - This can degrade structure instead of improving it

Because of this behavior, `restructure` does **not work well with small models**. For small models, it is better to have sequential assembly an disable the restructuration.

#### How It Works
1. System retrieves relevant content based on `summaries_top_k`, `originals_top_k`, and `entities_top_k`
2. Generates an outline using the main prompt
3. Depending on `sequential` mode:
   - Non-sequential: Generates each paragraph independently using `prompt_paragraph`
   - Sequential: Generates paragraphs sequentially using `prompt_sequential_paragraph`
4. Applies final restructuring with `prompt_restructure`
5. Optionally adds citations if `citations: true`

#### Example Configuration
```yaml
- section:
    method: narrative
    title: "Main Narrative"
    name: main
    prompt: document_outline_with_retrieval
    method_parameters:
      summaries_top_k: 15
      originals_top_k: 8
      entities_top_k: 10
      paragraphs_count: 3
      auto_title: true
      sequential: false
      prompt_paragraph: paragraph_generator
      prompt_sequential_paragraph: sequential_paragraph_generator
      prompt_restructure: restructure_paragraphs
      citations: true
    num_lines:
      min: 30
      max: 50
```

## Choosing the Right Method

Use the **direct method** when:
- You need simple synthesis of existing content
- The section is relatively straightforward
- You want to generate introductions, conclusions, or summaries

Use the **narrative method** when:
- You need detailed, structured analysis
- The content requires multiple sub-sections
- You want sophisticated content assembly with entity retrieval
- You need citations and references
- The content benefits from either sequential or parallel assembly approaches
