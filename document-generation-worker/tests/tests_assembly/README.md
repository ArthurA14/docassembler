# Tests document assembly (schéma post /generate-document)

Ces tests couvrent les **composants bleus** du schéma de génération de rapports (API → documentManager → FileHandler → PromptProvider/TemplateService → document Factory → SectionGeneratorManager → SectionGenerator → documentWriter).

| Fichier | Composants | Rôle |
|--------|------------|------|
| `test_document_writer.py` | **documentWriter** | update_document / update_document_section (écriture en base) ; get_document, get_document_sections, get_document_status, get_document_section (lecture). |
| `test_document_factory.py` | **documentFactory**, **SectionGeneratorManager** | Validation du template (noms uniques, refs `using`, pas de cycle, une entrée) ; chargement depuis Path/str ; validate_document_availability (critical/optional) ; SectionGeneratorManager exige une seule section « entrée ». |
| `test_section_generator.py` | **SectionGenerator**, **DirectSectionGenerator** | Registre (from_code, get_document_requirements) ; document_writer sans document lève ; DirectSectionGenerator.run avec mocks (contenu et statut mis à jour). |
| `test_prompt_provider.py` | **PromptProvider** | get_prompt (inconnu lève) ; add_prompt_from_str / add_prompt ; overwrite. |
| `test_prompt_service.py` | **PromptService** | get_prompt, upsert_prompt, list_all_prompts, get_latest_content (délégation au storage mocké). |
| `test_template_service.py` | **TemplateService** | get_template, upsert_template, list_all_templates, template_exists (délégation au storage mocké). |

**Note** : documentManager (orchestration, create_new_document) est déjà couvert dans `tests_importation/test_document_manager.py`. Les stores, LLM et embedder sont mockés pour garder des tests unitaires rapides.
