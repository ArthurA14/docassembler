import yaml
from fastapi import APIRouter, HTTPException
from document_common.schemas.document_template import DocumentTemplate
from document_common.services.template_service import TemplateService
from document_common.utils.logger import Logger

from api_gateway.models.template import (
    TemplateItem,
    TemplateListResponse,
    TemplateResponse,
    TemplateUpsertRequest,
)

router = APIRouter(prefix="/templates", tags=["Template"])
logger = Logger().get_logger("api.routes.templates")

template_service = TemplateService()

@router.get(
    "",
    summary="Lists all templates",
    description="Return the list of all existing templates with their id and content.",
    status_code=200,
    response_description="List of all templates"
)
async def list_templates() -> TemplateListResponse:
    """
    Lists all available templates.

    This endpoint returns templates already loaded in memory during application startup.
    Templates are automatically loaded from storage on first app startup.
    If you need to reload templates from YAML files, restart the application.

    Returns:
        - A list of template objects containing templateId and template content.
    """
    # Get and return all templates
    templates_raw = template_service.list_all_templates()

    templates = [
        TemplateItem(templateId=t.name, template=t.content) for t in templates_raw
    ]

    return TemplateListResponse(templates=templates)

@router.put(
    "/{template_name}",
    summary="Create a template",
    description="Creates a new template. The template_name in the URL must match the template_name field in the YAML content.",
    responses={
        400: {"description": "Template name mismatch, invalid YAML, or content does not match documentTemplate schema."}
    },
)
async def upsert_template(template_name: str, payload: TemplateUpsertRequest) -> TemplateResponse:
    """
    Creates a template.

    Args:
        template_name: Unique name of the template
        payload: Template content (must be valid YAML matching documentTemplate schema)

    Returns:
        Confirmation message with template name

    Raises:
        400: If template name mismatch, invalid YAML, or content doesn't match schema
        500: If internal error occurs during upsert
    """
    try:
        # YAML validation
        data = yaml.safe_load(payload.content)

        if not isinstance(data, dict):
            raise HTTPException(status_code=400, detail="YAML content must be a mapping (dictionary)")

        # Template name consistency check
        yaml_template_name = data.get("template_name")
        if yaml_template_name != template_name:
            raise HTTPException(
                status_code=400,
                detail=f"Template name mismatch: URL has '{template_name}' but YAML has '{yaml_template_name}'",
            )

        # Pydantic schema validation against documentTemplate
        try:
            DocumentTemplate.model_validate(data)
        except Exception as pydantic_exc:
            raise HTTPException(
                status_code=400,
                detail=f"Template content does not match documentTemplate schema: {str(pydantic_exc)}",
            )

        # Upsert
        template_service.upsert_template(
            template_name=template_name, content=payload.content, description=payload.description, duplicate_rejection=True
        )

        return TemplateResponse(
            message=f"Template '{template_name}' saved successfully",
            template_name=template_name,
        )
    except HTTPException:
        # Re-raise HTTPException to preserve status code
        raise
    except yaml.YAMLError as e:
        raise HTTPException(status_code=400, detail=f"Invalid YAML format: {str(e)}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error upserting template '{template_name}': {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

# @router.delete(
#     "/{template_name}",
#     summary="Delete a template",
#     description="Deletes a template by name.",
#     responses={
#         404: {"description": "Error matching the template."},
#         500: {"description": "Internal error."},
#     },
# )
# async def delete_template(template_name: str) -> TemplateResponse:
#     """
#     Delete a template.

#     Args:
#         template_name: Name of the template to delete

#     Returns:
#         Confirmation message
#     """
#     try:
#         template_service.delete_template(template_name)
#         return TemplateResponse(
#             message = f"Template '{template_name}' deleted successfully",
#             template_name = template_name,
#         )
#     except ValueError as e:
#         raise HTTPException(status_code=404, detail=str(e))
#     except Exception as e:
#         logger.error(f"Error deleting template '{template_name}': {e}")
#         raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
