import yaml
from fastapi import APIRouter, HTTPException
from document_common.schemas.prompt_template import PromptDict
from document_common.services.prompt_service import PromptService
from document_common.utils.logger import Logger

from api_gateway.models.prompt import (
    PromptItem,
    PromptListResponse,
    PromptUpsertRequest,
    PromptUpsertResponse,
)

router = APIRouter(prefix="/prompts", tags=["Prompt"])
logger = Logger().get_logger("api.routes.prompts")
prompt_service = PromptService()

@router.get(
    "",
    summary="Lists all prompts",
    description="Return the list of all existing prompts with their metadata.",
    status_code=200,
    response_description="List of all prompts"
)
async def list_prompts() -> PromptListResponse:
    """
    Lists all available prompts.

    This endpoint returns prompts already loaded in memory during application startup.
    Prompts are automatically loaded from storage on first app startup.
    If you need to reload prompts from YAML files, restart the application.

    Returns:
        - A list of prompt objects containing promptId, type, mode, version and content.
    """
    # Don't force reload - prompts are already loaded at startup in documentManager.__init__()
    # Get and return all prompts
    prompts_list = prompt_service.list_all_prompts()
    prompts: list[PromptItem]= []
    for p in prompts_list:
        prompts.append(
            PromptItem(
                promptId=p["name"],
                type=p.get("type") or "unknown",
                mode=p.get("mode"),
                version=p.get("version") or 1,
                content=p.get("prompt") or "",
            )
        )

    return PromptListResponse(prompts=prompts)

@router.get(
    "/{prompt_name}",
    summary="Get a specific prompt",
    description="Retrieve a prompt by its name.",
    responses={
        404: {"description": "Prompt not found"},
    },
)
async def get_prompt(prompt_name: str) -> PromptItem:
    """
    Get a prompt by name.

    Args:
        prompt_name: Name of the prompt to retrieve

    Returns:
        Prompt object with metadata and content
    """
    prompt_data = prompt_service.get_prompt(prompt_name)

    if not prompt_data:
        raise HTTPException(status_code=404, detail=f"Prompt '{prompt_name}' not found")

    return PromptItem(
        promptId=prompt_data["name"],
        type=prompt_data.get("type") or "unknown",
        mode=prompt_data.get("mode"),
        version=prompt_data.get("version") or 1,
        content=prompt_data["content"],
    )

@router.put(
    "/{prompt_name}",
    summary="Creates a prompt",
    description="Creates a new prompt. The YAML content must include a version field.",
    responses={
        400: {"description": "Template name mismatch, invalid YAML or data format."},
        500: {"description": "Error upserting prompt"}
    },
)
async def upsert_prompt(prompt_name: str, payload: PromptUpsertRequest) -> PromptUpsertResponse:
    """
    Creates a prompt.

    Args:
        prompt_name: Unique name of the prompt (snake-case)
        payload: Prompt content (must include version field)

    Returns:
        Confirmation message with prompt name
    """
    try:
        # YAML validation
        data = yaml.safe_load(payload.content)

        # Check version field exists
        if "version" not in data:
            data["version"] = 1
        
        # Extract prompt name from YAML (first key after version)
        yaml_prompt_name = None
        for key in data.keys():
            if key != "version":
                yaml_prompt_name = key
                break

        # Prompt name consistency check (optional - could be different)
        if yaml_prompt_name and yaml_prompt_name != prompt_name:
            message = f"Prompt name mismatch: URL has '{prompt_name}' but YAML has '{yaml_prompt_name}'"
            logger.warning(message)
            raise HTTPException(status_code=400, detail=message)
        
        # Prompt Format Validation :
        PromptDict.model_validate(data[yaml_prompt_name])

        # Upsert
        prompt_service.upsert_prompt(
            prompt_name=yaml_prompt_name, content=payload.content, duplicate_rejection=True
        )

        return PromptUpsertResponse(
            message = f"Prompt '{prompt_name}' saved successfully",
            prompt_name = prompt_name,
            version = data.get("version"),
        )

    except HTTPException:
        # Re-raise HTTPException to preserve status code
        raise
    except yaml.YAMLError as e:
        raise HTTPException(status_code=400, detail=f"Invalid YAML format: {str(e)}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error upserting prompt '{prompt_name}': {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

# @router.delete(
#     "/{prompt_name}",
#     summary="Delete a prompt",
#     description="Deletes a prompt by name.",
#     responses={
#         404: {"description": "Prompt not found"},
#         500: {"description": "Error deleting prompt"}
#     },
# )
# async def delete_prompt(prompt_name: str) -> PromptDeleteResponse:
#     """
#     Delete a prompt.

#     Args:
#         prompt_name: Name of the prompt to delete

#     Returns:
#         Confirmation message
#     """
#     try:
#         deleted = prompt_service.delete_prompt(prompt_name)
#         if not deleted:
#             raise HTTPException(
#                 status_code=404, detail=f"Prompt '{prompt_name}' not found"
#             )

#         return PromptDeleteResponse(
#             message = f"Prompt '{prompt_name}' deleted successfully",
#             prompt_name = prompt_name,
#         )
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Error deleting prompt '{prompt_name}': {e}")
#         raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
