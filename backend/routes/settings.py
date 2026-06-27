"""User settings endpoints — capture/refine, generation defaults, and app settings."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from .. import config, models
from ..database import get_db
from ..services import settings as settings_service

router = APIRouter(prefix="/settings", tags=["settings"])


@router.get("", response_model=models.AppSettings)
async def get_app_settings():
    """Return current application settings (persisted in data/settings.json)."""
    return models.AppSettings(**config.load_app_settings())


@router.patch("", response_model=models.AppSettings)
async def update_app_settings(update: models.AppSettingsUpdate):
    """Partially update application settings.

    Only provided fields are written; the rest stay at their current value.
    The merged result is validated, persisted, and returned.
    """
    data = config.load_app_settings()
    if data == {} and config.get_settings_path().exists():
        raise HTTPException(status_code=500, detail="Failed to read settings")

    patch = update.model_dump(exclude_none=True)
    data.update(patch)
    validated = models.AppSettings(**data)
    config.save_app_settings(validated.model_dump())
    return validated


@router.get("/captures", response_model=models.CaptureSettingsResponse)
async def get_capture_settings_endpoint(db: Session = Depends(get_db)):
    return settings_service.get_capture_settings(db)


@router.put("/captures", response_model=models.CaptureSettingsResponse)
async def update_capture_settings_endpoint(
    patch: models.CaptureSettingsUpdate,
    db: Session = Depends(get_db),
):
    return settings_service.update_capture_settings(db, patch.model_dump(exclude_unset=True))


@router.get("/generation", response_model=models.GenerationSettingsResponse)
async def get_generation_settings_endpoint(db: Session = Depends(get_db)):
    return settings_service.get_generation_settings(db)


@router.put("/generation", response_model=models.GenerationSettingsResponse)
async def update_generation_settings_endpoint(
    patch: models.GenerationSettingsUpdate,
    db: Session = Depends(get_db),
):
    return settings_service.update_generation_settings(db, patch.model_dump(exclude_unset=True))
