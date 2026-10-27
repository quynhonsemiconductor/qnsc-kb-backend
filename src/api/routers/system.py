from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from src.api.deps import get_current_user
from src.core.build_info import get_build_info
from src.models import User

router = APIRouter()


class BuildInfoResponse(BaseModel):
    service: str
    version: str
    commit: str | None
    commit_short: str | None
    environment: str
    started_at: str
    image_digest: str | None


@router.get("/version", response_model=BuildInfoResponse)
async def get_version(_: User = Depends(get_current_user)) -> Any:
    """Which build answers: the release number and the commit it was built from.

    Signed-in users only. `/health/*` is public and stays free of this: a commit hash is
    what someone looks up to find the vulnerabilities of that exact build.
    """
    return (await get_build_info()).public()
