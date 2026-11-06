from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PreflightModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def parameters_digest(parameters: dict[str, Any]) -> str:
    canonical = json.dumps(
        parameters,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ConfirmationGrant(PreflightModel):
    subject_id: str = Field(min_length=1)
    tenant_id: str = Field(min_length=1)
    action: str = Field(min_length=1)
    resource_id: str = Field(min_length=1)
    parameters_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    resource_version: str = Field(min_length=1)
    authorization_version: str = Field(min_length=1)
    expires_at: datetime
    revoked_at: datetime | None = None

    @model_validator(mode="after")
    def timezone_required(self) -> ConfirmationGrant:
        if self.expires_at.utcoffset() is None or (
            self.revoked_at is not None and self.revoked_at.utcoffset() is None
        ):
            raise ValueError("Confirmation times must have a timezone")
        return self


class WriteAttempt(PreflightModel):
    subject_id: str = Field(min_length=1)
    tenant_id: str = Field(min_length=1)
    action: str = Field(min_length=1)
    resource_id: str = Field(min_length=1)
    parameters_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    resource_version: str = Field(min_length=1)
    executor: Literal["assistant", "approval_platform"]


class AuthorizationState(PreflightModel):
    subject_id: str = Field(min_length=1)
    tenant_id: str = Field(min_length=1)
    resource_id: str = Field(min_length=1)
    scopes: frozenset[str]
    version: str = Field(min_length=1)
    revoked: bool = False


class WritePolicy(PreflightModel):
    allowed_actions: frozenset[str]
    required_scope: str = Field(min_length=1)
    required_executor: Literal["assistant", "approval_platform"]
    platform_approval_required: bool = False


class PreflightResult(PreflightModel):
    can_execute: bool
    confirmation_valid: bool
    authorization_valid: bool
    reasons: list[str]


def check_write_preflight(
    grant: ConfirmationGrant | None,
    attempt: WriteAttempt,
    authorization: AuthorizationState,
    policy: WritePolicy,
    *,
    now: datetime,
    platform_approved: bool = False,
) -> PreflightResult:
    """Check trusted backend records immediately before an execution attempt."""
    if now.utcoffset() is None:
        raise ValueError("Preflight time must have a timezone")
    reasons: list[str] = []
    if grant is None:
        reasons.append("CONFIRMATION_MISSING")
    else:
        for field in (
            "subject_id",
            "tenant_id",
            "action",
            "resource_id",
            "parameters_digest",
            "resource_version",
        ):
            if getattr(grant, field) != getattr(attempt, field):
                reasons.append(f"CONFIRMATION_{field.upper()}_CHANGED")
        if grant.authorization_version != authorization.version:
            reasons.append("CONFIRMATION_AUTHORIZATION_CHANGED")
        if grant.revoked_at is not None or now >= grant.expires_at:
            reasons.append("CONFIRMATION_EXPIRED_OR_REVOKED")
    confirmation_valid = not reasons
    for field in ("subject_id", "tenant_id", "resource_id"):
        if getattr(authorization, field) != getattr(attempt, field):
            reasons.append(f"AUTHORIZATION_{field.upper()}_MISMATCH")
    if authorization.revoked:
        reasons.append("AUTHORIZATION_REVOKED")
    if policy.required_scope not in authorization.scopes:
        reasons.append("REQUIRED_SCOPE_MISSING")
    authorization_valid = not any(
        reason.startswith("AUTHORIZATION_") or reason == "REQUIRED_SCOPE_MISSING"
        for reason in reasons
    )
    if attempt.action not in policy.allowed_actions:
        reasons.append("ACTION_PROHIBITED")
    if attempt.executor != policy.required_executor:
        reasons.append("WRONG_EXECUTOR")
    if policy.platform_approval_required and not platform_approved:
        reasons.append("PLATFORM_APPROVAL_MISSING")
    return PreflightResult(
        can_execute=not reasons,
        confirmation_valid=confirmation_valid,
        authorization_valid=authorization_valid,
        reasons=reasons,
    )
