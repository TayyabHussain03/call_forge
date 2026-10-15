"""LanguageProfile serializer. Small, fully round-trippable."""

from __future__ import annotations

from app.application.sales_runtime.serializers.base import (
    PayloadRequirement,
    _coerce_enum,
    build_envelope,
    parse_envelope,
)
from app.conversation.understanding.contracts import (
    ConversationalRegister,
    LanguageProfile,
    LanguageScript,
)
from app.persistence.errors import InvalidPersistedPayloadError


LANGUAGE_PROFILE_KEY = "language_profile"
LANGUAGE_PROFILE_VERSION = 1


class LanguageProfileSerializer:
    @property
    def key(self) -> str:
        return LANGUAGE_PROFILE_KEY

    @property
    def version(self) -> int:
        return LANGUAGE_PROFILE_VERSION

    @property
    def requirement(self) -> PayloadRequirement:
        return PayloadRequirement.REQUIRED_FOR_ACTIVE_RECOVERY

    def serialize(self, profile: LanguageProfile) -> str:
        return build_envelope(self.key, self.version, {
            "primary_language": profile.primary_language,
            "secondary_language": profile.secondary_language,
            "mixed_language": profile.mixed_language,
            "script": profile.script.value,
            "conversational_register": profile.conversational_register.value,
        })

    def deserialize(self, envelope_json: str) -> LanguageProfile:
        payload = parse_envelope(envelope_json, self.key, (self.version,))
        try:
            return LanguageProfile(
                primary_language=payload["primary_language"],
                secondary_language=payload.get("secondary_language"),
                mixed_language=bool(payload.get("mixed_language", False)),
                script=_coerce_enum(
                    LanguageScript, payload["script"], "language.script",
                ),
                conversational_register=_coerce_enum(
                    ConversationalRegister,
                    payload["conversational_register"],
                    "language.register",
                ),
            )
        except (TypeError, KeyError, ValueError) as exc:
            raise InvalidPersistedPayloadError(
                f"LanguageProfile payload invalid: {exc}"
            ) from exc
