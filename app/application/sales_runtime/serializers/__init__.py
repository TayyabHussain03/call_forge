"""Per-contract snapshot serializers.

One serializer per domain contract. Envelope format carries contract key +
version so individual contracts evolve independently of the top-level
snapshot_version. Unsupported contract versions fail closed.
"""

from app.application.sales_runtime.serializers.base import (
    PayloadRequirement,
    SnapshotContractSerializer,
    build_envelope,
    parse_envelope,
)
from app.application.sales_runtime.serializers.bci import (
    BUSINESS_CONVERSATION_KEY,
    BusinessConversationSerializer,
)
from app.application.sales_runtime.serializers.language import (
    LANGUAGE_PROFILE_KEY,
    LanguageProfileSerializer,
)
from app.application.sales_runtime.serializers.qualification import (
    QUALIFICATION_KEY,
    QualificationSerializer,
)

__all__ = [
    "PayloadRequirement",
    "SnapshotContractSerializer",
    "build_envelope",
    "parse_envelope",
    "BUSINESS_CONVERSATION_KEY",
    "BusinessConversationSerializer",
    "LANGUAGE_PROFILE_KEY",
    "LanguageProfileSerializer",
    "QUALIFICATION_KEY",
    "QualificationSerializer",
]
