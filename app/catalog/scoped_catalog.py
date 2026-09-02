"""Campaign-scoped catalog view.

Ek campaign ke liye catalog ko scope karta hai — sirf woh services jo us campaign
mein authorized hain. Ye view ClaimValidator aur (future) selector ka input hai.

DESIGN: scoping deterministic hai — campaign ki authorized list ke bahar koi
service exist hi nahi karta is view mein. Isse authorization boundary structural
ban jaati hai.
"""

from __future__ import annotations

from app.catalog.models import Service, ServiceCatalog
from app.core.exceptions import ConfigurationError


class ScopedCatalog:
    """A read-only view of the catalog scoped to one campaign.

    Attributes:
        campaign_id: The campaign this view is scoped to.
    """

    def __init__(self, catalog: ServiceCatalog, campaign_id: str) -> None:
        """Scope a catalog to a campaign.

        Args:
            catalog: The full validated catalog.
            campaign_id: The campaign to scope to.

        Raises:
            ConfigurationError: Agar campaign defined nahi.
        """
        campaign = catalog.campaigns.get(campaign_id)
        if campaign is None:
            raise ConfigurationError(f"Unknown campaign: {campaign_id!r}")
        self._campaign_id = campaign_id
        # Sirf authorized services is view mein.
        self._services: dict[str, Service] = {
            sid: catalog.services[sid] for sid in campaign.service_ids
        }

    @property
    def campaign_id(self) -> str:
        """Return the scoped campaign id.

        Returns:
            str: The campaign id.
        """
        return self._campaign_id

    def is_service_authorized(self, service_id: str) -> bool:
        """Whether a service is authorized in this campaign.

        Args:
            service_id: The service id to check.

        Returns:
            bool: True if the service is in this campaign's scope.
        """
        return service_id in self._services

    def get_service(self, service_id: str) -> Service | None:
        """Return an authorized service, or None if not in scope.

        Args:
            service_id: The service id.

        Returns:
            Service | None: The service, or None.
        """
        return self._services.get(service_id)

    def authorized_service_ids(self) -> frozenset[str]:
        """Return all authorized service ids for this campaign.

        Returns:
            frozenset[str]: Authorized service ids.
        """
        return frozenset(self._services.keys())

    def applicable_services(self, signals: frozenset[str]) -> tuple[Service, ...]:
        """Return authorized services whose applicability signals match.

        ELIGIBILITY only (guide refinement): ye batata hai kaunse services is
        context signal ke liye ELIGIBLE hain — relevance NAHI ("client interested
        hai" future selector ka concern). Ek service applicable hai agar uske
        `applicable_when` mein koi bhi diya gaya signal ho.

        Args:
            signals: Context signals (e.g. {"existing_website"}).

        Returns:
            tuple[Service, ...]: Eligible authorized services.
        """
        return tuple(
            svc
            for svc in self._services.values()
            if svc.applicable_when & signals
        )