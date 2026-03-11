from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class MarketToken:
    token_id: str
    outcome: str  # "Yes" or "No"
    price: Optional[float] = None


@dataclass
class Market:
    condition_id: str
    question: str
    slug: str
    tokens: list[MarketToken]
    end_date: str
    volume_24h: float
    liquidity: float
    neg_risk: bool
    tick_size: float
    active: bool = True
    tags: list[str] = field(default_factory=list)
    event_slug: str = ""

    @property
    def yes_token_id(self) -> str:
        for t in self.tokens:
            if t.outcome.lower() == "yes":
                return t.token_id
        return self.tokens[0].token_id

    @property
    def no_token_id(self) -> str:
        for t in self.tokens:
            if t.outcome.lower() == "no":
                return t.token_id
        return self.tokens[-1].token_id

    @property
    def all_token_ids(self) -> list[str]:
        return [t.token_id for t in self.tokens]


@dataclass
class Event:
    event_id: str
    title: str
    slug: str
    markets: list[Market] = field(default_factory=list)
