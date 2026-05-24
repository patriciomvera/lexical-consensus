"""
ledger.py + voting.py (combined for Phase 1)
---------------------------------------------
The consensus mechanism — the heart of the distributed language emergence.

Inspired by distributed ledger principles:
  - Every label assignment is recorded immutably
  - Labels compete for dominance through weighted voting
  - Stable labels emerge through sustained majority agreement
  - Discarded labels are penalized in future rounds

This is NOT a blockchain implementation — it is a consensus dynamics
simulator that borrows the conceptual architecture.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional
import time

from ..agents.base_agent import LabelEvent, ConsensusSignal


# --- Configuration ---

CONVERGENCE_THRESHOLD = 0.70   # Fraction of agents that must agree for stability
REJECTION_THRESHOLD   = 0.20   # Below this fraction, label is discarded
STABILITY_WINDOW      = 10     # Number of consecutive rounds for confirmed stability


# --- Data structures ---

@dataclass
class LedgerEntry:
    """An immutable record of a single label assignment."""
    event: LabelEvent
    entry_id: int
    written_at: float = field(default_factory=time.time)


@dataclass
class LabelStatus:
    """Current consensus status for a specific (image, label) pair."""
    image_hash: str
    label: str
    vote_fraction: float        # Fraction of agents currently using this label
    is_stable: bool
    is_discarded: bool
    stable_since_round: Optional[int] = None
    consecutive_stable_rounds: int = 0


class ConsensusLedger:
    """
    Records all label assignments and computes consensus status per round.

    The ledger is the shared communication channel between agents.
    Agents do NOT read each other's labels directly — they only receive
    feedback about the stability of their own labels.
    """

    def __init__(self, n_agents: int):
        self.n_agents = n_agents
        self.entries: list[LedgerEntry] = []
        self._entry_counter = 0

        # Current consensus state: image_hash -> {label -> LabelStatus}
        self._consensus: dict[str, dict[str, LabelStatus]] = defaultdict(dict)

        # History of consensus states per round
        self.consensus_history: list[dict] = []

    def write(self, event: LabelEvent) -> LedgerEntry:
        """
        Record a label assignment event.
        This is the only way to add data to the ledger.
        """
        entry = LedgerEntry(
            event=event,
            entry_id=self._entry_counter,
        )
        self.entries.append(entry)
        self._entry_counter += 1
        return entry

    def compute_round_consensus(self, round_number: int) -> dict[str, list[LabelStatus]]:
        """
        After all agents have submitted labels for a round,
        compute the consensus status for each image.

        Returns: image_hash -> list of LabelStatus (one per label that received votes)
        """
        # Gather all non-feedback events from this round
        round_events = [
            e.event for e in self.entries
            if e.event.round_number == round_number
            and "consensus_feedback" not in e.event.notes
        ]

        # Count votes per (image, label) pair
        vote_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        agent_votes: dict[str, set] = defaultdict(set)   # Track which agents voted on each image

        for event in round_events:
            vote_counts[event.image_hash][event.label] += 1
            agent_votes[event.image_hash].add(event.agent_id)

        # Compute fractions and stability
        result: dict[str, list[LabelStatus]] = {}

        for image_hash, label_counts in vote_counts.items():
            n_voters = len(agent_votes[image_hash])
            statuses = []

            for label, count in label_counts.items():
                fraction = count / n_voters if n_voters > 0 else 0.0
                is_stable = fraction >= CONVERGENCE_THRESHOLD
                is_discarded = fraction < REJECTION_THRESHOLD

                # Track consecutive stability
                prev_status = self._consensus[image_hash].get(label)
                consecutive = 0
                stable_since = None

                if prev_status and is_stable:
                    consecutive = prev_status.consecutive_stable_rounds + 1
                    stable_since = prev_status.stable_since_round or round_number
                elif is_stable:
                    consecutive = 1
                    stable_since = round_number

                status = LabelStatus(
                    image_hash=image_hash,
                    label=label,
                    vote_fraction=fraction,
                    is_stable=is_stable,
                    is_discarded=is_discarded,
                    stable_since_round=stable_since,
                    consecutive_stable_rounds=consecutive,
                )
                statuses.append(status)
                self._consensus[image_hash][label] = status

            result[image_hash] = sorted(statuses, key=lambda s: s.vote_fraction, reverse=True)

        # Store snapshot
        self.consensus_history.append({
            "round": round_number,
            "consensus": {
                img: [s.__dict__ for s in statuses]
                for img, statuses in result.items()
            }
        })

        return result

    def generate_feedback(
        self,
        agent_id: str,
        round_number: int,
        consensus: dict[str, list[LabelStatus]]
    ) -> list[ConsensusSignal]:
        """
        Generate consensus feedback signals for a specific agent.

        The agent only learns whether ITS OWN labels are stable or discarded.
        It does NOT learn what labels other agents used.
        """
        signals = []

        # Get this agent's labels for the round
        agent_events = {
            e.event.image_hash: e.event
            for e in self.entries
            if e.event.agent_id == agent_id
            and e.event.round_number == round_number
            and "consensus_feedback" not in e.event.notes
        }

        for image_hash, event in agent_events.items():
            statuses = consensus.get(image_hash, [])
            my_status = next(
                (s for s in statuses if s.label == event.label),
                None
            )

            if my_status:
                signal = ConsensusSignal(
                    image_hash=image_hash,
                    your_label=event.label,
                    is_stable=my_status.is_stable,
                    is_discarded=my_status.is_discarded,
                    round_number=round_number,
                )
            else:
                # No status found — treat as discarded
                signal = ConsensusSignal(
                    image_hash=image_hash,
                    your_label=event.label,
                    is_stable=False,
                    is_discarded=True,
                    round_number=round_number,
                )
            signals.append(signal)

        return signals

    def get_stable_lexicon(self) -> dict[str, str]:
        """
        Return the current stable lexicon: image_hash -> dominant_label
        for all images where consensus has been reached.
        """
        stable = {}
        for image_hash, label_map in self._consensus.items():
            stable_labels = [
                s for s in label_map.values()
                if s.is_stable and s.consecutive_stable_rounds >= STABILITY_WINDOW
            ]
            if stable_labels:
                # Take the label with highest vote fraction
                best = max(stable_labels, key=lambda s: s.vote_fraction)
                stable[image_hash] = best.label
        return stable

    def convergence_summary(self, round_number: int) -> dict:
        """
        Summary statistics for the current round.
        Used for metrics and logging.
        """
        all_images = set(self._consensus.keys())
        stable_images = {
            img for img, label_map in self._consensus.items()
            if any(s.is_stable for s in label_map.values())
        }
        fully_stable_images = {
            img for img, label_map in self._consensus.items()
            if any(
                s.is_stable and s.consecutive_stable_rounds >= STABILITY_WINDOW
                for s in label_map.values()
            )
        }

        return {
            "round": round_number,
            "total_images": len(all_images),
            "images_with_any_consensus": len(stable_images),
            "images_fully_converged": len(fully_stable_images),
            "convergence_rate": len(fully_stable_images) / len(all_images) if all_images else 0.0,
            "stable_lexicon_size": len(self.get_stable_lexicon()),
        }
