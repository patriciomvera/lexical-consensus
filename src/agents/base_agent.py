"""
base_agent.py
-------------
Abstract base class for all agents in the lexical consensus experiment.

An agent is any entity that can:
  - Receive a visual stimulus (image) and a context (what it has learned so far)
  - Produce a label for that stimulus
  - Update its internal state based on consensus feedback

Both the human Tutor and the LLM Learner agents implement this interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import time


@dataclass
class LabelEvent:
    """A single labeling event — the atomic unit of the consensus ledger."""
    agent_id: str
    image_hash: str
    label: str
    confidence: float          # 0.0–1.0, agent's self-reported confidence
    round_number: int
    timestamp: float = field(default_factory=time.time)
    is_stable: Optional[bool] = None    # Set by consensus mechanism post-hoc
    notes: str = ""


@dataclass
class ConsensusSignal:
    """
    Feedback from the consensus mechanism to an agent.
    Agents receive only stability status of their OWN labels —
    not the labels assigned by other agents.
    This mirrors the principle of indirect social learning.
    """
    image_hash: str
    your_label: str
    is_stable: bool            # Did your label reach consensus threshold?
    is_discarded: bool         # Did your label fall below rejection threshold?
    round_number: int


class BaseAgent(ABC):
    """
    Abstract base class for all agents.

    Every agent has:
    - A unique ID
    - A local lexicon: the labels it currently believes are correct
    - A history of all its labeling decisions
    """

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.local_lexicon: dict[str, str] = {}   # image_hash -> label
        self.history: list[LabelEvent] = []
        self.round = 0

    @abstractmethod
    def receive_instruction(self, image_hash: str, description: str, label: str) -> None:
        """
        Receive a tutor instruction.

        Args:
            image_hash: Unique identifier of the image being described.
            description: Verbal/textual description of the image's properties.
            label:       The artificial label the tutor assigns to this image.
        """
        pass

    @abstractmethod
    def label_image(self, image_hash: str, image_description: str) -> LabelEvent:
        """
        Label a new image using only the vocabulary acquired so far.

        Args:
            image_hash:         Unique identifier of the image.
            image_description:  Visual properties of the image (no label hints).

        Returns:
            A LabelEvent with the agent's label and confidence.
        """
        pass

    @abstractmethod
    def receive_consensus_feedback(self, signal: ConsensusSignal) -> None:
        """
        Receive feedback from the consensus mechanism.
        The agent may update its local lexicon based on this signal.

        Args:
            signal: ConsensusSignal indicating stability of agent's label.
        """
        pass

    def update_lexicon(self, image_hash: str, label: str) -> None:
        """Update the local belief about the correct label for an image."""
        self.local_lexicon[image_hash] = label

    def log_event(self, event: LabelEvent) -> None:
        """Record a labeling event to the agent's local history."""
        self.history.append(event)

    def advance_round(self) -> None:
        """Move to the next interaction round."""
        self.round += 1

    def get_vocabulary(self) -> set[str]:
        """Return the set of distinct labels this agent currently uses."""
        return set(self.local_lexicon.values())

    def __repr__(self) -> str:
        return f"Agent(id={self.agent_id}, round={self.round}, vocab_size={len(self.get_vocabulary())})"
