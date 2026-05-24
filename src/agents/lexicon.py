"""
lexicon.py
----------
Capa Léxica — el vocabulario emergente del agente.

Esta capa es lo que el experimento construye.
Empieza completamente vacía. Se llena durante el experimento
a través de instrucción del tutor y retroalimentación de consenso.

Implementa las dos condiciones del experimento:

  CONDICIÓN 1 — Denominación (Naming):
    percepción visual → etiqueta
    "El tutor dice 'vorpal', el niño aprende que ese cluster se llama vorpal"

  CONDICIÓN 2 — Reconocimiento inverso (Grounding):
    etiqueta → representación visual
    "El niño escucha 'vorpal' y activa la representación interna del objeto"

Esta segunda condición es la que distingue comprensión de memorización:
si el agente solo memorizó la asociación imagen→etiqueta, falla en imágenes
nuevas. Si genuinamente construyó una representación del cluster, la generaliza.

Implementación: centroide por etiqueta en el espacio de embeddings DINOv2.
Cada vez que el agente ve una nueva instancia de "vorpal", actualiza el centroide.
La clasificación de imágenes nuevas usa distancia al centroide más cercano.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class LexiconEntry:
    """
    Una entrada del léxico: la representación interna de una etiqueta.

    Contiene todas las instancias vistas de esa categoría y el centroide
    que las resume — la "imagen mental" del concepto.
    """
    label: str
    embeddings: list[np.ndarray] = field(default_factory=list)
    centroid: Optional[np.ndarray] = None
    confidence: float = 0.0        # Coherencia interna del cluster (0–1)
    round_created: int = 0
    round_last_updated: int = 0
    n_consensus_confirmations: int = 0   # Veces que el consenso confirmó esta etiqueta
    n_consensus_rejections: int = 0      # Veces que el consenso rechazó esta etiqueta

    def add_embedding(self, embedding: np.ndarray, round_number: int) -> None:
        """Agregar una nueva instancia al cluster y actualizar el centroide."""
        self.embeddings.append(embedding.copy())
        self.centroid = np.mean(self.embeddings, axis=0)
        # Normalizar centroide
        norm = np.linalg.norm(self.centroid)
        if norm > 0:
            self.centroid = self.centroid / norm
        self.confidence = self._compute_cohesion()
        self.round_last_updated = round_number

    def _compute_cohesion(self) -> float:
        """
        Coherencia interna del cluster: similitud promedio de todas las
        instancias con el centroide. Alta cohesión = concepto bien definido.
        """
        if len(self.embeddings) < 2:
            return 1.0
        similarities = [
            float(np.dot(emb / np.linalg.norm(emb), self.centroid))
            for emb in self.embeddings
            if np.linalg.norm(emb) > 0
        ]
        return float(np.mean(similarities)) if similarities else 0.0

    def distance_to(self, embedding: np.ndarray) -> float:
        """
        Distancia coseno entre un embedding y el centroide de esta entrada.
        0 = idéntico, 2 = opuesto.
        """
        if self.centroid is None:
            return float("inf")
        sim = float(np.dot(embedding, self.centroid))
        return 1.0 - sim   # Convertir similitud a distancia

    def __repr__(self) -> str:
        return (
            f"LexiconEntry(label='{self.label}', "
            f"n={len(self.embeddings)}, "
            f"cohesion={self.confidence:.2f}, "
            f"confirmations={self.n_consensus_confirmations})"
        )


class LexiconLayer:
    """
    El léxico aprendido del agente: mapeo bidireccional entre
    etiquetas artificiales y regiones del espacio de embeddings.

    Empieza completamente vacío.
    Se construye únicamente a través de:
      1. Instrucción del tutor (grounding inicial)
      2. Auto-actualización al ver nuevas instancias
      3. Retroalimentación del mecanismo de consenso
    """

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self._entries: dict[str, LexiconEntry] = {}   # label -> LexiconEntry

    # ------------------------------------------------------------------
    # CONDICIÓN 1: Percepción → Etiqueta (Naming)
    # ------------------------------------------------------------------

    def assign(self, embedding: np.ndarray, label: str, round_number: int) -> None:
        """
        Asignar una etiqueta a un embedding.
        Si la etiqueta no existe, crear una nueva entrada.
        Si ya existe, agregar este embedding al cluster.
        """
        if label not in self._entries:
            entry = LexiconEntry(label=label, round_created=round_number)
            self._entries[label] = entry
        self._entries[label].add_embedding(embedding, round_number)

    def classify(self, embedding: np.ndarray) -> tuple[str, float]:
        """
        CONDICIÓN 1: Dado un embedding visual, retornar la etiqueta más probable
        y la confianza de esa clasificación.

        Usa distancia al centroide más cercano en el espacio DINOv2.

        Returns:
            (etiqueta, confianza) donde confianza ∈ [0, 1]
        """
        if not self._entries:
            return "UNKNOWN", 0.0

        distances = {
            label: entry.distance_to(embedding)
            for label, entry in self._entries.items()
            if entry.centroid is not None
        }

        if not distances:
            return "UNKNOWN", 0.0

        best_label = min(distances, key=distances.get)
        best_distance = distances[best_label]

        # Convertir distancia a confianza
        # distancia 0 → confianza 1.0, distancia 1 → confianza 0.0
        confidence = max(0.0, 1.0 - best_distance)

        # Penalizar si hay ambigüedad (segunda opción muy cercana)
        if len(distances) > 1:
            sorted_dists = sorted(distances.values())
            margin = sorted_dists[1] - sorted_dists[0]
            if margin < 0.1:   # Margen pequeño = ambigüedad
                confidence *= margin / 0.1

        return best_label, confidence

    # ------------------------------------------------------------------
    # CONDICIÓN 2: Etiqueta → Representación Visual (Grounding inverso)
    # ------------------------------------------------------------------

    def ground(self, label: str) -> Optional[np.ndarray]:
        """
        CONDICIÓN 2: Dado un string de etiqueta, recuperar la representación
        visual interna (centroide del cluster).

        Si el agente genuinamente aprendió el concepto, este centroide
        debería permitirle clasificar nuevas instancias correctamente.

        Returns:
            Vector centroide del cluster, o None si la etiqueta no existe.
        """
        entry = self._entries.get(label)
        if entry is None or entry.centroid is None:
            return None
        return entry.centroid.copy()

    def recognize_from_label(
        self,
        label: str,
        candidate_embeddings: dict[str, np.ndarray]
    ) -> tuple[Optional[str], float]:
        """
        CONDICIÓN 2 — test completo: dado solo la etiqueta "vorpal",
        ¿cuál de estas imágenes candidatas ES un vorpal?

        Esto testea el grounding inverso: la etiqueta activa la representación
        y esa representación permite clasificar imágenes nuevas.

        Args:
            label:                La etiqueta artificial a reconocer.
            candidate_embeddings: image_hash -> embedding de imágenes candidatas.

        Returns:
            (image_hash del mejor match, confianza)
        """
        centroid = self.ground(label)
        if centroid is None:
            return None, 0.0

        best_hash = None
        best_similarity = -1.0

        for img_hash, embedding in candidate_embeddings.items():
            sim = float(np.dot(embedding, centroid))
            if sim > best_similarity:
                best_similarity = sim
                best_hash = img_hash

        return best_hash, max(0.0, best_similarity)

    # ------------------------------------------------------------------
    # Actualización por consenso
    # ------------------------------------------------------------------

    def reinforce(self, label: str) -> None:
        """El consenso confirmó esta etiqueta — aumentar confianza."""
        if label in self._entries:
            self._entries[label].n_consensus_confirmations += 1

    def penalize(self, label: str) -> None:
        """
        El consenso rechazó esta etiqueta.
        Si acumula demasiados rechazos, eliminar del léxico.
        El agente tendrá que re-aprender esta categoría.
        """
        if label in self._entries:
            self._entries[label].n_consensus_rejections += 1
            # Si los rechazos superan 3x las confirmaciones, olvidar
            entry = self._entries[label]
            if (entry.n_consensus_rejections > 2 and
                    entry.n_consensus_rejections > entry.n_consensus_confirmations * 3):
                del self._entries[label]

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------

    def get_labels(self) -> list[str]:
        """Retornar todas las etiquetas conocidas."""
        return list(self._entries.keys())

    def get_entry(self, label: str) -> Optional[LexiconEntry]:
        return self._entries.get(label)

    def vocabulary_size(self) -> int:
        return len(self._entries)

    def is_empty(self) -> bool:
        return len(self._entries) == 0

    def summary(self) -> dict:
        """Resumen del estado actual del léxico — para logging y métricas."""
        return {
            "agent_id": self.agent_id,
            "vocabulary_size": self.vocabulary_size(),
            "entries": {
                label: {
                    "n_instances": len(entry.embeddings),
                    "cohesion": round(entry.confidence, 3),
                    "confirmations": entry.n_consensus_confirmations,
                    "rejections": entry.n_consensus_rejections,
                    "round_created": entry.round_created,
                }
                for label, entry in self._entries.items()
            }
        }

    def __repr__(self) -> str:
        labels = list(self._entries.keys())
        return f"LexiconLayer(agent={self.agent_id}, labels={labels})"
