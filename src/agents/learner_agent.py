"""
learner_agent.py  (v2 — arquitectura de tres capas)
----------------------------------------------------
El agente aprendiz rediseñado.

Arquitectura inspirada directamente en el proceso descrito en Vera et al.:

  "El niño reconoce el objeto y los adultos lo ayudan asignando la etiqueta,
  y esa etiqueta tiene un consenso en una comunidad determinada y ahí se
  forma la palabra. Luego la 'magia' ocurre de vuelta: cuando esa etiqueta
  es escuchada por el niño, este tiene la representación multimodal de la
  etiqueta dentro de su cerebro."
                                        — Anonymous, 2024

TRES CAPAS:

  CAPA 1 — Percepción (DINOv2, congelada)
    imagen → vector [384 dims]
    Idéntica en todos los agentes. Sin etiquetas. No cambia.
    Equivalente: corteza visual compartida.

  CAPA 2 — Léxico (aprendido durante el experimento)
    vector ↔ etiqueta artificial
    Empieza vacío. Se construye por instrucción + consenso.
    Bidireccional: naming (v→e) y grounding inverso (e→v).
    Equivalente: el vocabulario emergente.

  CAPA 3 — Consenso (externo, en ledger.py)
    Las etiquetas compiten entre agentes.
    Solo sobreviven las que alcanzan acuerdo mayoritario.
    Equivalente: la comunidad lingüística.

El agente NO usa LLMs para etiquetar imágenes.
La clasificación es puramente geométrica en el espacio DINOv2.
Esto elimina el problema de contaminación por pre-entrenamiento lingüístico.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union
from pathlib import Path

import numpy as np
from PIL import Image

from .base_agent import BaseAgent, LabelEvent, ConsensusSignal
from .perception import PerceptionLayer
from .lexicon import LexiconLayer


# Umbral mínimo de confianza para emitir una etiqueta
# Por debajo de esto, el agente emite "UNCERTAIN" en lugar de adivinar
CONFIDENCE_THRESHOLD = 0.30


@dataclass
class GroundingTestResult:
    """
    Resultado del test de Condición 2 (grounding inverso).

    Documenta si el agente, dado solo la etiqueta, puede identificar
    correctamente una imagen nueva del mismo concepto.
    """
    agent_id: str
    label: str
    selected_image_hash: Optional[str]
    correct_image_hash: str
    is_correct: bool
    confidence: float
    round_number: int


class LearnerAgent(BaseAgent):
    """
    Agente aprendiz con arquitectura de tres capas.

    Cada instancia de LearnerAgent tiene:
    - Su propia PerceptionLayer (DINOv2, idéntica a todos, congelada)
    - Su propio LexiconLayer (construido durante el experimento, único por agente)

    La PerceptionLayer puede ser compartida entre agentes para eficiencia,
    ya que es idéntica y solo-lectura. Ver factory method create_agents().
    """

    def __init__(
        self,
        agent_id: str,
        perception: Optional[PerceptionLayer] = None,
    ):
        super().__init__(agent_id)

        # Capa 1: Percepción — puede ser compartida (singleton) entre agentes
        self.perception = perception or PerceptionLayer()

        # Capa 2: Léxico — único por agente, empieza vacío
        self.lexicon = LexiconLayer(agent_id=agent_id)

        # Log de instrucciones recibidas del tutor
        self._tutor_instructions: list[dict] = []

    # ------------------------------------------------------------------
    # Instrucción del tutor (grounding inicial)
    # ------------------------------------------------------------------

    def receive_instruction(
        self,
        image_hash: str,
        description: str,
        label: str,
        image: Optional[Union[Image.Image, np.ndarray, str, Path]] = None,
    ) -> None:
        """
        Recibir instrucción del tutor.

        El tutor asigna una etiqueta artificial a una imagen específica,
        acompañada de una descripción verbal de sus propiedades visuales.

        Si se provee la imagen, el agente la codifica con DINOv2 y actualiza
        su léxico inmediatamente. Si no, registra la instrucción para cuando
        vea la imagen.

        Args:
            image_hash:  Identificador único de la imagen.
            description: Descripción verbal del tutor (solo propiedades visuales).
            label:       Etiqueta artificial asignada por el tutor.
            image:       La imagen en sí (opcional si ya fue procesada antes).
        """
        instruction = {
            "image_hash": image_hash,
            "description": description,
            "label": label,
            "round": self.round,
        }
        self._tutor_instructions.append(instruction)

        if image is not None:
            embedding = self.perception.encode(image)
            self.lexicon.assign(embedding, label, self.round)
            self.update_lexicon(image_hash, label)

    def receive_image_for_instruction(
        self,
        image_hash: str,
        image: Union[Image.Image, np.ndarray, str, Path],
    ) -> None:
        """
        Procesar una imagen para la que ya existe una instrucción del tutor.
        Llamar cuando la imagen se entrega después de la instrucción verbal.
        """
        pending = next(
            (i for i in self._tutor_instructions if i["image_hash"] == image_hash),
            None
        )
        if pending:
            embedding = self.perception.encode(image)
            self.lexicon.assign(embedding, pending["label"], self.round)
            self.update_lexicon(image_hash, pending["label"])

    # ------------------------------------------------------------------
    # CONDICIÓN 1: Percepción → Etiqueta (Naming)
    # ------------------------------------------------------------------

    def label_image(
        self,
        image_hash: str,
        image: Union[Image.Image, np.ndarray, str, Path],
    ) -> LabelEvent:
        """
        CONDICIÓN 1 — Nombrar: dada una imagen nueva, asignar una etiqueta
        del vocabulario aprendido.

        Proceso:
          1. DINOv2 convierte la imagen en un vector
          2. LexiconLayer encuentra el centroide más cercano
          3. Retornar la etiqueta de ese centroide

        Si el léxico está vacío o la confianza es muy baja, emitir UNCERTAIN.

        Args:
            image_hash: Identificador único de la imagen.
            image:      La imagen a clasificar.

        Returns:
            LabelEvent para el ledger de consenso.
        """
        embedding = self.perception.encode(image)

        if self.lexicon.is_empty():
            label, confidence = "UNCERTAIN", 0.0
        else:
            label, confidence = self.lexicon.classify(embedding)
            if confidence < CONFIDENCE_THRESHOLD:
                label = "UNCERTAIN"

        # Si el agente emite una etiqueta válida, actualizar su léxico
        # con esta nueva instancia (aprendizaje continuo)
        if label != "UNCERTAIN":
            self.lexicon.assign(embedding, label, self.round)
            self.update_lexicon(image_hash, label)

        event = LabelEvent(
            agent_id=self.agent_id,
            image_hash=image_hash,
            label=label,
            confidence=confidence,
            round_number=self.round,
        )
        self.log_event(event)
        return event

    # ------------------------------------------------------------------
    # CONDICIÓN 2: Etiqueta → Representación Visual (Grounding inverso)
    # ------------------------------------------------------------------

    def test_inverse_grounding(
        self,
        label: str,
        candidate_images: dict[str, Union[Image.Image, np.ndarray, str, Path]],
        correct_image_hash: str,
    ) -> GroundingTestResult:
        """
        CONDICIÓN 2 — Grounding inverso: dado solo el string de la etiqueta,
        puede el agente identificar qué imagen corresponde a ese concepto?

        Este test distingue comprensión de memorización superficial.
        Un agente que solo memorizó pares (imagen, etiqueta) falla aquí
        con imágenes nuevas. Un agente que construyó una representación
        del cluster generaliza correctamente.

        Args:
            label:               La etiqueta artificial (e.g. 'vorpal').
            candidate_images:    image_hash -> imagen. Incluye la correcta
                                 y distractores.
            correct_image_hash:  Cuál es la imagen correcta (ground truth).

        Returns:
            GroundingTestResult con resultado y métricas.
        """
        candidate_embeddings = {
            img_hash: self.perception.encode(img)
            for img_hash, img in candidate_images.items()
        }

        selected_hash, confidence = self.lexicon.recognize_from_label(
            label, candidate_embeddings
        )

        return GroundingTestResult(
            agent_id=self.agent_id,
            label=label,
            selected_image_hash=selected_hash,
            correct_image_hash=correct_image_hash,
            is_correct=(selected_hash == correct_image_hash),
            confidence=confidence,
            round_number=self.round,
        )

    # ------------------------------------------------------------------
    # Retroalimentación de consenso
    # ------------------------------------------------------------------

    def receive_consensus_feedback(self, signal: ConsensusSignal) -> None:
        """
        Recibir retroalimentación del mecanismo de consenso.

        Confirmada → reforzar el cluster en el léxico.
        Rechazada  → penalizar, posiblemente eliminar del léxico.

        El agente NO sabe qué etiqueta usaron los otros agentes.
        Solo sabe si la suya fue validada por la comunidad o no.
        """
        if signal.is_stable:
            self.lexicon.reinforce(signal.your_label)
        elif signal.is_discarded:
            self.lexicon.penalize(signal.your_label)
            if signal.your_label not in self.lexicon.get_labels():
                if signal.image_hash in self.local_lexicon:
                    del self.local_lexicon[signal.image_hash]

        self.log_event(LabelEvent(
            agent_id=self.agent_id,
            image_hash=signal.image_hash,
            label=signal.your_label,
            confidence=0.0,
            round_number=self.round,
            is_stable=signal.is_stable,
            notes=f"consensus: stable={signal.is_stable}, discarded={signal.is_discarded}",
        ))

    # ------------------------------------------------------------------
    # Estado y diagnóstico
    # ------------------------------------------------------------------

    def vocabulary_size(self) -> int:
        return self.lexicon.vocabulary_size()

    def lexicon_summary(self) -> dict:
        return self.lexicon.summary()

    def __repr__(self) -> str:
        return (
            f"LearnerAgent(id={self.agent_id}, "
            f"round={self.round}, "
            f"vocab={self.vocabulary_size()})"
        )


# ------------------------------------------------------------------
# Factory — crear múltiples agentes compartiendo la PerceptionLayer
# ------------------------------------------------------------------

def create_agents(n: int, device: str = "auto") -> list[LearnerAgent]:
    """
    Crear N agentes que comparten una sola instancia de PerceptionLayer.

    Correcto científicamente (todos perciben igual) y eficiente
    computacionalmente (DINOv2 se carga una sola vez).

    Args:
        n:      Número de agentes.
        device: 'auto', 'cpu', o 'cuda'.

    Returns:
        Lista de LearnerAgent listos para el experimento.
    """
    shared_perception = PerceptionLayer(device=device)
    return [
        LearnerAgent(agent_id=f"agent_{i:02d}", perception=shared_perception)
        for i in range(n)
    ]
