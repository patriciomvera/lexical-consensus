"""
perception.py
-------------
Capa de Percepción — el substrato visual compartido.

Usa DINOv2 (Meta, 2023): un encoder visual entrenado por auto-supervisión,
sin ninguna etiqueta categórica. Sabe que dos manzanas son visualmente
similares, pero no tiene representación lingüística de "manzana".

Esta capa es:
  - IDÉNTICA en todos los agentes (substrato perceptual compartido)
  - CONGELADA durante el experimento (no aprende nada nuevo)
  - CIEGA a etiquetas (su espacio de embeddings es puramente visual)

Es el equivalente computacional de la corteza visual en el paper de Vera et al.:
todos los agentes perciben el mundo de la misma forma, pero las etiquetas
emergen por consenso social — no están precargadas.

Referencia: Oquab et al. (2023) DINOv2: Learning Robust Visual Features
without Supervision. https://arxiv.org/abs/2304.07193
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


# DINOv2 small — buen balance entre calidad y velocidad para el experimento
DINO_MODEL = "facebook/dinov2-small"
EMBEDDING_DIM = 384   # Dimensión del espacio de representación de DINOv2-small


class PerceptionLayer:
    """
    Encoder visual sin etiquetas basado en DINOv2.

    Convierte cualquier imagen en un vector de 384 dimensiones
    que captura su estructura visual sin ningún sesgo categórico.

    El espacio de embeddings está organizado por similitud visual:
    dos imágenes cercanas en este espacio son visualmente similares,
    sin importar cómo se llamen en ningún idioma.
    """

    def __init__(self, device: str = "auto"):
        self.device = self._resolve_device(device)
        self._processor = None
        self._model = None
        self._cache: dict[str, np.ndarray] = {}

    def _resolve_device(self, device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _load(self) -> None:
        """Carga el modelo lazy — solo cuando se necesita por primera vez."""
        if self._model is None:
            print(f"  [Perception] Cargando DINOv2 ({DINO_MODEL})...")
            self._processor = AutoImageProcessor.from_pretrained(DINO_MODEL)
            self._model = AutoModel.from_pretrained(DINO_MODEL).to(self.device)
            self._model.eval()  # Congelado — nunca en modo training
            print(f"  [Perception] Listo. Dispositivo: {self.device}, dim: {EMBEDDING_DIM}")

    def encode(self, image: Union[Image.Image, np.ndarray, str, Path]) -> np.ndarray:
        """
        Convierte una imagen en su vector de representación visual.

        Args:
            image: PIL Image, array numpy, o path a archivo de imagen.

        Returns:
            Vector numpy de shape (384,), normalizado a norma unitaria.
        """
        self._load()

        # Normalizar input
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        elif isinstance(image, Image.Image):
            image = image.convert("RGB")

        # Calcular hash para cache
        img_hash = self._hash_image(image)
        if img_hash in self._cache:
            return self._cache[img_hash]

        # Encoding
        inputs = self._processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self._model(**inputs)
            # Usamos el CLS token — representación global de la imagen
            embedding = outputs.last_hidden_state[:, 0, :]
            embedding = F.normalize(embedding, dim=-1)  # Norma unitaria

        vector = embedding.squeeze().cpu().numpy()
        self._cache[img_hash] = vector
        return vector

    def encode_batch(self, images: list) -> np.ndarray:
        """
        Codifica múltiples imágenes eficientemente.

        Returns:
            Array de shape (n_images, 384).
        """
        return np.stack([self.encode(img) for img in images])

    def similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """
        Similitud coseno entre dos vectores.
        Rango: -1 (opuestos) a 1 (idénticos).
        Los vectores están normalizados, así que es solo el producto punto.
        """
        return float(np.dot(vec_a, vec_b))

    @staticmethod
    def _hash_image(image: Image.Image) -> str:
        """Hash reproducible de una imagen para cache."""
        return hashlib.md5(image.tobytes()).hexdigest()

    @property
    def embedding_dim(self) -> int:
        return EMBEDDING_DIM
