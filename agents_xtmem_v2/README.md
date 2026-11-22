# Agentes de Lexical Consensus ante los estímulos de Lewis y Frank

Corrida exploratoria del 22 de noviembre de 2026 para la segunda versión de *Visual Similarity in Adult Word Extension and Artificial Word Learning*. Las respuestas humanas y los casos de interés se conocían de antemano. Los agentes vieron las imágenes de Lewis y Frank (2018). La réplica multi-encoder de los episodios originales de CIFAR tiene un protocolo propio.

[Protocolo de la corrida](protocol.md) · [Figuras en PDF](outputs/Agentes_y_generalizacion.pdf) · [Tablas de resultados](outputs/) · [Versiones y huellas de los datos](outputs/run.json)

## Qué se ejecutó

Las tres categorías son animales, vegetales y vehículos. En cada una se enseñó una etiqueta anónima con uno, tres ejemplos subordinados, tres básicos o tres superordinados. Se evaluaron las 24 imágenes fijas después de cada enseñanza. Cada imagen recibió 600 respuestas humanas por condición y dominio, de un estudio con 520 participantes únicos. El modelo produce una salida determinista por combinación con semilla fija. El conjunto principal contiene ocho representaciones entrenadas y dos controles (una ResNet aleatoria y los píxeles). Se incluyeron por separado dos inicializaciones adicionales de la ResNet y una variante CLS de MAE. Se usó el preprocesamiento nativo archivado.

Los soportes y las imágenes de prueba son disjuntos. Se utilizaron las implementaciones originales de los aprendices y los embeddings almacenados; el cálculo verifica los hashes Git de los módulos, comprueba la identidad de las elecciones humanas en los 13 archivos y contrasta las distancias al centroide de las 172.800 filas por representación con la salida archivada. El error máximo global fue inferior a 4,4 × 10⁻⁷. Los identificadores de dominio solo organizan los episodios o evalúan las salidas. Los aprendices ven vectores y números de etiqueta.

## Dos tareas

**Una palabra.** El centroide, el aprendiz de dos prototipos, el kNN ponderado de tres vecinos y el control aleatorio reciben solo ejemplos positivos de la palabra actual. Sus puntuaciones permiten ordenar las imágenes. Con tres ejemplos y una sola palabra, el kNN suma las tres similitudes; su orden coincide matemáticamente con el del centroide. La regresión logística y el SVM precisan al menos dos clases.

**Aceptación de varias imágenes.** Para los tres aprendices geométricos añadimos una regla simétrica que usa solo el soporte y una candidata a la vez. Con cuatro imágenes (los tres ejemplos más una candidata), cada imagen se puntúa usando las otras tres como enseñanza. La candidata es aceptada si al menos uno de los tres ejemplos obtiene una puntuación igual o menor que la suya (tolerancia 10⁻⁶). Así el criterio se adapta a la dispersión de los ejemplos e incluso puede admitir elementos de otros dominios. La puntuación inicial y la decisión final pueden diferir en el orden relativo de dos candidatas porque cada una se compara con un grupo distinto. En la condición de un ejemplo conservamos los scores; la simetría de dos imágenes impide una decisión de extensión informativa con esta regla. Las celdas correspondientes quedan vacías.

**Umbral fijo.** Como contraste, se calcularon las fracciones de aceptación de los 12 ítems superordinados para todos los umbrales posibles sobre el score bruto, usando el mismo umbral en ambos tipos de enseñanza y en los tres dominios. Se informa el máximo salto que puede producir un umbral fijo en estas imágenes, sin calibrarlo con las respuestas humanas. Este máximo es una cota descriptiva para la regla especificada.

**Tres palabras.** En cuatro episodios, uno por condición, los seis aprendices reciben soportes etiquetados de los tres dominios en simultáneo (un identificador anónimo por dominio). La salida nativa predict asigna una etiqueta por imagen. Esto introduce evidencia contrastiva adicional respecto del ensayo humano. Los logits y los scores de nombres distintos tienen escalas propias. El modelo aleatorio con semilla 42 es un control, y sus datos no se interpretan como evidencia sobre el encoder.

## Lectura de los resultados

Con una sola palabra y centroide, cuatro de los ocho encoders entrenados puntúan el auto por encima de la tortuga y cuatro hacen lo contrario tras ver el bus, la moto y el tractor. El aprendiz de dos prototipos invierte el orden del centroide en DINOv2-small, DINOv2-base y OpenCLIP. En adultos hubo 458 elecciones del auto y una de la tortuga en 600 sesiones.

En los 12 candidatos superordinados, el salto humano entre enseñanza estrecha y amplia fue de 76,43 puntos porcentuales. Para los ocho encoders entrenados y centroide, el máximo alcanzable con un umbral fijo sobre su puntuación va de 8,33 a 66,67 puntos. Al usar el criterio sensible a la dispersión, el ViT supervisado pasó de 0/12 a 10/12 y rechazó los 48 candidatos de otros dominios tras la enseñanza amplia. DINOv2-small pasó de 0/12 a 7/12 y aceptó 6/48 de fuera. La ResNet aleatoria (semilla 0) pasó de 1/12 a 9/12 y aceptó 38/48 de fuera. Esos 48 son combinaciones de dominio enseñado e imagen, con reutilización de fotografías. Las proporciones humanas se calculan sobre decisiones por sesión, las del modelo sobre ítems. No se estimaron tasas de una población de agentes ni un mecanismo causal de la elección humana.

## Archivos de resultados

| Archivo | Contenido |
|---|---|
| [one_word_items.csv](outputs/one_word_items.csv) | Scores por etiqueta única, decisiones cuando están definidas y recuentos humanos. |
| [three_word_items.csv](outputs/three_word_items.csv) | Score de cada etiqueta objetivo, predicción nativa y recuentos humanos. Cada imagen tiene una sola predicción, representada frente a tres posibles palabras objetivo. |
| [broad_teaching_pairs.csv](outputs/broad_teaching_pairs.csv) | Los cinco pares discutidos, con dirección del ranking, decisiones y comparador humano. A/B son las imágenes identificadas en cada fila. |
| [condition_by_domain.csv](outputs/condition_by_domain.csv) | Resumen por dominio, condición y tipo de candidata. La columna items indica el número de candidatas y la fracción model_fraction su proporción aceptada. |
| [broad_minus_narrow.csv](outputs/broad_minus_narrow.csv) | Fracciones de selección y diferencias en los 12 candidatos superordinados de los tres dominios, incluyendo la cota para un umbral fijo. Las fracciones se expresan entre 0 y 1. |
| [run.json](outputs/run.json) | Commit, versiones, SHA-256 de cada cache de embeddings y verificación de distancias. |
| [Agentes_y_generalizacion.pdf](outputs/Agentes_y_generalizacion.pdf) | Dos figuras sobre la tarea de una palabra y la clasificación original de tres palabras. |

Los scores ausentes de una clase en la votación de kNN se guardan como -inf, tal como los devuelve la implementación original. Los scores de los modelos aleatorios dependen de la semilla. La fracción accepted en los ítems individuales es una decisión 0/1 cuando está definida. Una diferencia mínima entre scores puede producir un orden determinado sin implicar una diferencia sustantiva de aceptación.

## Reproducción

Desde una copia de esta rama, usar Python 3.12.14 y las dependencias indicadas. El script comprueba que los módulos de los aprendices, los 13 archivos de embeddings y las 13 tablas humanas coincidan byte por byte con los del commit de entrada `23e4082`. El directorio `reproduced/` recibe las salidas de la nueva ejecución.

```sh
git clone --branch agents-xtmem-v2 --single-branch https://github.com/patriciomvera/lexical-consensus.git
cd lexical-consensus/agents_xtmem_v2
python -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python run.py .. --out reproduced
python plot.py reproduced
for f in outputs/*.csv outputs/run.json; do cmp "$f" "reproduced/${f##*/}" || exit 1; done
```

Fuentes: [Lexical Consensus, código y datos](https://github.com/patriciomvera/lexical-consensus/tree/23e4082b0bd7a33889714d0d179fa5a12791fb6c), [estímulos y diseño XTMEM](https://github.com/mllewis/XTMEM/blob/220ed17ec2b85d30a4053711209252fdbc3cb166/experiments/exp1/js/exp1.js), [Lewis y Frank, OSF](https://osf.io/yekhj/).
