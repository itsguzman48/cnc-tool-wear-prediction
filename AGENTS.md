# Instrucciones para el agente

## Contexto
Soy Íñigo, sysadmin (ASIR) cursando la especialización en IA y Big Data. Este repo es un proyecto de portfolio: tengo que entender y poder defender cada línea en una entrevista técnica.

## Cómo quiero que trabajes
- Modo tutor: antes de escribir el código de una fase, explícame en pocas líneas qué vas a hacer y por qué.
- Trabaja fase a fase. No empieces la siguiente fase hasta que yo lo pida.
- Código y comentarios en español. Los términos técnicos habituales (dataset, pipeline, feature, deploy) en inglés.
- Comenta el porqué de cada decisión no obvia, no el qué.
- Pídeme confirmación antes de instalar paquetes nuevos o borrar archivos.
- La carpeta data/ está ignorada por git: no copies datos a otras carpetas del repo.

## Rigor técnico
- Separa train y test por experimento, nunca por filas sueltas: las filas de un mismo experimento están correlacionadas y mezclarlas infla las métricas (data leakage).
- Hay pocos experimentos: evalúa con validación cruzada agrupada por experimento (GroupKFold o LeaveOneGroupOut) y deja clara esa limitación en los resultados.
- Fija semillas aleatorias para que los resultados sean reproducibles.
- Empieza siempre por un baseline simple antes de probar modelos complejos.
- Si no estás seguro de algo (versión de una librería, comportamiento de una función), dilo en vez de inventarlo.

## Al terminar cada fase
- Resume qué se ha hecho, qué archivos se han creado y qué resultados han salido.
- Propón un mensaje de commit corto en español.
