# Model Card — Sistema Algorítmico de Reclutamiento

> Tarjeta de modelo elaborada siguiendo el estándar propuesto por **Mitchell et al. (2019)** en *Model Cards for Model Reporting*. Este documento representa lo que la empresa del caso analizado debería haber publicado para cumplir con principios mínimos de transparencia algorítmica.

---

## 1. Información general del modelo

| Campo | Detalle |
|---|---|
| **Nombre del modelo** | Hiring Classifier v1.0 (caso hipotético) |
| **Tipo de modelo** | Clasificador binario supervisado (Random Forest, 100 estimadores, max_depth=8) |
| **Fecha del reporte** | Mayo de 2026 |
| **Versión** | 1.0 |
| **Autor del reporte** | Juan Felipe Guerrero Urueña |
| **Contexto** | Actividad académica — Unidad 3, Electiva II IA Avanzada, Tecnológica del Oriente |
| **Licencia del reporte** | MIT |

---

## 2. Uso previsto

### Casos de uso primarios
Apoyar a un equipo de talento humano en la priorización inicial de hojas de vida durante procesos de selección, devolviendo una probabilidad de "ajuste al perfil" entre 0 y 1 para cada candidato.

### Usuarios previstos
Reclutadores y analistas de talento humano dentro de una empresa privada, no candidatos finales.

### Usos **fuera del alcance** del modelo
- Decisiones automáticas de contratación o rechazo **sin revisión humana**.
- Aplicación en cargos directivos, sensibles o de alta confianza, donde el juicio humano es irreemplazable.
- Uso en sectores regulados (salud, justicia, educación pública) sin validación específica adicional.
- Cualquier uso fuera del territorio donde fue auditado el modelo.

---

## 3. Factores

### Grupos relevantes evaluados
- **Género:** masculino, femenino
- **Edad:** ≤45 años, >45 años
- **Combinaciones interseccionales:** los cuatro cruces género × rango etario

### Factores no evaluados (limitación documentada)
El modelo **no fue auditado** sobre etnia, discapacidad, orientación sexual, religión o estatus socioeconómico. Cualquier despliegue real requeriría auditoría adicional sobre estos atributos protegidos.

---

## 4. Métricas

### Métricas de desempeño técnico
- **Accuracy global** (proporción de predicciones correctas)
- **Precision, Recall, F1-Score** (por clase)

### Métricas de equidad
- **Tasa de aceptación por grupo** (paridad demográfica)
- **Disparate Impact Ratio** (regla del 4/5 — umbral 0.80)
- **True Positive Rate por grupo** (igualdad de oportunidades)
- **Diferencia absoluta de TPR entre grupos**

### Umbral regulatorio aplicado
La **regla del 4/5** (Equal Employment Opportunity Commission, USA): una práctica de selección se considera discriminatoria cuando el ratio de selección del grupo desfavorecido sobre el favorecido cae por debajo de **0.80**. Esta regla, aunque originada en EE.UU., es la referencia operativa más utilizada internacionalmente para evaluar disparidad algorítmica en contextos laborales.

---

## 5. Datos de entrenamiento

### Conjunto de datos
- **Origen:** Simulado mediante el script \`bias_audit.py\` (semilla fija = 42, reproducible)
- **Tamaño:** 600 registros sintéticos
- **División:** 75% entrenamiento, 25% prueba, estratificación por clase
- **Variables:** edad, género, años de experiencia, nivel educativo, puntaje de prueba

### Sesgo estructural deliberado
El dataset incorpora un sesgo histórico calibrado a niveles realistas reportados en auditorías de sistemas comerciales (Amazon 2018, ~8 pp; LinkedIn 2021, ~6 pp): para igual calificación objetiva, las mujeres tienen 8 pp menos de probabilidad histórica de contratación, y las personas mayores de 45 años tienen 6 pp menos. Esta calibración permite estudiar empíricamente el comportamiento de las estrategias de mitigación.

### Limitaciones de los datos
Por tratarse de datos sintéticos generados con fines educativos, **no se debe extrapolar el desempeño** del modelo a poblaciones reales. Cualquier despliegue productivo requeriría datos representativos del mercado laboral objetivo, idealmente curados con la participación de comunidades afectadas.

---

## 6. Evaluación cuantitativa

### Resultados de la auditoría sin mitigación

| Métrica | Valor | Estado |
|---|---|---|
| Accuracy global | 0.820 | ✓ |
| Tasa aceptación hombres | 0.750 | — |
| Tasa aceptación mujeres | 0.513 | — |
| **DIR — Género** | **0.684** | **✗ Incumple regla 4/5** |
| Tasa aceptación ≤45 años | 0.407 | — |
| Tasa aceptación >45 años | 0.884 | — |
| **DIR — Edad** | **0.461** | **✗ Incumple regla 4/5** |

### Resultados con reweighing (Kamiran & Calders, 2012)

| Métrica | Valor | Cambio | Estado |
|---|---|---|---|
| Accuracy global | 0.853 | +3.33 pp | ✓ |
| **DIR — Género** | **0.788** | **+0.105** | **△ Borderline (≈ umbral)** |
| **DIR — Edad** | **0.502** | **+0.041** | **✗ Mejora insuficiente** |

### Interpretación

El reweighing **mejora ambas dimensiones de equidad** y, contraintuitivamente, también la precisión global — porque el sesgo original era contraproducente para el aprendizaje del modelo. Sin embargo, **una sola intervención no basta**: el grupo etario >45 años sigue sin alcanzar el umbral regulatorio. Esto confirma empíricamente el argumento del informe sobre la necesidad de **combinar estrategias** (preprocesamiento + ajuste de umbrales + supervisión humana) en lugar de confiar en una sola técnica.

---

## 7. Consideraciones éticas

### Riesgos identificados
- **Discriminación algorítmica** por reproducción de sesgos históricos (género, edad).
- **Sesgo por proxy** mediante variables aparentemente neutras (años de experiencia continua puede correlacionar con edad).
- **Pérdida de autonomía humana** si los reclutadores se subordinan al puntaje del modelo.
- **Falta de explicabilidad** ante candidatos rechazados.

### Mitigaciones implementadas en este reporte
- Reweighing en preprocesamiento (Kamiran & Calders, 2012).
- Auditoría sistemática con métricas desagregadas por grupo protegido.
- Documentación pública vía este Model Card.

### Mitigaciones **pendientes** para uso real
- Eliminación o atenuación de variables proxy.
- Calibración de umbrales por grupo (post-processing).
- Protocolo obligatorio de revisión humana sobre decisiones negativas.
- Canal de apelación para candidatos.
- Auditoría externa por parte de un tercero independiente.
- Validación prospectiva sobre datos reales del mercado objetivo.

---

## 8. Recomendaciones de uso responsable

1. **Nunca usar el modelo como decisor autónomo.** Su rol es informar a un reclutador humano, no reemplazarlo.
2. **Auditar trimestralmente** con métricas desagregadas por todos los atributos protegidos relevantes en la jurisdicción de operación.
3. **Documentar las decisiones** del modelo y mantener un registro auditable de las predicciones realizadas.
4. **Ofrecer explicaciones** a los candidatos cuyas postulaciones sean descartadas, junto con un canal de apelación efectivo.
5. **Cumplir con la normativa local:** en Colombia, Ley 1581 de 2012 (protección de datos personales) y, cuando aplique, las regulaciones laborales sectoriales del Ministerio del Trabajo.

---

## 9. Marco normativo aplicable

| Norma | Jurisdicción | Aplicabilidad |
|---|---|---|
| Ley 1581 de 2012 (Habeas Data) | Colombia | Tratamiento de datos personales de candidatos |
| Recomendación UNESCO sobre Ética en IA (2021) | Internacional | Principios rectores: equidad, transparencia, rendición de cuentas |
| Ley de IA (AI Act) | Unión Europea | Aplicable si la empresa opera o recluta en la UE; reclutamiento está clasificado como uso de alto riesgo |
| Regla del 4/5 (EEOC, 1978) | EE.UU. (referencia internacional) | Umbral operativo para Disparate Impact Ratio |

---

## 10. Referencias

- Barocas, S., Hardt, M., & Narayanan, A. (2023). *Fairness and Machine Learning: Limitations and Opportunities*. MIT Press. https://fairmlbook.org
- Kamiran, F., & Calders, T. (2012). Data preprocessing techniques for classification without discrimination. *Knowledge and Information Systems*, 33(1), 1–33. https://doi.org/10.1007/s10115-011-0463-8
- Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji, I. D., & Gebru, T. (2019). Model Cards for Model Reporting. *Proceedings of the ACM Conference on Fairness, Accountability, and Transparency*, 220–229. https://doi.org/10.1145/3287560.3287596
- UNESCO. (2021). *Recomendación sobre la Ética de la Inteligencia Artificial*. https://unesdoc.unesco.org/ark:/48223/pf0000380455_spa

---

**Versión:** 1.0 · **Última actualización:** 2026-05-26 · **Mantenido por:** [Juan Felipe Guerrero Urueña](https://www.grupoguerreroherrera.com/)
