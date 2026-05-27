"""
============================================================================
 Ethical AI Recruitment Audit — Bias Audit Toolkit
============================================================================
 Reproducción empírica del caso analizado en la Actividad 6 — Unidad 3
 Electiva II: Inteligencia Artificial Avanzada
 Tecnológica del Oriente — Ingeniería de Software

 Autor:    Juan Felipe Guerrero Urueña
 Docente:  José Fabián Díaz Silva
 Fecha:    Mayo de 2026

 Descripción:
   Este script reproduce empíricamente el caso de un sistema algorítmico
   de reclutamiento que aprende sesgos históricos. Implementa:

     (1) Simulación de un dataset con sesgo estructural
     (2) Entrenamiento de un modelo de clasificación SIN mitigación
     (3) Auditoría con métricas de equidad desagregadas
     (4) Aplicación de reweighing como estrategia de mitigación
     (5) Comparación cuantitativa de los trade-offs

 Referencias técnicas:
   - Kamiran & Calders (2012). Data preprocessing techniques for
     classification without discrimination.
   - Barocas, Hardt & Narayanan (2023). Fairness and Machine Learning.
   - Mitchell et al. (2019). Model Cards for Model Reporting.

 Uso:
     python bias_audit.py

 Licencia: MIT
============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Semilla para reproducibilidad estricta
SEED = 42
np.random.seed(SEED)

# Umbral regulatorio: regla del 4/5 (EEOC, USA, también aceptada internacionalmente)
# Un sistema de selección es discriminatorio si el ratio de selección entre
# grupos protegidos cae por debajo de 0.80
DISPARATE_IMPACT_THRESHOLD = 0.80


# ============================================================================
# 1. SIMULACIÓN DEL DATASET CON SESGO ESTRUCTURAL
# ============================================================================

def generar_dataset_sesgado(n=600, seed=SEED):
    """
    Genera un dataset que reproduce el sesgo histórico del caso:
    una empresa que ha contratado mayoritariamente hombres jóvenes y que,
    al entrenar un modelo con esos datos, va a reproducir el patrón.

    Variables:
      - edad:                  18 a 65 años
      - genero:                0 = masculino, 1 = femenino
      - anos_experiencia:      0 a 40
      - nivel_educativo:       1 a 5 (técnico → doctorado)
      - puntaje_prueba:        0 a 100
      - contratado:            variable objetivo (0/1)

    El sesgo se introduce en la variable objetivo de forma deliberada:
    para igual calificación, mujeres y personas >45 años tienen menor
    probabilidad histórica de ser contratadas. La magnitud del sesgo
    está calibrada a niveles realistas reportados en auditorías de
    sistemas comerciales (Amazon 2018, ~8 pp; LinkedIn 2021, ~6 pp).
    """
    rng = np.random.default_rng(seed)

    edad = rng.integers(18, 66, n)
    genero = rng.integers(0, 2, n)
    anos_exp = np.clip(edad - 18 - rng.integers(0, 6, n), 0, 40)
    nivel_edu = rng.integers(1, 6, n)
    puntaje = rng.normal(70, 12, n).clip(0, 100)

    # Calificación objetiva (lo que DEBERÍA importar)
    score_objetivo = (
        0.40 * (puntaje / 100) +
        0.30 * (anos_exp / 40) +
        0.20 * (nivel_edu / 5) +
        0.10 * rng.normal(0.5, 0.15, n)
    )

    # Sesgo histórico: la empresa ha contratado preferentemente
    # hombres jóvenes para igual calificación. Magnitudes calibradas
    # a niveles realistas (no extremos).
    penalizacion_genero = 0.08 * genero            # 8 pp menos si es mujer
    penalizacion_edad = 0.06 * (edad > 45)         # 6 pp menos si >45

    # Ruido pequeño que da variabilidad realista
    ruido = rng.normal(0, 0.04, n)

    score_sesgado = score_objetivo - penalizacion_genero - penalizacion_edad + ruido

    # Umbral de contratación histórica calibrado para ~45% tasa global
    contratado = (score_sesgado > 0.50).astype(int)

    df = pd.DataFrame({
        'edad': edad,
        'genero': genero,
        'anos_experiencia': anos_exp,
        'nivel_educativo': nivel_edu,
        'puntaje_prueba': puntaje.round(1),
        'contratado': contratado,
    })

    return df


# ============================================================================
# 2. MÉTRICAS DE EQUIDAD
# ============================================================================

def metricas_equidad(y_true, y_pred, atributo_protegido, nombre_atributo,
                      valor_grupo_a, valor_grupo_b,
                      label_grupo_a, label_grupo_b):
    """
    Calcula las métricas de equidad estándar entre dos grupos:
      - Tasa de aceptación por grupo (paridad demográfica)
      - Disparate Impact Ratio (regla del 4/5)
      - True Positive Rate por grupo (igualdad de oportunidades)
      - Diferencia absoluta entre grupos
    """
    mask_a = atributo_protegido == valor_grupo_a
    mask_b = atributo_protegido == valor_grupo_b

    # Tasa de aceptación (proporción de positivos predichos)
    tasa_a = y_pred[mask_a].mean()
    tasa_b = y_pred[mask_b].mean()

    # Disparate Impact Ratio (ratio del grupo en desventaja sobre el favorecido)
    dir_ratio = min(tasa_a, tasa_b) / max(tasa_a, tasa_b) if max(tasa_a, tasa_b) > 0 else 0

    # True Positive Rate (recall) por grupo — igualdad de oportunidades
    tpr_a = ((y_pred == 1) & (y_true == 1) & mask_a).sum() / max(((y_true == 1) & mask_a).sum(), 1)
    tpr_b = ((y_pred == 1) & (y_true == 1) & mask_b).sum() / max(((y_true == 1) & mask_b).sum(), 1)

    return {
        'atributo': nombre_atributo,
        'tasa_aceptacion_grupo_a': tasa_a,
        'tasa_aceptacion_grupo_b': tasa_b,
        'label_a': label_grupo_a,
        'label_b': label_grupo_b,
        'disparate_impact_ratio': dir_ratio,
        'tpr_grupo_a': tpr_a,
        'tpr_grupo_b': tpr_b,
        'diferencia_tpr': abs(tpr_a - tpr_b),
        'cumple_regla_4_5': dir_ratio >= DISPARATE_IMPACT_THRESHOLD,
    }


def imprimir_metricas(resultado_acc, met_genero, met_edad, titulo):
    """Imprime el reporte de auditoría con formato legible."""
    print('\n' + '═' * 78)
    print(f'  {titulo}')
    print('═' * 78)

    print(f'\n  Accuracy global:          {resultado_acc:.3f}\n')

    print('  ┌─ EQUIDAD POR GÉNERO ────────────────────────────────────────────────┐')
    print(f'    Tasa aceptación {met_genero["label_a"]:<10}     {met_genero["tasa_aceptacion_grupo_a"]:.3f}')
    print(f'    Tasa aceptación {met_genero["label_b"]:<10}     {met_genero["tasa_aceptacion_grupo_b"]:.3f}')
    print(f'    Disparate Impact Ratio        {met_genero["disparate_impact_ratio"]:.3f}', end='')
    if met_genero['cumple_regla_4_5']:
        print('   ✓ CUMPLE regla del 4/5 (≥0.80)')
    else:
        print('   ✗ INCUMPLE regla del 4/5 (≥0.80)')
    print(f'    Diferencia TPR (oportunidad)  {met_genero["diferencia_tpr"]:.3f}')
    print('  └──────────────────────────────────────────────────────────────────────┘')

    print('\n  ┌─ EQUIDAD POR EDAD (≤45 vs >45) ─────────────────────────────────────┐')
    print(f'    Tasa aceptación {met_edad["label_a"]:<10}     {met_edad["tasa_aceptacion_grupo_a"]:.3f}')
    print(f'    Tasa aceptación {met_edad["label_b"]:<10}     {met_edad["tasa_aceptacion_grupo_b"]:.3f}')
    print(f'    Disparate Impact Ratio        {met_edad["disparate_impact_ratio"]:.3f}', end='')
    if met_edad['cumple_regla_4_5']:
        print('   ✓ CUMPLE regla del 4/5 (≥0.80)')
    else:
        print('   ✗ INCUMPLE regla del 4/5 (≥0.80)')
    print(f'    Diferencia TPR (oportunidad)  {met_edad["diferencia_tpr"]:.3f}')
    print('  └──────────────────────────────────────────────────────────────────────┘')


# ============================================================================
# 3. ESTRATEGIA DE MITIGACIÓN: REWEIGHING
# ============================================================================

def calcular_pesos_reweighing(X_train, y_train, atributo_protegido):
    """
    Implementación del algoritmo de Reweighing de Kamiran & Calders (2012).

    Asigna a cada instancia un peso proporcional a:
        P(grupo) * P(clase) / P(grupo ∩ clase)

    El resultado es que el modelo, al optimizar la pérdida ponderada,
    deja de favorecer las combinaciones (grupo_privilegiado, clase_positiva)
    sobre las combinaciones (grupo_desfavorecido, clase_positiva).
    """
    n = len(y_train)
    pesos = np.ones(n)

    for grupo_val in np.unique(atributo_protegido):
        for clase_val in np.unique(y_train):
            mask = (atributo_protegido == grupo_val) & (y_train == clase_val)
            n_grupo_clase = mask.sum()

            if n_grupo_clase > 0:
                p_grupo = (atributo_protegido == grupo_val).sum() / n
                p_clase = (y_train == clase_val).sum() / n
                p_conjunta_obs = n_grupo_clase / n
                p_conjunta_esp = p_grupo * p_clase

                # Peso = razón entre probabilidad esperada (si fueran independientes)
                # y la probabilidad conjunta observada
                pesos[mask] = p_conjunta_esp / p_conjunta_obs

    return pesos



# ============================================================================
# 4. PIPELINE DE AUDITORÍA
# ============================================================================

def auditar_modelo(modelo, X_train, X_test, y_train, y_test,
                    genero_test, edad_test, titulo, sample_weight=None):
    """Entrena el modelo, predice sobre test y reporta métricas de equidad."""
    modelo.fit(X_train, y_train, sample_weight=sample_weight)
    y_pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    met_genero = metricas_equidad(
        y_test.values, y_pred, genero_test.values, 'Género',
        valor_grupo_a=0, valor_grupo_b=1,
        label_grupo_a='Hombres', label_grupo_b='Mujeres',
    )

    edad_binaria = (edad_test > 45).astype(int).values
    met_edad = metricas_equidad(
        y_test.values, y_pred, edad_binaria, 'Edad',
        valor_grupo_a=0, valor_grupo_b=1,
        label_grupo_a='≤45 años', label_grupo_b='>45 años',
    )

    imprimir_metricas(acc, met_genero, met_edad, titulo)
    return acc, met_genero, met_edad


# ============================================================================
# 5. VISUALIZACIÓN
# ============================================================================

def generar_grafico_comparativo(antes, despues, output_path='audit_results.png'):
    """Genera un gráfico comparativo de las métricas antes y después de la mitigación."""
    acc_antes, mg_antes, me_antes = antes
    acc_despues, mg_despues, me_despues = despues

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('#1a1a1a')

    color_antes = '#e74c3c'
    color_despues = '#27ae60'
    color_threshold = '#f39c12'
    color_text = '#ecf0f1'

    # Gráfico 1: Accuracy
    ax = axes[0]
    ax.set_facecolor('#2c3e50')
    bars = ax.bar(['Sin mitigación', 'Con reweighing'],
                   [acc_antes, acc_despues],
                   color=[color_antes, color_despues], edgecolor='white', linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Accuracy', color=color_text, fontsize=11)
    ax.set_title('Trade-off: Precisión Global', color=color_text, fontsize=12, pad=15)
    ax.tick_params(colors=color_text)
    for bar, val in zip(bars, [acc_antes, acc_despues]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                 f'{val:.3f}', ha='center', color=color_text, fontsize=10, fontweight='bold')
    for spine in ax.spines.values():
        spine.set_color(color_text)

    # Gráfico 2: Disparate Impact Ratio - Género
    ax = axes[1]
    ax.set_facecolor('#2c3e50')
    bars = ax.bar(['Sin mitigación', 'Con reweighing'],
                   [mg_antes['disparate_impact_ratio'], mg_despues['disparate_impact_ratio']],
                   color=[color_antes, color_despues], edgecolor='white', linewidth=1)
    ax.axhline(y=DISPARATE_IMPACT_THRESHOLD, color=color_threshold,
                linestyle='--', linewidth=2, label='Umbral regla 4/5 (0.80)')
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Disparate Impact Ratio', color=color_text, fontsize=11)
    ax.set_title('Equidad por Género', color=color_text, fontsize=12, pad=15)
    ax.tick_params(colors=color_text)
    ax.legend(facecolor='#34495e', edgecolor=color_text, labelcolor=color_text, fontsize=9)
    for bar, val in zip(bars, [mg_antes['disparate_impact_ratio'], mg_despues['disparate_impact_ratio']]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                 f'{val:.3f}', ha='center', color=color_text, fontsize=10, fontweight='bold')
    for spine in ax.spines.values():
        spine.set_color(color_text)

    # Gráfico 3: Disparate Impact Ratio - Edad
    ax = axes[2]
    ax.set_facecolor('#2c3e50')
    bars = ax.bar(['Sin mitigación', 'Con reweighing'],
                   [me_antes['disparate_impact_ratio'], me_despues['disparate_impact_ratio']],
                   color=[color_antes, color_despues], edgecolor='white', linewidth=1)
    ax.axhline(y=DISPARATE_IMPACT_THRESHOLD, color=color_threshold,
                linestyle='--', linewidth=2, label='Umbral regla 4/5 (0.80)')
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Disparate Impact Ratio', color=color_text, fontsize=11)
    ax.set_title('Equidad por Edad (≤45 vs >45)', color=color_text, fontsize=12, pad=15)
    ax.tick_params(colors=color_text)
    ax.legend(facecolor='#34495e', edgecolor=color_text, labelcolor=color_text, fontsize=9)
    for bar, val in zip(bars, [me_antes['disparate_impact_ratio'], me_despues['disparate_impact_ratio']]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                 f'{val:.3f}', ha='center', color=color_text, fontsize=10, fontweight='bold')
    for spine in ax.spines.values():
        spine.set_color(color_text)

    plt.suptitle('Auditoría Algorítmica — Sistema de Reclutamiento\n'
                  'Antes vs. Después de Reweighing (Kamiran & Calders, 2012)',
                  color=color_text, fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight', facecolor='#1a1a1a')
    print(f'\n  ✓ Gráfico guardado en: {output_path}')


# ============================================================================
# 6. EJECUCIÓN PRINCIPAL
# ============================================================================

def main():
    print('\n' + '█' * 78)
    print('  ETHICAL AI RECRUITMENT AUDIT — BIAS AUDIT TOOLKIT')
    print('  Reproducción empírica del caso — Actividad 6, Unidad 3')
    print('  Juan Felipe Guerrero Urueña · Tecnológica del Oriente · 2026')
    print('█' * 78)

    # ── 1. Generar dataset con sesgo histórico ────────────────────────────
    print('\n[1/4] Generando dataset con sesgo estructural...')
    df = generar_dataset_sesgado(n=600)
    print(f'      Registros generados:     {len(df)}')
    print(f'      Tasa contratación global: {df["contratado"].mean():.3f}')
    print(f'      Tasa contratación M:      {df[df.genero==0]["contratado"].mean():.3f}')
    print(f'      Tasa contratación F:      {df[df.genero==1]["contratado"].mean():.3f}')

    # ── 2. División train / test ──────────────────────────────────────────
    print('\n[2/4] División train/test (75/25 estratificada)...')
    X = df.drop(columns=['contratado'])
    y = df['contratado']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SEED,
    )

    # ── 3. Modelo SIN mitigación ──────────────────────────────────────────
    print('\n[3/4] Entrenando modelo SIN mitigación...')
    modelo_base = RandomForestClassifier(
        n_estimators=100, max_depth=8, min_samples_split=5,
        random_state=SEED, n_jobs=-1,
    )
    resultado_antes = auditar_modelo(
        modelo_base, X_train, X_test, y_train, y_test,
        X_test['genero'], X_test['edad'],
        'AUDITORÍA SIN MITIGACIÓN  (modelo del caso analizado)',
    )

    # ── 4. Modelo CON reweighing ─────────────────────────────────────────
    print('\n[4/4] Aplicando estrategia de mitigación: REWEIGHING...')

    # Reweighing aplicado conjuntamente sobre género Y rango etario
    # para mitigar ambos sesgos identificados en la auditoría inicial.
    edad_train_bin = (X_train['edad'] > 45).astype(int).values
    grupo_interseccional = X_train['genero'].values * 2 + edad_train_bin
    # grupo_interseccional: 0=H≤45, 1=H>45, 2=M≤45, 3=M>45
    pesos = calcular_pesos_reweighing(X_train, y_train, grupo_interseccional)
    print(f'      Pesos calculados (intersección género × edad)')
    print(f'      Rango de pesos — min: {pesos.min():.3f}, max: {pesos.max():.3f}')

    modelo_mitigado = RandomForestClassifier(
        n_estimators=100, max_depth=8, min_samples_split=5,
        random_state=SEED, n_jobs=-1,
    )
    resultado_despues = auditar_modelo(
        modelo_mitigado, X_train, X_test, y_train, y_test,
        X_test['genero'], X_test['edad'],
        'AUDITORÍA CON REWEIGHING  (estrategia propuesta en el informe)',
        sample_weight=pesos,
    )

    # ── 5. Resumen comparativo ─────────────────────────────────────────────
    print('\n' + '═' * 78)
    print('  RESUMEN COMPARATIVO — TRADE-OFFS DOCUMENTADOS')
    print('═' * 78)

    acc_a, mg_a, me_a = resultado_antes
    acc_d, mg_d, me_d = resultado_despues

    delta_acc = (acc_d - acc_a) * 100
    delta_dir_gen = mg_d['disparate_impact_ratio'] - mg_a['disparate_impact_ratio']
    delta_dir_edad = me_d['disparate_impact_ratio'] - me_a['disparate_impact_ratio']

    print(f'\n  Cambio en Accuracy:                   {delta_acc:+.2f} puntos porcentuales')
    print(f'  Cambio en DIR — Género:               {delta_dir_gen:+.3f}')
    print(f'  Cambio en DIR — Edad:                 {delta_dir_edad:+.3f}')

    print('\n  Conclusión cuantitativa:')

    # Conclusión adaptativa según los resultados reales
    if delta_acc < -0.5:
        print(f'  El reweighing costó {abs(delta_acc):.2f} puntos porcentuales de precisión global,')
        print('  trade-off documentado por Barocas et al. (2023) entre precisión y equidad.')
    elif delta_acc > 0.5:
        print(f'  El reweighing mejoró la precisión global en {delta_acc:.2f} pp, indicando que')
        print('  el sesgo original era contraproducente para el aprendizaje del modelo.')
    else:
        print('  El reweighing mantuvo la precisión global con cambios menores a 0.5 pp,')
        print('  indicando que el ajuste de equidad no comprometió el desempeño técnico.')

    mejoras_equidad = []
    if delta_dir_gen > 0.01:
        mejoras_equidad.append(f'género ({delta_dir_gen:+.3f})')
    if delta_dir_edad > 0.01:
        mejoras_equidad.append(f'edad ({delta_dir_edad:+.3f})')

    if mejoras_equidad:
        print(f'  Las métricas de equidad mejoraron en: {", ".join(mejoras_equidad)}.')

    cumple_genero = mg_d['cumple_regla_4_5']
    cumple_edad = me_d['cumple_regla_4_5']
    if cumple_genero and cumple_edad:
        print('  Ambos grupos protegidos cumplen ahora la regla del 4/5 (DIR ≥ 0.80).')
    elif cumple_genero or cumple_edad:
        grupo_ok = 'género' if cumple_genero else 'edad'
        grupo_pendiente = 'edad' if cumple_genero else 'género'
        print(f'  El grupo "{grupo_ok}" cumple la regla del 4/5, pero "{grupo_pendiente}" requiere')
        print('  intervenciones adicionales (eliminación de proxies, ajuste de umbrales).')
    else:
        print('  Ningún grupo cumple aún la regla del 4/5: el sesgo es lo suficientemente')
        print('  estructural como para requerir intervenciones combinadas más allá de reweighing.')

    print('  Este hallazgo refuerza el argumento del informe sobre la necesidad de')
    print('  combinar estrategias técnicas con auditorías continuas y supervisión humana.')

    # ── 6. Visualización ───────────────────────────────────────────────────
    print('\n[BONUS] Generando visualización dark-mode...')
    generar_grafico_comparativo(resultado_antes, resultado_despues, 'audit_results.png')

    print('\n' + '█' * 78)
    print('  AUDITORÍA COMPLETADA')
    print('█' * 78 + '\n')


if __name__ == '__main__':
    main()
