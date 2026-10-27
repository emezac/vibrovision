# Fundamentos matemáticos de VibroVision 3

Distinguimos identidades bajo un modelo, incertidumbre estimada y observaciones. El reporte general explica el impacto sin exigir estas derivaciones.

## 1. Modelo e identificabilidad

Por eje y durante una ventana aproximadamente estacionaria:

\[
M\ddot x+C\dot x+Kx=f,\quad K=\operatorname{diag}(k_g)+L_W,\quad y=Sx+\eta.
\]

M,S son diagonales positivas; K es simétrica con acoplamientos negativos fuera de la diagonal. S representa magnificación por ROI, no mezcla arbitraria de ejes. Se necesitan suficientes coordenadas, modos observables y excitación rica. Partes ocultas, movimiento fuera del plano, excitación coloreada y no estacionariedad pueden invalidar el modelo.

SSI-cov estima polos y formas mediante covarianzas retardadas, estabilización entre órdenes y asociación por frecuencia/MAC. Antes del SVD se normalizan los canales por su desviación y después se restauran las coordenadas de las formas: este preacondicionamiento evita que una magnificación fija cambie la identificación numérica. Si faltan modos no reconstruimos una rigidez completa. En píxeles:

\[
B_y=SM^{-1}KS^{-1}.
\]

Sumar filas representa k_g/m sólo con una escala espacial común. `common_scale=True` es una hipótesis declarada/calibrada; sin ella se habla de rigidez dinámica local.

## 2. Operador invariante

Definimos D_ii=(B_y)_ii y D_ij=−√((B_y)_ij(B_y)_ji) para i≠j. Entonces:

\[
(B_y)_{ij}(B_y)_{ji}=\frac{s_iK_{ij}}{m_is_j}\frac{s_jK_{ji}}{m_js_i}
=\frac{K_{ij}^2}{m_im_j},
\quad\boxed{D=M^{-1/2}KM^{-1/2}}.
\]

La diagonal también coincide. Los autovalores de D son frecuencias naturales al cuadrado, en s^(-2). La cancelación exige acoplamientos recíprocos del modelo y escalas positivas. No identifica masas, fuerzas absolutas ni todos los montajes si se desconocen escalas relativas.

`invariant_operator` verifica signos, positividad y semejanza diagonal: QB≈BᵀQ, con Q=MS^(-2) en el modelo ideal. Ajusta log q mediante diferencias de log de acoplamientos recíprocos, ponderadas por su magnitud relativa. Informa residuos de ciclos, proyección y semejanza. La tolerancia 0,10 es un control configurable de compatibilidad, no un nivel de confianza. Se rechazan desviaciones grandes; no se ocultan tras simetrizar.

En T=[Ψ;ΨΛ], A=TΛT^(-1), el bloque superior [0,I] se impone algebraicamente. El campo histórico `consistency` no valida el modelo. El condicionamiento se evalúa adimensionalizando velocidades por una frecuencia de referencia y se rechaza por encima de 10^6.

## 3. Tres espectros distintos

El cociente de Rayleigh generalizado, usando L_W semidefinido positivo y vectores coordenados, demuestra:

\[
\min_i\frac{k_{g,i}}{m_i}\le\lambda_{min}(D)
\le\min_i\frac{k_{g,i}+\sum_jw_{ij}}{m_i}.
\]

La implementación sólo utiliza la interpretación inferior por sumas de filas con escala común. La demostración es mecánica e independiente del artículo financiero.

Los polos de A describen oscilación y amortiguamiento: λ=−ζω_n+iω_n√(1−ζ²). La tasa r=−Re λ tiene unidades s^(-1); r_min=ζω_min exige ζ común. El simulador impone esa hipótesis al construir C. Con amortiguamiento viscoso fijo, reducir rigidez puede dejar intacta la parte real de los polos.

Para M,K,C positivas, E=(ẋᵀMẋ+xᵀKx)/2 satisface Ė=−ẋᵀCẋ. Con C semidefinida deben excluirse modos no amortiguados. La covarianza bajo fuerza aleatoria obedece AP+PAᵀ+GQ_fGᵀ=0 si A es Hurwitz.

El Laplaciano normalizado de los acoplamientos tiene λ₂ adimensional y mide conectividad relativa. L_norm(cW)=L_norm(W): no detecta la escala absoluta de rigidez. Los nodos aislados aportan ceros y se cuentan componentes. Para n≤12 evaluamos todos los cortes no equivalentes; para n mayor, barrido de v₂/√grado. La partición es descriptiva si las aristas son inciertas. Cheeger relaciona h²/2≤λ₂≤2h bajo las hipótesis de grafo no negativo. [Chung](https://fanchung.ucsd.edu/research/outline.html).

## 4. Sensibilidad y pérdida finita

Para modos simples con Kφ_r=ω_r²Mφ_r y φ_rᵀMφ_r=1:

\[
\partial\omega_r^2/\partial k_{g,i}=\phi_{r,i}^2,
\qquad\partial\omega_r^2/\partial w_{ij}=(\phi_{r,i}-\phi_{r,j})^2.
\]

Se demuestra diferenciando y multiplicando por φ_rᵀ. Explica por qué un modo puede ser casi ciego a una falla local. Es de primer orden con M fijo; localizar muchos parámetros exige rango suficiente o hipótesis como daño disperso. `modal_sensitivities` requiere M/K calibradas.

Para K'=K−δbbᵀ, δ≥0, usar b=e_i−e_j para unión y e_i para montaje. Con K>0 y R_b=bᵀK^(-1)b:

\[
K'>0\iff\delta R_b<1,
\]
\[
(K')^{-1}=K^{-1}+\frac{\delta}{1-\delta R_b}K^{-1}bb^TK^{-1},
\quad\Delta\log\det K=\log(1-\delta R_b).
\]

Son Sherman–Morrison y el lema del determinante. R_b es flexibilidad estática/resistencia efectiva con conexiones a tierra. `stiffness_loss` lo implementa y se comprueba contra inversión directa. No se estima una pérdida física absoluta desde los dos clips actuales.

## 5. Incertidumbre y abstención

Se remuestrean contribuciones de covarianza por bloques multicanal, centradas en la covarianza original. No se barajan frames. Cada bloque dura al menos cuatro segundos y supera el soporte de retardos; se exigen seis bloques, ocho réplicas y éxito físico en ≥60 % de ellas, con mínimo de seis válidas. Cada réplica conserva asociación modal.

El percentil 95 de ||D_b−D_hat||₂ es un **radio empírico condicionado al modelo, sin cobertura certificada**. Excluye sesgo estructural y no estacionariedad; descartar réplicas inválidas puede sesgarlo. Es necesario evaluar sensibilidad al bloque, selección y cobertura independiente.

Si se dispone de una cota válida ||D_hat−D||₂≤ε, Weyl implica |λ_r(D_hat)−λ_r(D)|≤ε. Entre ventanas se comparan cambios con ε_0+ε_t. Davis–Kahan controla rotación de subespacios en orden ε/g, con separación g. [Yu, Wang y Samworth](https://arxiv.org/abs/1405.0680).

Usamos esas relaciones condicionalmente con radios estimados. Para localizar exigimos cambio del operador y de la entrada relevante superior a la suma de radios en un mismo eje con soporte. Se publican los ejes utilizados; uno puede bastar si el otro es incierto. Una partición modal inestable no se presenta como prueba de localización. Para modos degenerados debe compararse el subespacio, no una base arbitraria; PR no cuenta universalmente comunidades.

Salidas: `nominal`, `abstain`, `collect_more_data`, `change_unlocalized`, `localized_change`. Nominal no demuestra ausencia de toda falla. El agente confirma temporalmente y pide aprobación humana. [Incertidumbre específica de SSI](https://svibs.com/wp-content/uploads/2023/11/2011_1.pdf).

## 6. Condición para la pendiente α

Para P=P_0+qP_d, dos contribuciones de ley de potencia:

\[
\alpha_{local}=\frac{\alpha_0P_0+\alpha_dqP_d}{P_0+qP_d}.
\]

La pendiente cae si lo añadido es más plano; puede subir si es más inclinado. Con x=log f y bins/pesos fijos:

\[
\frac{d\hat\alpha}{dq}=-\frac{\operatorname{Cov}_w(x,P_d/(P_0+qP_d))}{\operatorname{Var}_w(x)}.
\]

Si P_d/P_0 crece con f, la derivada es no positiva. Máscaras de tonos/SNR cambiantes limitan la aplicación a la implementación; una resonancia no es una ley de potencia global. No afirmamos que toda falla haga caer α.

La invariancia a ganancia constante vale con selección fija. El intervalo WLS es aproximado: la comprobación de cobertura en 200 señales ideales excluye tracker, compresión y resonancias. No se vende como cobertura universal del 95 %.

## 7. Espectro conjunto e información de imagen

Se estima S_y(f) conservando relaciones de fase, y se comparan autovalores generalizados S_t v=λ S_0 v. Sin regularizar son invariantes al mismo cambio invertible de coordenadas. Usamos shrinkage diagonal relativo del 2 % y piso diagonal de referencia de 10^(-8); comprobamos invariancia a ganancias diagonales fijas, no a congruencia arbitraria de esa regularización. La dispersión de log λ elimina una sola ganancia global. Es descriptiva, sin umbral de alarma calibrado. Se interpolan matrices complejas a una banda común cuando los fps difieren.

En S_y=H S_f H*+N, una excitación arbitraria impide separar estructura de fuerza unívocamente. Hace falta controlar/medir excitación o asumir estabilidad aproximada de su espectro. Hablamos de candidato de forzamiento, no causalidad universal. Los ejes x/y del aro no son piezas distintas.

Para imagen I=aT(x−u)+b+η, Jacobiano J y nuisance Z=[T,1], el complemento de Schur es:

\[
F_{eff}=J^T\Sigma^{-1}J-J^T\Sigma^{-1}Z(Z^T\Sigma^{-1}Z)^+Z^T\Sigma^{-1}J.
\]

`nuisance_fisher` lo implementa y se verifica que eliminar nuisance no añade información. Es una comprobación algebraica, no un reemplazo ya calibrado de todo el tracker. El modelo IC de ruido conserva su alcance de primer orden y ruido blanco; no es exacto con codec. Restar una referencia común introduce correlación entre ROIs. La selección óptima de ROIs por información mecánica queda como extensión posterior.

## 8. Relación con el artículo y evidencia

El manuscrito financiero motivó operadores, conectividad y perturbaciones. No dependemos de su teorema 2.8: el máximo autovalor de adyacencia normalizada es 1, y usar 1−Δ requiere excluir la dirección estacionaria. Tampoco trasladamos una equivalencia universal gap–MPR–riesgo–criticidad. En OU, σ²(1−exp(−2rT))/(2r) tiende a σ²T para T fijo y r→0; una divergencia 1/r requiere otro régimen.

`python eval/run_math_validation.py --real` ejecuta álgebra, video sintético nuevo a través de OpenCV/SSI, cobertura ideal de α y comparación nueva de clips. Guarda versiones, hashes, resultados y figura. No repite toda la evaluación v2 ni certifica desempeño industrial v3. Faltan, entre otros, excitación coloreada, modos ocultos/cercanos, amortiguamiento no proporcional, cambios de cámara, daño múltiple, estudio de bloques y referencia independiente.

## 9. Alcance del laboratorio de videos MIT

`video_lab.py` admite Lucas–Kanade incremental con control de ida/vuelta e IC con plantilla fija para movimientos pequeños. Conserva sólo cuadros consecutivos y las series medidas. La fracción de características supervivientes (LK) o el residuo fotométrico normalizado (IC), junto con un detector de discontinuidades, constituyen controles de seguimiento, no intervalos de confianza. LK puede acumular deriva; una referencia opcional sólo resta traslación común e introduce correlación entre canales.

Para las series centradas por Welch, la matriz espectral hermítica admite S(f)=U(f)Λ(f)U(f)*. Se presenta el autovector asociado al mayor autovalor, normalizado por su componente máxima y con fase global fijada. Es una **forma operacional descriptiva**; su interpretación como modo estructural requiere hipótesis adicionales sobre excitación, separación modal y modelo. La coherencia cuadrática entre i,j es |S_ij|²/(S_ii S_jj), estimada con al menos tres segmentos Welch solapados; esos segmentos no son ensayos independientes. Coherencia alta no demuestra causalidad ni identifica daño.

Dos movimientos de igual amplitud y espectro por canal pueden tener fases opuestas entre canales. Una prueba del laboratorio conserva las PSD individuales al invertir el signo de un canal y comprueba que cambia la forma operacional. Eso verifica información que un indicador escalar por región no retiene. No establece superioridad universal frente a todos los métodos modales existentes.

En la clase del MIT se cambian las condiciones iniciales, no se documenta pérdida de rigidez. Por ello la salida estructural es siempre abstención para estos ejemplos. No se aplica la reconstrucción completa SSI/D a los fragmentos como si fueran una excitación estacionaria multicanal validada. La base de tiempo de adquisición no se infiere de un montaje o del FPS del archivo; sin su verificación se publican frecuencias en tiempo de reproducción.

### Montaje LISS suministrado por el usuario

La inspección local de `tres.mp4` y `cuatro.mp4` mostró un mismo montaje LISS de 141,24 s. Los hashes idénticos impiden tratarlos como ensayos independientes. El perfil usa sólo 100–110 s del puente original, con regiones de una estructura continua; no son coordenadas completas de un modelo de masas discretas. No se identifica D ni se infiere daño a partir de esta escena.

Se contrastan IC y LK sobre exactamente los mismos cuadros y regiones. Se informa correlación de series sin tendencia, RMS de su diferencia y coincidencia de los máximos espectrales dentro de dos bins. Se exige además correlación ≥0,8 en todas las regiones para declarar consistencia descriptiva. Ambos criterios son heurísticos, no una prueba estadística calibrada. El máximo espectral global se conserva aunque sea ancho y no supere un filtro de prominencia: su mayor potencia no demuestra una resonancia. El desacuerdo puede proceder de ruido, codec, deriva o distintas sensibilidades modales de los estimadores: no prueba por sí solo que uno sea correcto. Coincidencia tampoco certifica un modo físico. Las escenas amplificadas ×400 y el tubo ralentizado se excluyen del análisis cuantitativo de este perfil.
