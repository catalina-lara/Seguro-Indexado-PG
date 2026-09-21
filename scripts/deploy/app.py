"""
Simulador de activación — Seguro Agrícola Indexado de Café (Boyacá)

Consume el mismo deploy/config.json que produce Indice_parametrico.ipynb (coeficientes del
índice, umbrales, curva de payout, estadísticas por municipio) y las métricas del modelo que
exporta Modelado.ipynb — no hay lógica duplicada ni números escritos a mano en este archivo.
"""

import json
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st

# ------------------------------------------------------------------------------------
# Carga de configuración
# ------------------------------------------------------------------------------------

DEPLOY_DIR = Path(__file__).parent
CONFIG_PATH = DEPLOY_DIR / "config.json"
MODELO_PATH = DEPLOY_DIR / "modelo_rendimiento.pkl"
FEATURES_PATH = DEPLOY_DIR / "features_modelo.json"
PROMEDIOS_PATH = DEPLOY_DIR / "promedios_municipio_features.json"
HISTORICO_PATH = DEPLOY_DIR / "historico_prediccion.json"

st.set_page_config(
    page_title="Simulador de activación · Seguro Agrícola Indexado de Café",
    page_icon="☕",
    layout="wide",
)

# Paleta y tipografía acordadas: fondo papel / tinta oscura en modo claro, café tostado
# intenso / crema en modo oscuro. Streamlit solo aplica nuestro tema personalizado cuando el
# usuario tiene "Claro" seleccionado — si elige "Oscuro" en su menú, usa su propio tema por
# defecto. Por eso forzamos ambos modos con CSS + variables, detectando el modo activo con
# varios selectores (cubre distintas versiones de Streamlit) en vez de depender de su tema oscuro.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Work+Sans:wght@400;500;600&display=swap');

:root {
    --bg-app: #ECE4D3; --surface: #F5EFE3; --ink: #2B2620; --ink-soft: #6B5F4F;
    --brick-bg: #EADAD3; --brick-border: #A13D2E; --brick-text: #A13D2E;
    --leaf-bg: #DEE3D3; --leaf-border: #4B6043; --leaf-text: #4B6043;
    --accent: #A13D2E; --accent-hover: #7A2D22;
}

/* Modo oscuro: café tostado casi negro, texto crema cálido, acentos más saturados para que
   el rojo/verde "pop" contra un fondo oscuro en vez de verse apagados. */
html[data-theme="dark"], body[data-theme="dark"], .stApp[data-theme="dark"],
[data-testid="stAppViewContainer"][data-theme="dark"] {
    --bg-app: #19120D; --surface: #2A1E17; --ink: #F2E8D9; --ink-soft: #C9B49B;
    --brick-bg: #4A2118; --brick-border: #E8593F; --brick-text: #F0917C;
    --leaf-bg: #1C3226; --leaf-border: #5FBF8C; --leaf-text: #8FDBAF;
    --accent: #E8593F; --accent-hover: #F0917C;
}
@media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
        --bg-app: #19120D; --surface: #2A1E17; --ink: #F2E8D9; --ink-soft: #C9B49B;
        --brick-bg: #4A2118; --brick-border: #E8593F; --brick-text: #F0917C;
        --leaf-bg: #1C3226; --leaf-border: #5FBF8C; --leaf-text: #8FDBAF;
        --accent: #E8593F; --accent-hover: #F0917C;
    }
}

.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stSidebar"] {
    background-color: var(--bg-app) !important; color: var(--ink) !important;
}
html, body, [class*="css"] { font-family: 'Work Sans', sans-serif; color: var(--ink); }
h1, h2, h3, p, span, label, div { color: var(--ink); }
h1, h2, h3 { font-family: 'Fraunces', serif !important; font-weight: 500 !important; }

.stButton > button {
    background-color: var(--ink); border: none; border-radius: 3px;
}
.stButton > button, .stButton > button * { color: var(--bg-app) !important; }
.stButton > button:hover { background-color: var(--accent); }
.stButton > button:hover, .stButton > button:hover * { color: var(--surface) !important; }

[data-testid="stMetric"] {
    background-color: var(--surface) !important; border-radius: 3px; padding: 12px 16px;
    border: 1px solid var(--ink-soft);
}
[data-testid="stMetricLabel"], [data-testid="stMetricValue"], [data-testid="stMetricDelta"] { color: var(--ink) !important; }

.estado-seguro {
    border-radius: 8px; padding: 20px 24px; text-align: center; margin: 8px 0 4px;
}
.estado-seguro.activo { background: var(--brick-bg); border: 1.5px solid var(--brick-border); }
.estado-seguro.no-activo { background: var(--leaf-bg); border: 1.5px solid var(--leaf-border); }
.estado-seguro .titulo { font-family: 'Fraunces', serif; font-weight: 500; font-size: 20px; margin: 0 0 6px; }
.estado-seguro.activo .titulo { color: var(--brick-text); }
.estado-seguro.no-activo .titulo { color: var(--leaf-text); }
.estado-seguro .detalle { font-size: 14px; color: var(--ink); margin: 0; }

/* Sliders: forzar nuestro acento en vez del rojo por defecto de Streamlit (sobre todo en modo
   oscuro, donde nuestro tema personalizado no aplica). Cubrimos varios selectores porque
   BaseWeb no expone una sola clase estable entre versiones de Streamlit. */
div[data-baseweb="slider"] div[role="slider"] {
    background-color: var(--accent) !important; border-color: var(--accent) !important;
    box-shadow: none !important;
}
div[data-baseweb="slider"] > div > div:nth-child(2),
div[data-baseweb="slider"] div[data-testid="stSliderTrackFilled"] {
    background: var(--accent) !important;
}

/* Radio (Vista: Analista/Caficultor) — Streamlit 1.6x usa react-aria-components, con
   estado reflejado en el atributo data-selected sobre el propio nodo interactivo. */
[data-testid="stRadioOption"][data-selected] { color: var(--accent) !important; }
[data-testid="stRadioOption"][data-selected] div {
    background-color: var(--accent) !important; border-color: var(--accent) !important;
}

/* Selectbox (Municipio) — mismo cambio de librería; ya no usa data-baseweb="select". */
[data-testid="stSelectbox"] div {
    background-color: var(--surface) !important; border-color: var(--ink-soft) !important;
}
[data-testid="stSelectbox"] input {
    background-color: transparent !important; color: var(--ink) !important;
}
[data-testid="stSelectbox"] svg { color: var(--ink) !important; }

/* Lista desplegable del selectbox: testid real confirmado en el bundle instalado. */
[data-testid="stSelectboxVirtualDropdown"] { background-color: var(--surface) !important; }
[data-testid="stSelectboxVirtualDropdown"] [role="option"] {
    background-color: var(--surface) !important; color: var(--ink) !important;
}
[data-testid="stSelectboxVirtualDropdown"] [role="option"]:hover,
[data-testid="stSelectboxVirtualDropdown"] [role="option"][aria-selected="true"] {
    background-color: var(--accent) !important; color: var(--bg-app) !important;
}
</style>
""", unsafe_allow_html=True)

if not CONFIG_PATH.exists():
    st.error(
        f"No se encontró `{CONFIG_PATH.name}` junto a este archivo. Corre "
        f"`Modelado.ipynb` y luego `Indice_parametrico.ipynb` completos — la última celda "
        f"de cada uno deja los archivos que esta app necesita en `deploy/`."
    )
    st.stop()

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

# Colores de gráficas (Altair, fondo transparente): distintos para modo claro y oscuro,
# detectando el tema activo en cada render con st.context.theme.type. Si Streamlit aún no
# reporta el tema (primera carga), asumimos claro.
try:
    ES_OSCURO = st.context.theme.type == "dark"
except Exception:
    ES_OSCURO = False

if ES_OSCURO:
    CHART_COLORS = ["#E8B84B", "#7FB3D9"]  # Real (caramelo brillante), Predicho (azul cielo)
    CHART_GRID = "#C9B49B"
    CHART_TEXT = "#C9B49B"
else:
    CHART_COLORS = ["#8B5E34", "#3D6E8F"]  # Real (café tostado), Predicho (azul petróleo)
    CHART_GRID = "#6B5F4F"
    CHART_TEXT = "#6B5F4F"

# El panel de rendimiento (predicción interactiva, evolución anual, calibración) es opcional:
# solo requiere los 4 archivos que exporta la nueva sección de Modelado.ipynb. Si aún no los has
# generado, el resto de la app (índice y activación del seguro) sigue funcionando igual.
PANEL_RENDIMIENTO_DISPONIBLE = all(
    p.exists() for p in [MODELO_PATH, FEATURES_PATH, PROMEDIOS_PATH, HISTORICO_PATH]
)

if PANEL_RENDIMIENTO_DISPONIBLE:
    modelo_rendimiento = joblib.load(MODELO_PATH)
    with open(FEATURES_PATH, "r", encoding="utf-8") as f:
        FEATURES_INFO = json.load(f)
    with open(PROMEDIOS_PATH, "r", encoding="utf-8") as f:
        PROMEDIOS_MUNICIPIO = json.load(f)
    with open(HISTORICO_PATH, "r", encoding="utf-8") as f:
        HISTORICO = pd.DataFrame(json.load(f))

# ------------------------------------------------------------------------------------
# Traducción a lenguaje llano (misma lógica que la versión HTML del simulador)
# ------------------------------------------------------------------------------------

LABELS = {
    "temp_media_anual_anomalia": ("Temperatura del año", "promedio anual"),
    "pdsi_media_anual": ("Nivel de sequía (promedio del año)", "escala estándar de sequía de Palmer"),
    "pdsi_min_anual": ("Peor momento de sequía del año", "el mes más seco, misma escala"),
    "soil_moisture_media": ("Humedad del suelo", "promedio del año"),
    "precip_anual_mm_anomalia": ("Lluvia del año (CHIRPS)", "acumulado anual"),
    "pr_anual_mm_anomalia": ("Lluvia del año (TerraClimate)", "acumulado anual, otra fuente satelital"),
    "balance_hidrico_anual_anomalia": ("Agua disponible en el suelo", "lluvia menos lo que se evapora en el año"),
}

DESCRIPTORES = {
    "temp_media_anual_anomalia": ("anomaly", [
        "Mucho más frío de lo normal", "Más frío de lo normal", "Temperatura normal",
        "Más caliente de lo normal", "Mucho más caliente de lo normal"]),
    "precip_anual_mm_anomalia": ("anomaly", [
        "Mucha menos lluvia de lo normal", "Menos lluvia de lo normal", "Lluvia normal",
        "Más lluvia de lo normal", "Mucha más lluvia de lo normal"]),
    "pr_anual_mm_anomalia": ("anomaly", [
        "Mucha menos lluvia de lo normal", "Menos lluvia de lo normal", "Lluvia normal",
        "Más lluvia de lo normal", "Mucha más lluvia de lo normal"]),
    "balance_hidrico_anual_anomalia": ("anomaly", [
        "Suelo mucho más seco de lo normal", "Suelo más seco de lo normal", "Balance normal",
        "Suelo más húmedo de lo normal", "Suelo mucho más húmedo de lo normal"]),
    "soil_moisture_media": ("relative", [
        "Humedad muy baja", "Humedad baja", "Humedad normal", "Humedad alta", "Humedad muy alta"]),
    "pdsi_media_anual": ("pdsi", None),
    "pdsi_min_anual": ("pdsi", None),
}


def descriptor_pdsi(v: float) -> str:
    if v >= 4:
        return "Húmedo extremo"
    if v >= 3:
        return "Muy húmedo"
    if v >= 2:
        return "Húmedo moderado"
    if v >= 1:
        return "Ligeramente húmedo"
    if v >= -1:
        return "Normal"
    if v >= -2:
        return "Sequía ligera"
    if v >= -3:
        return "Sequía moderada"
    if v >= -4:
        return "Sequía severa"
    return "Sequía extrema"


def calcular_z(v: dict, val: float, muni_data: dict | None) -> float:
    """Reconstruye la misma anomalía estandarizada de dos pasos que usa el notebook:
    1) valor absoluto -> anomalía por municipio (o promedio departamental si no hay municipio)
    2) esa anomalía -> z-score departamental (global_zscore), sin aplicar aún el signo de impacto.
    """
    if v["es_anomalia"]:
        media_abs = v["abs_rango"]["mean"]
        std_abs = v["abs_rango"]["std"]
        base = (muni_data or {}).get("valores_base", {}).get(v["columna_base"])
        if base and "media_hist" in base:
            media_abs = base["media_hist"]
            std_abs = base["std_hist"]
        anomalia_equivalente = (val - media_abs) / std_abs
    else:
        anomalia_equivalente = val
    return (anomalia_equivalente - v["global_zscore"]["mean"]) / v["global_zscore"]["std"]


def descriptor_texto(v: dict, val: float, muni_data: dict | None) -> str:
    tipo, palabras = DESCRIPTORES.get(v["nombre"], (None, None))
    if tipo is None:
        return ""
    if tipo == "pdsi":
        return descriptor_pdsi(val)
    z = calcular_z(v, val, muni_data)
    if z < -1.5:
        idx = 0
    elif z < -0.5:
        idx = 1
    elif z <= 0.5:
        idx = 2
    elif z <= 1.5:
        idx = 3
    else:
        idx = 4
    return palabras[idx]


def valor_inicial(v: dict, muni_data: dict | None) -> float:
    base = (muni_data or {}).get("valores_base", {}).get(v["columna_base"])
    if not base:
        return v["abs_rango"]["mean"]
    return base["media_hist"] if v["es_anomalia"] else base["valor_tipico"]


def subindice(variables: list[dict], state: dict, muni_data: dict | None) -> float:
    zs = [calcular_z(v, state[v["nombre"]], muni_data) * v["signo"] for v in variables]
    return sum(zs) / len(zs)


def polyval(coefs: list[float], x: float) -> float:
    """coefs = [a, b, c] para a*x^2 + b*x + c (orden de np.polyfit)."""
    r = 0.0
    for c in coefs:
        r = r * x + c
    return r


# ------------------------------------------------------------------------------------
# Sidebar: vista y municipio
# ------------------------------------------------------------------------------------

st.sidebar.title("☕ Seguro Indexado · Boyacá")
vista = st.sidebar.radio("Vista", ["Analista", "Caficultor"], horizontal=True)

municipios = sorted(CONFIG.get("municipios", {}).keys())
opciones_muni = ["Promedio departamental (sin municipio)"] + municipios
municipio_sel = st.sidebar.selectbox("Municipio", opciones_muni)
muni_data = CONFIG["municipios"].get(municipio_sel) if municipio_sel in municipios else None

if muni_data:
    if muni_data["tiene_estacion"]:
        st.sidebar.caption(
            f"✅ {municipio_sel} tiene estación pluviométrica propia (dato directo, no interpolado) "
            f"· {muni_data['n_anios']} años en el panel."
        )
    else:
        st.sidebar.caption(
            f"⚠️ {municipio_sel} no tiene estación propia; el clima proviene de interpolación (GEE) "
            f"· {muni_data['n_anios']} años en el panel."
        )

st.title("Simulador de activación")
st.caption("Seguro Agrícola Indexado de Café · Boyacá")

# ------------------------------------------------------------------------------------
# KPIs del modelo (solo vista Analista)
# ------------------------------------------------------------------------------------

m = CONFIG["modelo"]

if vista == "Analista":
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R² (LOYO)", f"{m['r2']:.3f}", "Cumple N2" if m["cumple_n2"] else "No cumple N2", delta_color="off")
    c2.metric("RMSE", f"{m['rmse']:.3f} t/ha", "Cumple D1" if m["cumple_d1"] else "No cumple D1", delta_color="off")
    c3.metric("MAE", f"{m['mae']:.3f} t/ha", "Cumple D1" if m["cumple_d1"] else "No cumple D1", delta_color="off")
    c4.metric("Observaciones", f"{m['n_observaciones']}")
    st.divider()

    # --------------------------------------------------------------------------------
    # Panel de rendimiento: predicción interactiva, evolución anual, calibración,
    # comparación entre municipios. Solo vista Analista — el caficultor no necesita esto.
    # --------------------------------------------------------------------------------
    if not PANEL_RENDIMIENTO_DISPONIBLE:
        st.info(
            "El panel de rendimiento (predicción interactiva, evolución anual, calibración) "
            "aparece aquí una vez corras la nueva sección de exportación al final de "
            "`Modelado.ipynb` (modelo_rendimiento.pkl, features_modelo.json, "
            "promedios_municipio_features.json, historico_prediccion.json)."
        )
    else:
        st.subheader("Modelo de rendimiento")
        col_pred, col_evol = st.columns([1, 1.3])

        with col_pred:
            st.markdown("**Predicción interactiva**")
            municipio_modelo = municipio_sel if municipio_sel in PROMEDIOS_MUNICIPIO else None
            promedios = PROMEDIOS_MUNICIPIO.get(
                municipio_modelo,
                pd.DataFrame(PROMEDIOS_MUNICIPIO).T.mean().to_dict(),  # promedio departamental
            )

            valores_features = dict(promedios)  # arranca con el promedio del municipio/departamento
            for feat in FEATURES_INFO["top_shap"]:
                if feat not in promedios:
                    continue
                label, sub = LABELS.get(feat, (feat, ""))
                serie_feat = pd.Series({k: v.get(feat) for k, v in PROMEDIOS_MUNICIPIO.items()}).dropna()
                lo, hi = float(serie_feat.min()), float(serie_feat.max())
                if lo == hi:
                    hi = lo + 1.0
                val = st.slider(
                    label, min_value=lo, max_value=hi, value=float(promedios[feat]),
                    help=sub or feat, key=f"pred_{feat}",
                )
                valores_features[feat] = val

            if st.button("Predecir rendimiento"):
                x = pd.DataFrame([[valores_features.get(f, 0.0) for f in FEATURES_INFO["features"]]],
                                  columns=FEATURES_INFO["features"])
                pred_rendimiento = modelo_rendimiento.predict(x)[0]
                st.metric("Rendimiento estimado", f"{pred_rendimiento:.3f} t/ha")
                st.caption(
                    "Las variables no mostradas como slider quedan fijas en el promedio del "
                    f"municipio elegido ({municipio_sel})." if municipio_modelo else
                    "Las variables no mostradas como slider quedan fijas en el promedio departamental."
                )

        with col_evol:
            st.markdown(f"**Evolución anual del rendimiento — {municipio_sel if municipio_modelo else 'todos los municipios'}**")
            hist_muni = HISTORICO[HISTORICO["municipio"] == municipio_sel] if municipio_modelo else HISTORICO.groupby("anio")[["real", "pred"]].mean().reset_index()
            hist_muni = hist_muni.sort_values("anio")[["anio", "real", "pred"]]
            hist_long = hist_muni.melt(id_vars="anio", value_vars=["real", "pred"], var_name="serie", value_name="rendimiento")
            hist_long["serie"] = hist_long["serie"].map({"real": "Real", "pred": "Predicho (LOYO)"})

            chart_evol = alt.Chart(hist_long).mark_line(strokeWidth=2.5).encode(
                x=alt.X("anio:O", title=None),
                y=alt.Y("rendimiento:Q", title="t/ha"),
                color=alt.Color(
                    "serie:N", title=None,
                    scale=alt.Scale(domain=["Real", "Predicho (LOYO)"], range=CHART_COLORS),
                    legend=alt.Legend(orient="bottom"),
                ),
            ).properties(height=280, background="transparent").configure_axis(
                gridColor=CHART_GRID, domainColor=CHART_GRID, tickColor=CHART_GRID, labelColor=CHART_TEXT, titleColor=CHART_TEXT,
            ).configure_legend(labelColor=CHART_TEXT, titleColor=CHART_TEXT).configure_view(strokeWidth=0)
            st.altair_chart(chart_evol, use_container_width=True)

        col_calib, col_muni = st.columns(2)
        with col_calib:
            st.markdown("**Predicción vs. real (calibración)**")
            lo = float(min(HISTORICO["real"].min(), HISTORICO["pred"].min()))
            hi = float(max(HISTORICO["real"].max(), HISTORICO["pred"].max()))
            diagonal = pd.DataFrame({"real": [lo, hi], "pred": [lo, hi]})
            base = alt.Chart(HISTORICO).mark_circle(size=45, opacity=0.55, color=CHART_COLORS[0]).encode(
                x=alt.X("real:Q", title="real"), y=alt.Y("pred:Q", title="pred"),
            )
            linea_ref = alt.Chart(diagonal).mark_line(strokeDash=[3, 3], color=CHART_GRID).encode(x="real:Q", y="pred:Q")
            chart_calib = (base + linea_ref).properties(height=260, background="transparent").configure_axis(
                gridColor=CHART_GRID, domainColor=CHART_GRID, tickColor=CHART_GRID, labelColor=CHART_TEXT, titleColor=CHART_TEXT,
            ).configure_view(strokeWidth=0)
            st.altair_chart(chart_calib, use_container_width=True)
            st.caption(f"La dispersión respecto a la diagonal ideal refleja el R²={m['r2']:.3f} real del modelo — sin suavizar.")

        with col_muni:
            st.markdown("**Comparación entre municipios (rendimiento real)**")
            top_municipios = HISTORICO.groupby("municipio")["real"].mean().sort_values(ascending=False).head(10).index
            box_data = HISTORICO[HISTORICO["municipio"].isin(top_municipios)]
            chart_muni = alt.Chart(box_data).mark_boxplot(size=20, color=CHART_COLORS[1]).encode(
                x=alt.X("municipio:N", sort=list(top_municipios), title=None),
                y=alt.Y("real:Q", title="t/ha"),
            ).properties(height=260, background="transparent").configure_axis(
                gridColor=CHART_GRID, domainColor=CHART_GRID, tickColor=CHART_GRID, labelColor=CHART_TEXT, titleColor=CHART_TEXT,
            ).configure_view(strokeWidth=0)
            st.altair_chart(chart_muni, use_container_width=True)

        st.divider()

# ------------------------------------------------------------------------------------
# Sliders: condiciones climáticas
# ------------------------------------------------------------------------------------

st.subheader("Condiciones climáticas del año")

state: dict[str, float] = {}
col_sequia, col_exceso = st.columns(2)


def render_grupo(container, titulo: str, variables: list[dict]):
    with container:
        st.markdown(f"**{titulo}**")
        for v in variables:
            nombre = v["nombre"]
            label, sub = LABELS.get(nombre, (nombre, ""))
            default = valor_inicial(v, muni_data)
            paso = (v["abs_rango"]["max"] - v["abs_rango"]["min"]) / 200
            val = st.slider(
                label,
                min_value=float(v["abs_rango"]["min"]),
                max_value=float(v["abs_rango"]["max"]),
                value=float(default),
                step=float(paso) if paso > 0 else 0.01,
                help=sub,
                key=f"slider_{nombre}",
            )
            state[nombre] = val
            unidad = v.get("unidad", "")
            texto = descriptor_texto(v, val, muni_data)
            if vista == "Analista":
                st.caption(f"{texto} · {val:.1f} {unidad}".strip())
            else:
                st.caption(texto)


render_grupo(col_sequia, "Temperatura y sequía", CONFIG["sequia"]["variables"])
render_grupo(col_exceso, "Lluvia y humedad", CONFIG["exceso"]["variables"])

# ------------------------------------------------------------------------------------
# Cálculo del índice y resultado
# ------------------------------------------------------------------------------------

ic_sequia = subindice(CONFIG["sequia"]["variables"], state, muni_data)
ic_exceso = subindice(CONFIG["exceso"]["variables"], state, muni_data)

activado_sequia = ic_sequia >= CONFIG["sequia"]["umbral"]
activado_exceso = ic_exceso >= CONFIG["exceso"]["umbral"]
activado = activado_sequia or activado_exceso

sev_sequia = max(ic_sequia - CONFIG["sequia"]["umbral"], 0.0)
sev_exceso = max(ic_exceso - CONFIG["exceso"]["umbral"], 0.0)
severidad = max(sev_sequia, sev_exceso)

payout = 0.0
if activado:
    payout = polyval(CONFIG["payout"]["coeficientes_polinomio_grado2"], severidad)
    payout = min(max(payout, 0.0), CONFIG["payout"]["payout_max"])

st.divider()
st.subheader("Estado del seguro")

if vista == "Caficultor":
    if activado:
        st.markdown(
            '<div class="estado-seguro activo">'
            '<p class="titulo">🔴 Tu cosecha está en riesgo este año</p>'
            '<p class="detalle">El seguro se activó según las condiciones del clima. '
            'El pago se calcula según qué tan fuerte fue el evento.</p></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="estado-seguro no-activo">'
            '<p class="titulo">🟢 Tu cosecha está bien este año</p>'
            '<p class="detalle">Las condiciones del clima no alcanzaron el nivel que activa el seguro.</p></div>',
            unsafe_allow_html=True,
        )
else:
    if activado:
        motivo = (
            "Se superó el umbral de sequía y el de exceso hídrico."
            if activado_sequia and activado_exceso
            else "Se superó el umbral de sequía." if activado_sequia
            else "Se superó el umbral de exceso hídrico."
        )
        st.markdown(
            f'<div class="estado-seguro activo">'
            f'<p class="titulo">🔴 Seguro activado</p>'
            f'<p class="detalle">{motivo}<br>'
            f'<strong>Pago estimado:</strong> {payout:.3f} (escala de pérdida normalizada, '
            f'máx. {CONFIG["payout"]["payout_max"]:.2f})</p></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="estado-seguro no-activo">'
            '<p class="titulo">🟢 Seguro no activado</p>'
            '<p class="detalle">Ninguno de los dos subíndices climáticos superó su umbral de activación.</p></div>',
            unsafe_allow_html=True,
        )

    tech_df = pd.DataFrame([
        {"Subíndice": "IC sequía", "Valor": round(ic_sequia, 3), "Umbral": round(CONFIG["sequia"]["umbral"], 3), "Severidad": round(sev_sequia, 3)},
        {"Subíndice": "IC exceso", "Valor": round(ic_exceso, 3), "Umbral": round(CONFIG["exceso"]["umbral"], 3), "Severidad": round(sev_exceso, 3)},
    ])
    st.table(tech_df.set_index("Subíndice"))

    # ------------------------------------------------------------------------------
    # Explicabilidad del modelo (SHAP)
    # ------------------------------------------------------------------------------
    st.subheader("Explicabilidad del modelo (SHAP)")
    shap_df = pd.Series(m["shap_top10"]).sort_values(ascending=True).reset_index()
    shap_df.columns = ["variable", "importancia"]
    chart_shap = alt.Chart(shap_df).mark_bar(color=CHART_COLORS[0]).encode(
        x=alt.X("importancia:Q", title=None),
        y=alt.Y("variable:N", sort=None, title=None),
    ).properties(height=320, background="transparent").configure_axis(
        gridColor=CHART_GRID, domainColor=CHART_GRID, tickColor=CHART_GRID, labelColor=CHART_TEXT, titleColor=CHART_TEXT,
    ).configure_view(strokeWidth=0)
    st.altair_chart(chart_shap, use_container_width=True)

    st.caption(
        f"El modelo de rendimiento base ({m['iteracion_ganadora']} + {m['modelo_ganador']}, "
        f"R²={m['r2']:.3f}) no alcanza los criterios N2, D1, D2 ni D3 de la tabla de requerimientos "
        f"(desempeño mínimo, error absoluto, interpretabilidad SHAP top-3={m['shap_top3_pct']:.1%}, "
        f"y estabilidad temporal, variación de R² por año={m['d3_std_r2_anual']:.2f}). "
        f"D4 (trade-off lineal vs. no lineal) sí se cumple usando "
        f"{m.get('d4_modelo_lineal_usado', 'el mejor modelo lineal')} como base de comparación en vez de "
        f"la Regresión Lineal ordinaria. El índice climático hereda estas limitaciones — ver "
        f"`Modelado.ipynb` e `Indice_parametrico.ipynb` para el detalle completo."
    )

st.divider()
st.caption(
    f"Índice {CONFIG['version']} · generado {CONFIG['generado']} · "
    f"modelo {m['iteracion_ganadora']}+{m['modelo_ganador']} · "
    "Prototipo académico — no constituye asesoría actuarial ni oferta de seguro."
)
