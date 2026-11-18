import joblib
import pandas as pd
import streamlit as st

# A mesma inferência do app.py (o mesmo modelo, os mesmos campos e o mesmo predict),
# numa tela com mais recursos do Streamlit: barra lateral, abas, gráfico, histórico e download.

st.set_page_config(page_title="Preço de casas em Ames", page_icon="🏠", layout="wide")
st.title("🏠 Quanto vale esta casa?")
st.caption("App publicado por: duduzinho")


@st.cache_resource            # carrega o arquivo uma vez só, e não a cada clique
def carregar_modelo():
    return joblib.load("modelo_casas.pkl")    # o pipeline inteiro da Aula 11


modelo = carregar_modelo()

# as categorias do Ames, com o código que o modelo conhece e um rótulo para ler
BAIRROS = {"NAmes": "North Ames", "CollgCr": "College Creek", "OldTown": "Old Town",
           "Edwards": "Edwards", "Somerst": "Somerset", "Gilbert": "Gilbert",
           "NridgHt": "Northridge Heights", "Sawyer": "Sawyer", "NWAmes": "Northwest Ames",
           "SawyerW": "Sawyer West", "BrkSide": "Brookside", "Crawfor": "Crawford",
           "Mitchel": "Mitchell", "NoRidge": "Northridge", "Timber": "Timberland",
           "IDOTRR": "Iowa DOT and Rail Road", "ClearCr": "Clear Creek",
           "SWISU": "South & West of Iowa State University", "StoneBr": "Stone Brook",
           "Blmngtn": "Bloomington Heights", "MeadowV": "Meadow Village", "BrDale": "Briardale",
           "Veenker": "Veenker", "NPkVill": "Northpark Villa", "Blueste": "Bluestem"}
ESTILOS = {"1Story": "1 andar", "2Story": "2 andares",
           "1.5Fin": "1 andar e meio, o de cima acabado",
           "1.5Unf": "1 andar e meio, o de cima sem acabamento",
           "2.5Fin": "2 andares e meio, o de cima acabado",
           "2.5Unf": "2 andares e meio, o de cima sem acabamento",
           "SLvl": "níveis desencontrados", "SFoyer": "níveis desencontrados, entrada no meio"}
QUALIDADES = {"Ex": "excelente", "Gd": "boa", "TA": "média", "Fa": "regular"}

PES2_POR_M2 = 10.764          # o Ames mede área em pés quadrados: 1 m² = 10,764 pés²
TIPICA = dict(qualidade=6, area=136, porao=92, terreno=881, vagas=2,
              bairro="NAmes", estilo="1Story", externo="TA")   # os valores iniciais do formulário


def linha(qualidade, area, porao, terreno, vagas, bairro, estilo, externo):
    """Uma linha bruta, com os mesmos nomes de coluna do treino; o pipeline faz o resto."""
    return pd.DataFrame([{
        "OverallQual": qualidade, "GrLivArea": area * PES2_POR_M2,
        "TotalBsmtSF": porao * PES2_POR_M2, "LotArea": terreno * PES2_POR_M2,
        "GarageCars": vagas, "Neighborhood": bairro, "HouseStyle": estilo, "ExterQual": externo,
    }])


def dolares(valor):
    return f"US$ {valor:,.0f}".replace(",", ".")


# ── o formulário, na barra lateral
with st.sidebar:
    st.header("A casa")
    qualidade = st.slider("Qualidade geral do material e do acabamento", 1, 10, TIPICA["qualidade"])
    area = st.number_input("Área construída acima do solo (m²)", 31, 524, TIPICA["area"])
    porao = st.number_input("Área do porão (m²)", 0, 568, TIPICA["porao"])
    terreno = st.number_input("Área do terreno (m²)", 121, 19997, TIPICA["terreno"])
    vagas = st.slider("Vagas na garagem", 0, 4, TIPICA["vagas"])
    bairro = st.selectbox("Bairro", list(BAIRROS), format_func=lambda c: f"{c} ({BAIRROS[c]})")
    estilo = st.selectbox("Estilo da casa", list(ESTILOS), format_func=lambda c: f"{c} ({ESTILOS[c]})")
    externo = st.selectbox("Qualidade do acabamento externo", list(QUALIDADES), index=2,
                           format_func=lambda c: f"{c} ({QUALIDADES[c]})")

casa = dict(qualidade=qualidade, area=area, porao=porao, terreno=terreno, vagas=vagas,
            bairro=bairro, estilo=estilo, externo=externo)
preco = modelo.predict(linha(**casa))[0]
preco_tipica = modelo.predict(linha(**TIPICA))[0]

# ── o histórico fica na sessão do navegador: cada casa nova entra uma vez
if "historico" not in st.session_state:
    st.session_state.historico = []
registro = {"qualidade": qualidade, "área (m²)": area, "porão (m²)": porao, "terreno (m²)": terreno,
            "vagas": vagas, "bairro": bairro, "estilo": estilo, "acabamento externo": externo,
            "preço previsto (US$)": round(preco)}
if not st.session_state.historico or st.session_state.historico[-1] != registro:
    st.session_state.historico.append(registro)

aba_preco, aba_area, aba_linha, aba_historico = st.tabs(
    ["Preço", "E se a área mudar?", "O que o modelo recebe", "Histórico"])

with aba_preco:
    diferenca = round(preco - preco_tipica)    # o sinal na frente faz a seta subir ou descer
    if diferenca == 0:
        st.metric("Preço previsto", dolares(preco))
        st.write("É o preço da casa típica. Mude um campo na barra lateral e compare.")
    else:
        st.metric("Preço previsto", dolares(preco),
                  delta=f"{'-' if diferenca < 0 else '+'}{dolares(abs(diferenca))} em relação à casa típica")
    st.caption("A casa típica é a dos valores iniciais do formulário. "
               "É uma estimativa do modelo, não uma avaliação do imóvel.")
    with st.expander("Como o modelo chega a esse número"):
        st.write("É o pipeline da Aula 11, treinado com 1.460 vendas de Ames (Iowa, EUA), de 2006 "
                 "a 2010: ele preenche o que falta, coloca os números na mesma escala, transforma as "
                 "categorias em colunas e aplica uma regressão linear com regularização (Ridge).")

with aba_area:
    # o mesmo predict, chamado para várias áreas, com o resto da casa fixo
    areas = list(range(31, 525, 7))
    precos = [modelo.predict(linha(**{**casa, "area": a}))[0] for a in areas]
    st.write("O preço previsto para a mesma casa, mudando só a área construída:")
    st.line_chart(pd.DataFrame({"área construída (m²)": areas, "preço previsto (US$)": precos}),
                  x="área construída (m²)", y="preço previsto (US$)")
    st.caption(f"A sua casa tem {area} m² e sai por {dolares(preco)}. "
               "Cada ponto da linha é uma chamada ao mesmo modelo.")

with aba_linha:
    st.write("A linha que o app monta e entrega ao `predict`, com as áreas já em pés quadrados:")
    st.dataframe(linha(**casa), hide_index=True)

with aba_historico:
    historico = pd.DataFrame(st.session_state.historico)
    st.write(f"As {len(historico)} casas que você simulou nesta sessão:")
    st.dataframe(historico, hide_index=True)
    st.download_button("Baixar o histórico (CSV)", historico.to_csv(index=False).encode("utf-8"),
                       file_name="casas_simuladas.csv", mime="text/csv")
    if st.button("Limpar o histórico"):
        st.session_state.historico = []
        st.rerun()
