import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Preço de casas em Ames", page_icon="🏠")
st.title("🏠 Quanto vale esta casa?")
st.caption("App publicado por: duduzinho")
st.write("Preencha as características da casa. O modelo foi treinado com 1.460 vendas "
         "de Ames (Iowa, EUA), de 2006 a 2010, e o preço sai em dólares.")


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

col1, col2 = st.columns(2)
with col1:
    qualidade = st.slider("Qualidade geral do material e do acabamento", 1, 10, 6)
    area = st.number_input("Área construída acima do solo (m²)", 31, 524, 136)
    porao = st.number_input("Área do porão (m²)", 0, 568, 92)
    terreno = st.number_input("Área do terreno (m²)", 121, 19997, 881)
with col2:
    vagas = st.slider("Vagas na garagem", 0, 4, 2)
    bairro = st.selectbox("Bairro", list(BAIRROS), format_func=lambda c: f"{c} ({BAIRROS[c]})")
    estilo = st.selectbox("Estilo da casa", list(ESTILOS), format_func=lambda c: f"{c} ({ESTILOS[c]})")
    externo = st.selectbox("Qualidade do acabamento externo", list(QUALIDADES), index=2,
                           format_func=lambda c: f"{c} ({QUALIDADES[c]})")

PES2_POR_M2 = 10.764          # o Ames mede área em pés quadrados: 1 m² = 10,764 pés²

# uma linha bruta, com os mesmos nomes de coluna do treino; o pipeline faz o resto
casa = pd.DataFrame([{
    "OverallQual": qualidade, "GrLivArea": area * PES2_POR_M2,
    "TotalBsmtSF": porao * PES2_POR_M2, "LotArea": terreno * PES2_POR_M2,
    "GarageCars": vagas, "Neighborhood": bairro, "HouseStyle": estilo, "ExterQual": externo,
}])
preco = modelo.predict(casa)[0]

st.metric("Preço previsto", f"US$ {preco:,.0f}".replace(",", "."))
st.caption("É uma estimativa do modelo, não uma avaliação do imóvel.")
