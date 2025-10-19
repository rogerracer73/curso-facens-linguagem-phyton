"""Rotinas de limpeza e geração de relatórios para o dataset Disney+.

Este módulo foi escrito para atender ao exercício proposto, que exige:
- leitura manual do arquivo ``disney_plus_shows.txt`` sem o uso de csv/pandas;
- padronização e validação dos campos do dataset;
- criação de classes ``Filme`` e ``Serie`` que representem cada linha válida;
- geração de um arquivo revisado ``disney_plus_shows_rev1.txt``;
- produção de relatórios de limpeza e estatísticos em arquivos de texto.

A implementação é propositalmente verbosa e rica em comentários para explicar
cada etapa do processamento, deixando o fluxo fácil de acompanhar em um
contexto didático.
"""

from __future__ import annotations

import os
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Constantes auxiliares
# ---------------------------------------------------------------------------

# Caminho do arquivo principal e dos arquivos gerados durante o processo.
BASE_DATASET_PATH = Path("disney_plus_shows.txt")
REVISION_DATASET_PATH = Path("disney_plus_shows_rev1.txt")
CLEANING_REPORT_PATH = Path("relatorio_limpeza_disney.txt")
ANALYTICS_REPORT_PATH = Path("relatorio-disney.txt")

# Colunas esperadas no dataset. O exercício cita estes rótulos e os três
# últimos campos são importantes para validações numéricas adicionais.
EXPECTED_COLUMNS: Tuple[str, ...] = (
    "imdb_id",
    "title",
    "plot",
    "type",
    "rated",
    "year",
    "released_at",
    "added_at",
    "runtime",
    "genre",
    "director",
    "writer",
    "actors",
    "language",
    "imdb_score",
    "metascore",
    "imdb_votes",
)

# Campos que devem ser interpretados como numéricos. ``year`` e ``runtime``
# são tratados como inteiros, enquanto as notas são convertidas para float.
INTEGER_FIELDS = {"year", "runtime", "imdb_votes"}
FLOAT_FIELDS = {"imdb_score", "metascore"}

# Caracteres possíveis de separação de colunas. Será escolhido o que gerar o
# maior número de colunas consistentes logo na primeira linha.
CANDIDATE_DELIMITERS = ("|", ";", ",", "\t")

# Valores considerados nulos no dataset e que serão transformados em ``None``.
NULL_STRINGS = {"", "N/A", "NA", "NULL", "NONE"}


# ---------------------------------------------------------------------------
# Funções de apoio para limpeza dos dados
# ---------------------------------------------------------------------------

def detectar_delimitador(header_line: str) -> str:
    """Descobre o delimitador mais provável a partir do cabeçalho.

    Percorre a lista de delimitadores candidatos e seleciona aquele que gera
    a quantidade de colunas igual ao número de campos esperados. Caso nenhum
    deles produza a quantidade exata, o delimitador que gerar mais campos é
    utilizado como fallback.
    """

    melhor_delimitador = CANDIDATE_DELIMITERS[0]
    maior_quantidade = 1

    for delimitador in CANDIDATE_DELIMITERS:
        quantidade = len(parse_line(header_line, delimitador))
        if quantidade == len(EXPECTED_COLUMNS):
            return delimitador
        if quantidade > maior_quantidade:
            melhor_delimitador = delimitador
            maior_quantidade = quantidade

    return melhor_delimitador


def parse_line(line: str, delimiter: str) -> List[str]:
    """Divide uma linha de texto em colunas respeitando aspas.

    O parser abaixo é simples, porém suficiente para lidar com campos que
    possam ter o delimitador dentro de aspas duplas. Ele percorre caractere a
    caractere, identificando trechos delimitados por aspas e acumulando o
    conteúdo em uma lista de colunas.
    """

    values: List[str] = []
    current = []
    inside_quotes = False

    for char in line.strip("\n\r"):
        if char == '"':
            inside_quotes = not inside_quotes
            continue
        if char == delimiter and not inside_quotes:
            values.append("".join(current))
            current = []
        else:
            current.append(char)

    values.append("".join(current))
    return values


def normalizar_string(valor: Optional[str]) -> Optional[str]:
    """Padroniza campos de texto conforme as regras do exercício.

    - Valores nulos permanecem ``None``;
    - Espaços em branco são removidos por completo;
    - Pontuações simples (vírgulas e pontos) são eliminadas;
    - Todo o texto é convertido para maiúsculas.
    """

    if valor is None:
        return None

    texto = valor.strip()
    if not texto:
        return None

    sem_espacos = texto.replace(" ", "")
    sem_pontuacao = sem_espacos.replace(",", "").replace(".", "")
    return sem_pontuacao.upper()


def normalizar_lista(valor: Optional[str]) -> Tuple[str, ...]:
    """Normaliza campos com múltiplos itens (ex.: atores, idiomas).

    A estratégia é dividir o conteúdo por diferentes separadores (vírgula,
    ponto e vírgula, barra ou pipe), normalizar cada item individualmente
    através da função :func:`normalizar_string` e eliminar duplicatas vazias.
    O resultado final é retornado como tupla ordenada, que é imutável e pode
    ser utilizada com segurança nas classes ``Filme`` e ``Serie``.
    """

    if valor is None:
        return ()

    bruto = valor.replace("\n", " ")
    itens: List[str] = []
    item_atual: List[str] = []

    # Dividimos manualmente porque não podemos depender de bibliotecas
    # externas. Os separadores considerados foram escolhidos com base nos
    # formatos mais comuns observados em datasets similares.
    for caractere in bruto:
        if caractere in {",", ";", "|", "/"}:
            itens.append("".join(item_atual))
            item_atual = []
        else:
            item_atual.append(caractere)

    itens.append("".join(item_atual))

    normalizados = []
    for item in itens:
        texto = normalizar_string(item)
        if texto:
            normalizados.append(texto)

    # A ordenação torna o resultado determinístico, o que facilita testes.
    return tuple(sorted(dict.fromkeys(normalizados)))


def converter_inteiro(valor: Optional[str]) -> Optional[int]:
    """Converte strings em inteiros, tratando símbolos indesejados."""

    if valor is None:
        return None

    texto = valor.replace(",", "").replace(" ", "").strip()
    if not texto or texto.upper() in NULL_STRINGS:
        return None

    try:
        return int(float(texto))
    except ValueError:
        return None


def converter_decimal(valor: Optional[str]) -> Optional[float]:
    """Converte strings em ``float`` aceitando vírgulas como decimal."""

    if valor is None:
        return None

    texto = valor.replace(" ", "").strip().replace(",", ".")
    if not texto or texto.upper() in NULL_STRINGS:
        return None

    try:
        return float(texto)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Classes de domínio: Filme e Serie
# ---------------------------------------------------------------------------

@dataclass
class MidiaBase:
    """Classe base contendo os campos compartilhados entre filmes e séries."""

    imdb_id: Optional[str] = None
    title: Optional[str] = None
    plot: Optional[str] = None
    tipo: Optional[str] = None
    rated: Optional[str] = None
    year: Optional[int] = None
    released_at: Optional[str] = None
    added_at: Optional[str] = None
    runtime: Optional[int] = None
    genre: Optional[str] = None
    director: Optional[str] = None
    writer: Optional[str] = None
    actors: Tuple[str, ...] = ()
    language: Tuple[str, ...] = ()
    imdb_score: Optional[float] = None
    metascore: Optional[float] = None
    imdb_votes: Optional[int] = None

    # As propriedades abaixo definem getters e setters explícitos. O uso do
    # ``@dataclass`` simplifica a criação do ``__init__`` e do ``repr``, mas
    # mantemos os acessos controlados para garantir que a limpeza permaneça
    # consistente caso valores posteriores sejam atribuídos.

    @property
    def titulo(self) -> Optional[str]:
        return self.title

    @titulo.setter
    def titulo(self, valor: Optional[str]) -> None:
        self.title = normalizar_string(valor)

    @property
    def diretor(self) -> Optional[str]:
        return self.director

    @diretor.setter
    def diretor(self, valor: Optional[str]) -> None:
        self.director = normalizar_string(valor)

    @property
    def linguas(self) -> Tuple[str, ...]:
        return self.language

    @linguas.setter
    def linguas(self, valor: Optional[str]) -> None:
        self.language = normalizar_lista(valor)

    @property
    def elenco(self) -> Tuple[str, ...]:
        return self.actors

    @elenco.setter
    def elenco(self, valor: Optional[str]) -> None:
        self.actors = normalizar_lista(valor)

    def to_row(self, colunas: Sequence[str]) -> List[str]:
        """Transforma o objeto novamente em linha de texto.

        O método é usado na geração do arquivo revisado. Os campos são
        obtidos dinamicamente através do ``getattr`` para permitir que a
        função funcione mesmo que novas colunas sejam adicionadas futuramente.
        """

        linha: List[str] = []
        for coluna in colunas:
            atributo = coluna if coluna != "type" else "tipo"
            valor = getattr(self, atributo, None)

            if isinstance(valor, tuple):
                linha.append(";".join(valor))
            elif valor is None:
                linha.append("")
            else:
                linha.append(str(valor))
        return linha


class Filme(MidiaBase):
    """Representa uma linha do dataset correspondente a um filme."""


class Serie(MidiaBase):
    """Representa uma linha do dataset correspondente a uma série."""


# ---------------------------------------------------------------------------
# Função principal de limpeza e carregamento
# ---------------------------------------------------------------------------

def abrir_filmes_disney(arquivo: Path = BASE_DATASET_PATH) -> List[MidiaBase]:
    """Abre, higieniza e carrega o dataset Disney+.

    A função executa todas as etapas solicitadas:
    1. leitura linha a linha do arquivo base;
    2. remoção de registros inválidos (tipos desconhecidos, campos faltantes,
       número incorreto de colunas etc.);
    3. normalização de strings e conversão de campos numéricos;
    4. gravação do arquivo revisado e do relatório de limpeza;
    5. criação de instâncias das classes ``Filme`` ou ``Serie``.

    O retorno é uma lista contendo todos os objetos válidos.
    """

    registros: List[MidiaBase] = []
    linhas_revisadas: List[List[str]] = []

    # Dicionário que acumula o número de linhas removidas por motivo.
    linhas_removidas: Dict[str, int] = defaultdict(int)
    linhas_invalidas = 0

    if not arquivo.exists():
        raise FileNotFoundError(
            f"Arquivo {arquivo} não encontrado. Certifique-se de que o dataset "
            "está no mesmo diretório do script."
        )

    with arquivo.open("r", encoding="utf-8") as handle:
        linhas_brutas = [linha.rstrip("\n") for linha in handle]

    linhas_nao_vazias = [linha for linha in linhas_brutas if linha.strip()]
    if not linhas_nao_vazias:
        raise ValueError("O arquivo não contém linhas válidas para processamento.")

    cabecalho = linhas_nao_vazias[0]
    delimitador = detectar_delimitador(cabecalho)
    colunas = parse_line(cabecalho, delimitador)

    if len(colunas) != len(EXPECTED_COLUMNS):
        # Mantemos o cabeçalho detectado, porém avisamos na contagem.
        linhas_removidas["cabecalho_inesperado"] += 1

    # Índice das linhas de dados começa logo após o cabeçalho identificado.
    linhas_de_dados = linhas_nao_vazias[1:]

    for numero_linha, linha in enumerate(linhas_de_dados, start=2):
        valores = parse_line(linha, delimitador)

        if len(valores) != len(colunas):
            linhas_removidas["colunas_inconsistentes"] += 1
            continue

        registro_bruto = dict(zip(colunas, valores))

        tipo = normalizar_string(registro_bruto.get("type"))
        titulo = normalizar_string(registro_bruto.get("title"))

        if not titulo:
            linhas_removidas["titulo_ausente"] += 1
            continue

        if tipo not in {"MOVIE", "SERIES"}:
            linhas_removidas["tipo_invalido"] += 1
            continue

        dados_limpos: Dict[str, Optional[str]] = {}
        for coluna, valor in registro_bruto.items():
            dados_limpos[coluna] = None if valor.upper() in NULL_STRINGS else valor

        # Conversões numéricas e normalizações de texto.
        dados_num: Dict[str, Optional[float]] = {}
        for coluna in colunas:
            valor_atual = dados_limpos.get(coluna)

            if coluna in INTEGER_FIELDS:
                convertido = converter_inteiro(valor_atual)
                if valor_atual and convertido is None:
                    linhas_invalidas += 1
                dados_num[coluna] = convertido
            elif coluna in FLOAT_FIELDS:
                convertido = converter_decimal(valor_atual)
                if valor_atual and convertido is None:
                    linhas_invalidas += 1
                dados_num[coluna] = convertido
            elif coluna == "language":
                dados_num[coluna] = normalizar_lista(valor_atual)
            elif coluna == "actors":
                dados_num[coluna] = normalizar_lista(valor_atual)
            else:
                dados_num[coluna] = normalizar_string(valor_atual)

        # Preenchemos a estrutura padrão com os campos esperados. Campos ausentes
        # são preenchidos com ``None`` para manter consistência.
        kwargs: Dict[str, Optional[str]] = {
            "imdb_id": normalizar_string(dados_limpos.get("imdb_id")),
            "title": dados_num.get("title"),
            "plot": dados_num.get("plot"),
            "tipo": tipo,
            "rated": dados_num.get("rated"),
            "year": dados_num.get("year"),
            "released_at": dados_num.get("released_at"),
            "added_at": dados_num.get("added_at"),
            "runtime": dados_num.get("runtime"),
            "genre": dados_num.get("genre"),
            "director": dados_num.get("director"),
            "writer": dados_num.get("writer"),
            "actors": dados_num.get("actors", ()),
            "language": dados_num.get("language", ()),
            "imdb_score": dados_num.get("imdb_score"),
            "metascore": dados_num.get("metascore"),
            "imdb_votes": dados_num.get("imdb_votes"),
        }

        if tipo == "MOVIE":
            objeto: MidiaBase = Filme(**kwargs)
        else:
            objeto = Serie(**kwargs)

        registros.append(objeto)
        linhas_revisadas.append(objeto.to_row(colunas))

    # Persistimos o arquivo revisado usando o mesmo delimitador do original.
    with REVISION_DATASET_PATH.open("w", encoding="utf-8") as handle:
        handle.write(delimitador.join(colunas) + "\n")
        for linha in linhas_revisadas:
            handle.write(delimitador.join(linha) + "\n")

    # Produzimos o relatório de limpeza solicitado.
    linhas_restantes = Counter(type(obj).__name__ for obj in registros)
    relatorio = [
        "RELATÓRIO DE LIMPEZA DO DATASET DISNEY+",
        "======================================",
        f"Total de linhas processadas: {len(linhas_de_dados)}",
        f"Linhas válidas: {len(registros)}",
        "",
        "Linhas removidas por motivo:",
    ]

    for motivo, quantidade in sorted(linhas_removidas.items()):
        relatorio.append(f"- {motivo}: {quantidade}")

    relatorio.extend(
        [
            "",
            "Resumo por categoria:",
            f"- Filmes válidos: {linhas_restantes.get('Filme', 0)}",
            f"- Séries válidas: {linhas_restantes.get('Serie', 0)}",
            "",
            f"Campos numéricos inválidos detectados: {linhas_invalidas}",
        ]
    )

    with CLEANING_REPORT_PATH.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(relatorio) + "\n")

    print("\n".join(relatorio))
    return registros


# ---------------------------------------------------------------------------
# Rotina de geração do relatório analítico
# ---------------------------------------------------------------------------

def gerar_relatorio(registros: Sequence[MidiaBase]) -> None:
    """Produz o relatório consolidado solicitado no enunciado."""

    filmes = [item for item in registros if isinstance(item, Filme)]
    series = [item for item in registros if isinstance(item, Serie)]

    def media_valida(valores: Iterable[Optional[float]]) -> Optional[float]:
        numeros = [valor for valor in valores if valor is not None]
        if not numeros:
            return None
        return statistics.mean(numeros)

    media_imdb = media_valida(filme.imdb_score for filme in filmes)
    media_metascore = media_valida(filme.metascore for filme in filmes)
    media_votos = media_valida(filme.imdb_votes for filme in filmes)

    contador_linguas: Counter[str] = Counter()
    contador_atores: Counter[str] = Counter()
    contador_diretores: Counter[str] = Counter()
    filmes_por_titulo: Dict[str, List[Filme]] = defaultdict(list)
    filmes_por_ano: Counter[int] = Counter()

    for filme in filmes:
        for lingua in filme.language:
            contador_linguas[lingua] += 1
        for ator in filme.actors:
            contador_atores[ator] += 1
        if filme.director:
            contador_diretores[filme.director] += 1
        if filme.year is not None:
            filmes_por_ano[filme.year] += 1
        if filme.title:
            filmes_por_titulo[filme.title].append(filme)

    # Para incluir atores e idiomas presentes apenas em séries, repetimos o
    # processo com a coleção de séries válidas.
    for serie in series:
        for lingua in serie.language:
            contador_linguas[lingua] += 1
        for ator in serie.actors:
            contador_atores[ator] += 1

    top_linguas = contador_linguas.most_common(3)
    bottom_linguas = sorted(contador_linguas.items(), key=lambda item: (item[1], item[0]))[:3]
    top_atores = contador_atores.most_common(3)
    bottom_atores = sorted(contador_atores.items(), key=lambda item: (item[1], item[0]))[:3]

    diretor_mais_filmes = None
    if contador_diretores:
        diretor_mais_filmes = contador_diretores.most_common(1)[0]

    diretor_melhor_imdb = None
    diretor_melhor_metascore = None

    filmes_com_imdb = [filme for filme in filmes if filme.imdb_score is not None]
    if filmes_com_imdb:
        melhor_filme_imdb = max(filmes_com_imdb, key=lambda f: f.imdb_score)
        diretor_melhor_imdb = (melhor_filme_imdb.director, melhor_filme_imdb.title, melhor_filme_imdb.imdb_score)

    filmes_com_metascore = [filme for filme in filmes if filme.metascore is not None]
    if filmes_com_metascore:
        melhor_filme_meta = max(filmes_com_metascore, key=lambda f: f.metascore)
        diretor_melhor_metascore = (
            melhor_filme_meta.director,
            melhor_filme_meta.title,
            melhor_filme_meta.metascore,
        )

    ano_campeao = None
    if filmes_por_ano:
        ano_campeao = filmes_por_ano.most_common(1)[0]

    pior_serie_imdb = None
    series_com_imdb = [serie for serie in series if serie.imdb_score is not None]
    if series_com_imdb:
        pior = min(series_com_imdb, key=lambda s: s.imdb_score)
        pior_serie_imdb = (pior.title, pior.imdb_score)

    pior_serie_meta = None
    series_com_meta = [serie for serie in series if serie.metascore is not None]
    if series_com_meta:
        pior = min(series_com_meta, key=lambda s: s.metascore)
        pior_serie_meta = (pior.title, pior.metascore)

    remakes: List[Tuple[str, List[int]]] = []
    for titulo, filmes_lista in filmes_por_titulo.items():
        anos = {filme.year for filme in filmes_lista if filme.year is not None}
        if len(anos) > 1:
            remakes.append((titulo, sorted(anos)))

    linhas_relatorio = [
        "RELATÓRIO ANALÍTICO DISNEY+",
        "============================",
        f"Quantidade de filmes avaliados: {len(filmes)}",
        f"Quantidade de séries avaliadas: {len(series)}",
        "",
        f"Média IMDB dos filmes: {media_imdb:.2f}" if media_imdb is not None else "Média IMDB dos filmes: N/D",
        f"Média Metascore dos filmes: {media_metascore:.2f}" if media_metascore is not None else "Média Metascore dos filmes: N/D",
        f"Média de votos IMDB dos filmes: {media_votos:.0f}" if media_votos is not None else "Média de votos IMDB dos filmes: N/D",
        "",
        "Idiomas mais utilizados:",
    ]

    for lingua, contagem in top_linguas:
        linhas_relatorio.append(f"- {lingua}: {contagem}")

    linhas_relatorio.append("")
    linhas_relatorio.append("Idiomas menos utilizados:")
    for lingua, contagem in bottom_linguas:
        linhas_relatorio.append(f"- {lingua}: {contagem}")

    linhas_relatorio.append("")
    linhas_relatorio.append("Atores mais frequentes:")
    for ator, contagem in top_atores:
        linhas_relatorio.append(f"- {ator}: {contagem}")

    linhas_relatorio.append("")
    linhas_relatorio.append("Atores menos frequentes:")
    for ator, contagem in bottom_atores:
        linhas_relatorio.append(f"- {ator}: {contagem}")

    linhas_relatorio.append("")
    if diretor_mais_filmes:
        diretor, quantidade = diretor_mais_filmes
        linhas_relatorio.append(f"Diretor com mais filmes: {diretor} ({quantidade})")
    else:
        linhas_relatorio.append("Diretor com mais filmes: N/D")

    if diretor_melhor_imdb:
        diretor, titulo, nota = diretor_melhor_imdb
        linhas_relatorio.append(
            f"Filme mais popular no IMDB: {titulo} (Direção de {diretor}) com nota {nota:.2f}"
        )
    else:
        linhas_relatorio.append("Filme mais popular no IMDB: N/D")

    if diretor_melhor_metascore:
        diretor, titulo, nota = diretor_melhor_metascore
        linhas_relatorio.append(
            f"Filme mais popular no Metascore: {titulo} (Direção de {diretor}) com nota {nota:.2f}"
        )
    else:
        linhas_relatorio.append("Filme mais popular no Metascore: N/D")

    if ano_campeao:
        ano, quantidade = ano_campeao
        linhas_relatorio.append(f"Ano com mais lançamentos de filmes: {ano} ({quantidade})")
    else:
        linhas_relatorio.append("Ano com mais lançamentos de filmes: N/D")

    if pior_serie_imdb:
        titulo, nota = pior_serie_imdb
        linhas_relatorio.append(f"Pior série segundo IMDB: {titulo} (nota {nota:.2f})")
    else:
        linhas_relatorio.append("Pior série segundo IMDB: N/D")

    if pior_serie_meta:
        titulo, nota = pior_serie_meta
        linhas_relatorio.append(f"Pior série segundo Metascore: {titulo} (nota {nota:.2f})")
    else:
        linhas_relatorio.append("Pior série segundo Metascore: N/D")

    linhas_relatorio.append("")
    linhas_relatorio.append("Filmes com múltiplos lançamentos (possíveis remakes):")
    if remakes:
        for titulo, anos in remakes:
            anos_formatados = ", ".join(str(ano) for ano in anos)
            linhas_relatorio.append(f"- {titulo}: anos {anos_formatados}")
    else:
        linhas_relatorio.append("- Nenhum remake identificado")

    conteudo_relatorio = "\n".join(linhas_relatorio) + "\n"

    with ANALYTICS_REPORT_PATH.open("w", encoding="utf-8") as handle:
        handle.write(conteudo_relatorio)

    print(conteudo_relatorio)


if __name__ == "__main__":
    dados = abrir_filmes_disney()
    gerar_relatorio(dados)
