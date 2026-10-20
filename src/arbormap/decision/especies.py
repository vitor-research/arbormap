"""
Espécies para a arborização, com origem e bioma conferidos na lista oficial da flora brasileira

1. Lista de espécies e porte: Manual de Arborização Urbana de Fortaleza (Prefeitura de Fortaleza /
   SEUMA, 2020), tabelas 12 a 16 (p. 120 a 123): pequeno, médio e grande porte, palmeiras nativas
   e exóticas não invasoras. É a lista oficial mais próxima do clima de Teresina encontrada até agora.
2. Bioma (domínio fitogeográfico), origem e ocorrência no Piauí: Flora e Funga do Brasil (JBRJ),
   lido pela API do GBIF (conjunto aacd816d-662c-49d2-ad1a-97e66e2a2908).
3. Regras de uso (p. 67 a 73 do mesmo manual): largura da calçada por porte, espaçamento mínimo
   (5, 8 e 12 m), só pequeno porte sob fiação, distâncias de esquina e postes.

Machado et al. (2006), "Árvores nativas para a arborização de Teresina, Piauí" (REVSBAU 1(1): 10-18),
não pôde ser conferido no texto completo (site da revista fora do ar) e ainda não entra na lista.
"""

import json
import re
import time
import urllib.parse
import urllib.request

GBIF_FLORA_BR = "aacd816d-662c-49d2-ad1a-97e66e2a2908"
TABELAS = {"12": "pequeno", "13": "medio", "14": "grande", "15": "palmeira", "16": "exotica"}
PAGINAS = {"pequeno": "120-121", "medio": "121-122", "grande": "122", "palmeira": "123", "exotica": "123"}

# Regras do manual (p. 67-73), usadas na página e no plano
REGRAS = {
    "espacamento_m": {"pequeno": 5.0, "medio": 8.0, "grande": 12.0},
    "calcada_largura_m": {"pequeno": "1,50 a 2,00 m (só pequeno porte, arvoretas ou arbustos)",
                          "medio": "2,00 a 2,40 m (pequeno ou médio, até 8 m de altura)",
                          "grande": "2,40 m ou mais (grande porte até 12 m); acima de 12 m, só com 3,00 m ou mais"},
    "sob_fiacao": "em calçadas sob rede elétrica com 1,50 a 3,00 m, só pequeno porte",
    "calcada_minima": "abaixo de 1,50 m não se recomenda plantar árvores",
    "distancias_m": {"pequeno": {"esquina": 5, "poste": 3, "transformador": 5},
                     "medio": {"esquina": 5, "poste": 4, "transformador": 8},
                     "grande": {"esquina": 5, "poste": 5, "transformador": 12}},
    "muda_padrao": "copa formada com 3 a 5 galhos, fuste de no mínimo 1,80 m e DAP a partir de 3 cm (p. 14)",
    "rega": "10 a 20 L por muda, pelo menos 3 vezes por semana nos 3 primeiros meses e depois quando necessário (p. 93)",
    "tutor": "tutor com altura mínima de 2,30 m, amarrio sem arame (p. 87)",
    "protetor": "protetor de muda com altura máxima de 1,20 m (p. 88)",
    "fonte": "Prefeitura de Fortaleza (2020), Manual de Arborização Urbana de Fortaleza, SEUMA",
}


def ler_tabelas(txt):
    """Extrai (porte, família, nome científico, nome popular) das tabelas 12 a 16 do texto do manual."""
    linhas = [l.strip() for l in txt.splitlines()]
    out, porte, i = [], None, 0
    while i < len(linhas):
        m = re.match(r"Tabela (1[2-6]) -", linhas[i])
        if m:
            porte = TABELAS[m.group(1)]
        if porte and linhas[i].startswith("Fonte:"):
            porte = None
        if porte and re.fullmatch(r"\d{1,2}", linhas[i]) and i + 3 < len(linhas):
            fam, cien, pop = linhas[i + 1], linhas[i + 2], linhas[i + 3]
            if re.fullmatch(r"[A-Z][a-z]+aceae", fam) and re.fullmatch(r"[A-Z][a-z]+ [a-z-]+", cien):
                out.append({"porte": porte, "familia": fam, "nome_cientifico": cien, "nome_popular": pop.strip()})
                i += 4
                continue
        i += 1
    return out


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "arbormap/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def flora_do_brasil(nome):
    q = urllib.parse.quote(nome)
    res = _get(f"https://api.gbif.org/v1/species/search?datasetKey={GBIF_FLORA_BR}&q={q}&rank=SPECIES&limit=20")["results"]
    cand = [r for r in res if r.get("canonicalName") == nome]
    if not cand:
        return {"encontrada": False}
    r = sorted(cand, key=lambda x: x.get("taxonomicStatus") != "ACCEPTED")[0]
    key = r.get("acceptedKey") or r["key"]
    dist = _get(f"https://api.gbif.org/v1/species/{key}/distributions?limit=100")["results"]
    dominios, ufs, nativa, endemismo = set(), set(), None, None
    for d in dist:
        ufs.add(d.get("locationId", "").replace("BR-", ""))
        nativa = d.get("establishmentMeans") or nativa
        try:
            rem = json.loads(d.get("remarks") or "{}")
            dominios.update(rem.get("phytogeographicDomain", []))
            endemismo = rem.get("endemism") or endemismo
        except json.JSONDecodeError:
            pass
    acc = _get(f"https://api.gbif.org/v1/species/{key}")
    return {"encontrada": True, "nome_aceito": acc.get("canonicalName"), "status": r.get("taxonomicStatus"),
            "dominios": sorted(dominios), "ufs": sorted(u for u in ufs if u), "origem": (nativa or "").lower(),
            "endemica_do_brasil": endemismo == "Endemica", "gbif_key": key}


def montar(txt_manual, log=print):
    lista = ler_tabelas(txt_manual)
    for e in lista:
        try:
            e.update(flora_do_brasil(e["nome_cientifico"]))
        except Exception as ex:  # rede
            e.update({"encontrada": False, "erro": str(ex)})
        e["ocorre_no_piaui"] = "PI" in e.get("ufs", [])
        e["caatinga_ou_cerrado"] = bool({"Caatinga", "Cerrado"} & set(e.get("dominios", [])))
        e["pagina_manual"] = PAGINAS[e["porte"]]
        log(f"{e['porte']:8s} {e['nome_cientifico']:30s} {','.join(e.get('dominios', [])):40s} PI={e['ocorre_no_piaui']}")
        time.sleep(0.1)
    return {"especies": lista, "regras": REGRAS,
            "fontes": {"lista_e_porte": "Prefeitura de Fortaleza (2020). Manual de Arborização Urbana de Fortaleza. SEUMA, tabelas 12 a 16, p. 120-123.",
                       "bioma_e_origem": "Flora e Funga do Brasil (Jardim Botânico do Rio de Janeiro), via API do GBIF, conjunto " + GBIF_FLORA_BR,
                       "pendente": "Machado et al. (2006), Árvores nativas para a arborização de Teresina, Piauí, REVSBAU 1(1): 10-18 (texto completo não acessado)"}}
