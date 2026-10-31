"""Knowledge graphs and graph databases / 知识图谱与图数据库.

Run / 运行:
    python docs/20-knowledge-graphs/code/knowledge_graph.py

Covers / 包含:
    1. a small world written as documents: people, companies, cities, countries, and how they relate
       写成文档的小世界：人、公司、城市、国家，以及它们之间的关系
    2. building the graph: extract (subject, relation, object) triples from the text, then resolve
       different names of the same entity into one node
       建图：从文字中抽取（主语，关系，宾语）三元组，再把同一实体的不同叫法合并成一个节点
    3. a tiny graph database with a Cypher-like query language
       一个支持类 Cypher 查询语言的小型图数据库
    4. multi-hop questions: one round of keyword retrieval, retrieval hop by hop, and a graph query
       多跳问题：一次关键词检索、逐跳检索、图查询
    5. global questions and GraphRAG: counting over the whole corpus, communities and their summaries
       全局问题与 GraphRAG：在整个语料上计数、社区及其摘要
"""

import math
import random
import re
from collections import Counter, defaultdict

import torch

# ---------------------------------------------------------------------------
# 1. A small world, written as documents / 写成文档的小世界
# ---------------------------------------------------------------------------
SYL_A = "ma ve li to sa ri ka no be da lu fe mi ro ta ne si ko ga pe".split()
SYL_B = "ra len vik dor min sel tar nis bek lov run sha mar kel dun fis gal ton rik per".split()
COUNTRIES = ["Norland", "Estria", "Valmora", "Kesh", "Oribel"]
KINDS = ["Bank", "Air", "Labs", "Works"]


def made_up(rng, used):
    while True:
        w = (rng.choice(SYL_A) + rng.choice(SYL_B)).capitalize()
        if w not in used:
            used.add(w)
            return w


def make_world(n_people=600, n_companies=40, n_cities=25, seed=0):
    """The ground truth: entities and (subject, relation, object) facts. 真实的实体和关系。"""
    rng = random.Random(seed)
    used = set()
    cities = {made_up(rng, used) + rng.choice(["vik", "berg", "ton", "a"]): rng.choice(COUNTRIES) for _ in range(n_cities)}
    city_list = list(cities)
    hubs = city_list[:6]  # a few big cities host most companies / 少数大城市集中了大多数公司
    companies = {}
    for _ in range(n_companies):
        name = f"{made_up(rng, used)} {rng.choice(KINDS)}"
        companies[name] = dict(city=rng.choice(hubs) if rng.random() < 0.7 else rng.choice(city_list), year=rng.randint(1950, 2015))
    comp_list = list(companies)
    weights = [1 / (i + 1) ** 0.8 for i in range(len(comp_list))]  # some employers are much bigger / 有些公司大得多
    people = {}
    while len(people) < n_people:  # first names may repeat; full names may not / 名可以重复，全名不行
        first = (rng.choice(SYL_A) + rng.choice(SYL_B)).capitalize()
        last = (rng.choice(SYL_A) + rng.choice(SYL_A) + rng.choice(SYL_B)).capitalize()
        name = f"{first} {last}"
        if name in people or first in used or last in used:  # never clash with a company or city word
            continue
        people[name] = dict(born=rng.choice(city_list), works=rng.choices(comp_list, weights)[0], she=rng.random() < 0.5)
    names = list(people)
    by_company = defaultdict(list)
    for p, a in people.items():
        by_company[a["works"]].append(p)
    for p, a in people.items():  # friends are mostly colleagues / 朋友大多是同事
        pool = [q for q in by_company[a["works"]] if q != p]
        a["knows"] = sorted({rng.choice(pool) if pool and rng.random() < 0.8 else rng.choice(names) for _ in range(2)} - {p})
    for c in comp_list:
        companies[c]["founder"] = rng.choice(names)
    facts = set()
    for p, a in people.items():
        facts |= {(p, "BORN_IN", a["born"]), (p, "WORKS_AT", a["works"])} | {(p, "KNOWS", q) for q in a["knows"]}
    for c, a in companies.items():
        facts |= {(c, "LOCATED_IN", a["city"]), (c, "FOUNDED_BY", a["founder"])}
    for c, k in cities.items():
        facts.add((c, "IN_COUNTRY", k))
    return dict(people=people, companies=companies, cities=cities, facts=facts)


def alias(company):
    """How people's profiles often shorten a company's name: "Gasel Bank" -> "Gasel". 人物档案里常用的简称。"""
    return company.split()[0]


def documents(world, alias_rate=0.4, seed=1):
    """One short document per entity. Company documents use the official name ("Gasel Bank Ltd."); person
    documents often use a short form ("Gasel"). 每个实体一篇短文档。公司文档用正式名称，人物档案常用简称。"""
    rng = random.Random(seed)
    docs = []
    for p, a in world["people"].items():
        pron = "She" if a["she"] else "He"
        employer = alias(a["works"]) if rng.random() < alias_rate else a["works"]
        friends = " and ".join(a["knows"])
        docs.append((p, f"{p} was born in {a['born']}. {pron} works at {employer}." + (f" {pron} is a friend of {friends}." if friends else "")))
    for c, a in world["companies"].items():
        docs.append((c, f"{c} Ltd. is headquartered in {a['city']}. It was founded in {a['year']} by {a['founder']}."))
    for c, k in world["cities"].items():
        docs.append((c, f"{c} is a city in {k}."))
    return docs


# ---------------------------------------------------------------------------
# 2. Building the graph: extraction and entity resolution / 建图：抽取与实体消解
# ---------------------------------------------------------------------------
PATTERNS = [  # (regex on one sentence, relation, which group is the object) / 句式、关系
    (r"^(?P<s>.+?) was born in (?P<o>[A-Z]\w+)\.$", "BORN_IN"),
    (r"^(?P<s>She|He) works at (?P<o>.+?)\.$", "WORKS_AT"),
    (r"^(?P<s>She|He) is a friend of (?P<o>.+?)\.$", "KNOWS"),
    (r"^(?P<s>.+?) is headquartered in (?P<o>[A-Z]\w+)\.$", "LOCATED_IN"),
    (r"^(?P<s>It) was founded in \d+ by (?P<o>.+?)\.$", "FOUNDED_BY"),
    (r"^(?P<s>.+?) is a city in (?P<o>[A-Z]\w+)\.$", "IN_COUNTRY"),
]


def extract(docs):
    """Turn sentences into triples. A pronoun (She / He / It) refers to the document's first subject:
    the simplest form of coreference resolution. In practice an LLM does this step, with some errors.
    把句子变成三元组。代词指向文档的第一个主语：最简单的指代消解。实际中这一步通常交给大模型，也会出错。"""
    triples = []
    for _, text in docs:
        subject = None
        for sent in re.split(r"(?<=\.)\s+(?=[A-Z])", text):  # not after "Ltd. is" / 不在 "Ltd. is" 处断开
            for pat, rel in PATTERNS:
                m = re.match(pat, sent)
                if not m:
                    continue
                s = m.group("s")
                s = subject if s in ("She", "He", "It") else s
                subject = subject or s
                for o in m.group("o").split(" and ") if rel == "KNOWS" else [m.group("o")]:
                    triples.append((s, rel, o))
                break
    return triples


def resolve(triples):
    """Entity resolution: strip legal suffixes ("Gasel Bank Ltd." -> "Gasel Bank"), then map a short name
    to the one full name it uniquely starts ("Gasel" -> "Gasel Bank").
    实体消解：去掉公司后缀，再把简称映射到唯一以它开头的全称。"""
    strip = lambda n: re.sub(r"\s+(Ltd|Inc|Group)\.?$", "", n)  # noqa: E731
    triples = [(strip(s), r, strip(o)) for s, r, o in triples]
    full = {s for s, r, _ in triples if r == "LOCATED_IN"}
    by_first = defaultdict(set)
    for f in full:
        by_first[f.split()[0]].add(f)
    canon = lambda n: next(iter(by_first[n])) if n not in full and len(by_first.get(n, ())) == 1 else n  # noqa: E731
    return [(canon(s), r, canon(o)) for s, r, o in triples]


def precision_recall(found, truth):
    found, truth = set(found), set(truth)
    hit = len(found & truth)
    return hit / max(len(found), 1), hit / len(truth)


# ---------------------------------------------------------------------------
# 3. A tiny graph database with a Cypher-like language / 带类 Cypher 查询语言的小型图数据库
# ---------------------------------------------------------------------------
class Graph:
    """A property graph reduced to the essentials: labelled nodes and typed, directed edges, indexed by
    node and relation in both directions. 只保留要点的属性图：带标签的节点，带类型的有向边，双向索引。"""

    LABELS = {"BORN_IN": ("Person", "City"), "WORKS_AT": ("Person", "Company"), "KNOWS": ("Person", "Person"),
              "LOCATED_IN": ("Company", "City"), "FOUNDED_BY": ("Company", "Person"), "IN_COUNTRY": ("City", "Country")}

    def __init__(self, triples):
        self.out, self.inc, self.label = defaultdict(set), defaultdict(set), {}
        for s, r, o in triples:
            self.out[(s, r)].add(o)
            self.inc[(o, r)].add(s)
            ls, lo = self.LABELS[r]
            self.label.setdefault(s, ls)
            self.label.setdefault(o, lo)
        self.edges = len(set(triples))

    def nodes(self, label=None):
        return [n for n, lab in self.label.items() if label in (None, lab)]

    def step(self, node, rel, forward=True):
        return self.out[(node, rel)] if forward else self.inc[(node, rel)]

    def query(self, cypher):
        """MATCH (a:Label {name: "..."})-[:REL]->(b)<-[:REL]-(c:Label) RETURN c.name | RETURN count(c)
        Supports one chain of nodes and relations in either direction, a name filter on any node, and
        RETURN of one variable's names or count(DISTINCT ...). Enough to show how Cypher reads.
        只支持一条由节点和关系组成的链（任意方向）、任一节点上的名字过滤，以及返回一个变量的名字或去重计数。"""
        m = re.match(r"\s*MATCH\s+(.+?)\s+RETURN\s+(.+?)\s*$", cypher, re.S)
        pattern, ret = m.group(1), m.group(2)
        node_re = r"\((\w*)(?::(\w+))?(?:\s*\{name:\s*\"([^\"]+)\"\})?\)"
        nodes = [dict(var=v, label=l or None, name=n or None) for v, l, n in re.findall(node_re, pattern)]
        rels = [(r1 or r2, bool(r1)) for r1, r2 in re.findall(r"-\[:(\w+)\]->|<-\[:(\w+)\]-", pattern)]
        rows = [{nodes[0]["var"]: n} for n in self.nodes(nodes[0]["label"]) if nodes[0]["name"] in (None, n)]
        for (rel, forward), spec in zip(rels, nodes[1:]):
            prev = nodes[nodes.index(spec) - 1]["var"]
            rows = [dict(row, **{spec["var"]: n}) for row in rows for n in self.step(row[prev], rel, forward)
                    if spec["label"] in (None, self.label.get(n)) and spec["name"] in (None, n)]
        count = re.match(r"count\((?:DISTINCT\s+)?(\w+)\)", ret, re.I)
        if count:
            return len({row[count.group(1)] for row in rows})
        var = ret.split(".")[0]
        return sorted({row[var] for row in rows})


# ---------------------------------------------------------------------------
# 4. Multi-hop questions / 多跳问题
# ---------------------------------------------------------------------------
def tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25:
    """Keyword retrieval over whole documents (as in chapter 19). 文档级关键词检索（同 19 章）。"""

    def __init__(self, texts, k1=1.2, b=0.75):
        bags = [Counter(tokens(t)) for t in texts]
        vocab = {w: i for i, w in enumerate(sorted({w for bag in bags for w in bag}))}
        tf = torch.zeros(len(texts), len(vocab))
        for i, bag in enumerate(bags):
            for w, c in bag.items():
                tf[i, vocab[w]] = c
        lengths = tf.sum(1)
        df = (tf > 0).sum(0).float()
        idf = torch.log(1 + (len(texts) - df + 0.5) / (df + 0.5))
        self.vocab, self.w = vocab, idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * lengths / lengths.mean())[:, None])

    def top(self, query, k):
        q = torch.zeros(len(self.vocab))
        for w in set(tokens(query)):
            if w in self.vocab:
                q[self.vocab[w]] = 1.0
        return (self.w @ q).argsort(descending=True)[:k].tolist()


def hop_questions(world, n=200, seed=2):
    """(question, chain of documents needed, answer, Cypher query) for 1, 2 and 3 hops.
    1-3 跳的问题：问题、需要的文档链、答案、对应的 Cypher 查询。"""
    rng = random.Random(seed)
    people, companies, cities = world["people"], world["companies"], world["cities"]
    out = {1: [], 2: [], 3: []}
    for p in rng.sample(list(people), n):
        c = people[p]["works"]
        city = companies[c]["city"]
        out[1].append((f"Where was {p} born?", [p], people[p]["born"],
                       f'MATCH (p:Person {{name: "{p}"}})-[:BORN_IN]->(c:City) RETURN c.name'))
        out[2].append((f"In which city is the company that {p} works for headquartered?", [p, c], city,
                       f'MATCH (p:Person {{name: "{p}"}})-[:WORKS_AT]->(co:Company)-[:LOCATED_IN]->(c:City) RETURN c.name'))
        out[3].append((f"In which country is the company that {p} works for headquartered?", [p, c, city], cities[city],
                       f'MATCH (p:Person {{name: "{p}"}})-[:WORKS_AT]->(co:Company)-[:LOCATED_IN]->(c:City)-[:IN_COUNTRY]->(k:Country) RETURN k.name'))
    return out


def iterative_retrieval(bm25, docs, question, hops, k=5):
    """Retrieve hop by hop: search with the question, read the top document, pull out the next entity
    with the extraction patterns, and search again with it (like self-ask or IRCoT, minus the LLM).
    逐跳检索：用问题检索，读最靠前的文档，用抽取规则找出下一个实体，再用它检索（类似 self-ask / IRCoT，只是没有大模型）。"""
    found, query = [], question
    for h in range(hops):
        hits = bm25.top(query, k)
        found += hits
        if h == hops - 1:
            break
        rel = ["WORKS_AT", "LOCATED_IN"][h]
        nxt = [o for i in hits for s, r, o in resolve(extract([docs[i]])) if r == rel]  # first hit that has it / 第一个含该关系的结果
        if not nxt:
            break
        query = nxt[0]
    return found


def evidence_recall(found, docs, chain):
    titles = {docs[i][0] for i in found}
    return all(t in titles for t in chain)


# ---------------------------------------------------------------------------
# 5. Global questions and GraphRAG / 全局问题与 GraphRAG
# ---------------------------------------------------------------------------
def communities(graph, iters=20, seed=0):
    """Label propagation: every node repeatedly takes the most common label among its neighbours.
    GraphRAG uses the Leiden algorithm; the idea is the same: densely linked groups.
    标签传播：每个节点反复采用邻居中最常见的标签。GraphRAG 用 Leiden 算法，思想相同：找出连接紧密的群体。"""
    rng = random.Random(seed)
    nbrs = defaultdict(set)
    for (s, r), objs in graph.out.items():
        if r in ("WORKS_AT", "KNOWS"):
            for o in objs:
                nbrs[s].add(o)
                nbrs[o].add(s)
    label = {n: n for n in nbrs}
    nodes = list(nbrs)
    for _ in range(iters):
        rng.shuffle(nodes)
        for n in nodes:
            counts = Counter(label[m] for m in nbrs[n])
            best = max(counts.values())
            label[n] = min(l for l, c in counts.items() if c == best)
    groups = defaultdict(list)
    for n, l in label.items():
        groups[l].append(n)
    return sorted(groups.values(), key=len, reverse=True)


def summarize(graph, group):
    """A community report as GraphRAG would ask an LLM to write it; here filled in from the graph.
    GraphRAG 让大模型写的社区报告；这里直接从图里填。"""
    people = [n for n in group if graph.label[n] == "Person"]
    employers = Counter(next(iter(graph.step(p, "WORKS_AT")), None) for p in people)
    top, n = employers.most_common(1)[0]
    city = next(iter(graph.step(top, "LOCATED_IN")), "?")
    return dict(size=len(people), employers=dict(employers), text=f"{len(people)} people, {n} of them at {top} ({city}).")


def main():
    print("== 1. a small world, written as documents / 写成文档的小世界 ==")
    world = make_world()
    docs = documents(world)
    print(f"  {len(world['people'])} people, {len(world['companies'])} companies, {len(world['cities'])} cities, "
          f"{len(world['facts'])} true facts, {len(docs)} documents")
    p0 = next(iter(world["people"]))
    c0 = world["people"][p0]["works"]
    print("  " + next(t for n, t in docs if n == p0))
    print("  " + next(t for n, t in docs if n == c0))

    print("\n== 2. building the graph / 建图 ==")
    raw = extract(docs)
    pr_raw = precision_recall(raw, world["facts"])
    triples = resolve(raw)
    pr = precision_recall(triples, world["facts"])
    print(f"  extracted {len(raw)} triples: precision {pr_raw[0]:.0%}, recall {pr_raw[1]:.0%} before entity resolution; "
          f"{pr[0]:.0%} / {pr[1]:.0%} after")
    assert pr_raw[0] < 0.95 and pr[0] > 0.99 and pr[1] > 0.99
    graph, raw_graph = Graph(triples), Graph(raw)
    print(f"  graph: {len(graph.label)} nodes, {graph.edges} edges (without resolution: {len(raw_graph.label)} nodes)")

    print("\n== 3. Cypher-like queries / 类 Cypher 查询 ==")
    q = f'MATCH (p:Person {{name: "{p0}"}})-[:WORKS_AT]->(co:Company)-[:LOCATED_IN]->(c:City) RETURN c.name'
    print(f"  {q}\n    -> {graph.query(q)}")
    q = f'MATCH (p:Person)-[:WORKS_AT]->(co:Company {{name: "{c0}"}}) RETURN count(DISTINCT p)'
    print(f"  {q}\n    -> {graph.query(q)}")
    assert graph.query(q) == sum(a["works"] == c0 for a in world["people"].values())

    print("\n== 4. multi-hop questions: does the evidence reach the model? / 多跳问题：证据能不能送到模型面前 ==")
    bm25 = BM25([t for _, t in docs])
    results = {}
    for hops, qs in hop_questions(world).items():
        single = sum(evidence_recall(bm25.top(qn, 5), docs, chain) for qn, chain, _, _ in qs) / len(qs)
        iterative = sum(evidence_recall(iterative_retrieval(bm25, docs, qn, hops), docs, chain) for qn, chain, _, _ in qs) / len(qs)
        graph_ok = sum(graph.query(cy) == [ans] for _, _, ans, cy in qs) / len(qs)
        raw_ok = sum(raw_graph.query(cy) == [ans] for _, _, ans, cy in qs) / len(qs)
        results[hops] = (single, iterative, graph_ok, raw_ok)
        print(f"  {hops} hop(s): one BM25 search (top 5) {single:.0%}, hop-by-hop search {iterative:.0%}, "
              f"graph query {graph_ok:.0%} (graph without entity resolution {raw_ok:.0%})")
    assert results[1][0] > 0.9 and results[3][0] < 0.3                 # one search can't follow the chain / 一次检索跟不上链条
    assert results[3][1] > results[3][0] + 0.4 and results[3][2] > 0.99  # following it hop by hop works / 逐跳就能跟上
    assert results[2][3] < results[2][2] - 0.2                          # resolution matters / 实体消解很重要

    print("\n== 5. global questions and GraphRAG / 全局问题与 GraphRAG ==")
    truth = Counter(a["works"] for a in world["people"].values())
    biggest, n_big = truth.most_common(1)[0]
    counted = graph.query(f'MATCH (p:Person)-[:WORKS_AT]->(co:Company {{name: "{biggest}"}}) RETURN count(DISTINCT p)')
    seen = sum(1 for i in bm25.top(f"works at {biggest}", 10) if f"works at {biggest}" in docs[i][1] or f"works at {alias(biggest)}." in docs[i][1])
    print(f'  "How many people work at {biggest}?" truth {n_big}; graph count {counted}; counting in the top 10 retrieved documents: {seen}')
    assert counted == n_big and seen <= 10 < n_big
    groups = communities(graph)
    reports = [summarize(graph, g) for g in groups if any(graph.label[n] == "Person" for n in g)]
    print(f"  label propagation finds {len(reports)} communities; the largest: {reports[0]['text']}")
    # map-reduce over community reports: each report contributes its employer counts / 对社区报告做 map-reduce
    totals = Counter()
    for r in reports:
        totals.update(r["employers"])
    print(f'  "Which company employs the most people?" map-reduce over community reports -> {totals.most_common(1)[0][0]} '
          f"({totals.most_common(1)[0][1]}); truth {biggest} ({n_big})")
    assert totals.most_common(1)[0] == (biggest, n_big)
    purity = sum(max(Counter(r["employers"]).values()) for r in reports) / sum(r["size"] for r in reports)
    print(f"  {purity:.0%} of people share their community's main employer (friends are mostly colleagues)")


if __name__ == "__main__":
    main()
