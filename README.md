# news-digest

It reads your newsletter inbox, pulls the articles out, throws away the junk,
groups what is about the same thing, and writes one ranked summary you can read
in ten minutes.

The goal is not to reproduce ten newsletters. It is to answer one question:
**what happened in the world this week?**

## What it does

1. Read the inbox. Read-only -- nothing is marked or deleted.
2. Work out which newsletter each email is.
3. Pull the individual articles out of each email.
4. Unwrap the tracking links to find the real article URLs.
5. Bin the junk: sentence fragments, boilerplate, mastheads, sport.
6. Drop articles already stored.
7. Ask the model what each article is about, and drop what is not news.
8. Turn each one into numbers, so they can be compared across languages.
9. Group articles covering the same event.
10. Score and rank the groups.
11. Write a short summary for each.
12. Group related stories into bigger narratives.
13. Pick the top 12 plus 5 extras, without letting one region or topic take over.
14. Write the Markdown file and update this README.

About 3-5 minutes on Gemini, about 13 locally on Ollama.

It runs at zero cost with no third-party API at all: local embeddings via ONNX,
a local model via Ollama. See [Providers](doc/providers.md).

## Quick start

You need an inbox that receives the newsletters. Subscribe from a dedicated
address -- everything in the box is read, so personal mail there is just noise.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env          # then fill in NEWS_EMAIL_*
```

Check that the source rules match your mail before anything else. This is the
step that bites: the rules look right, and nothing matches.

```bash
news-digest sources --check   # fix config/sources.yaml until no source says NONE
```

Then:

```bash
news-digest run --week $(date -u +%G-W%V)
```

With no model or API key at all, a couple of minutes end to end:

```bash
news-digest run --no-llm --no-embeddings
```

## Documentation

| | |
|---|---|
| **[Setup](doc/setup.md)** | The inbox, the credentials, running locally, deploying to GitHub |
| **[Providers](doc/providers.md)** | Embeddings and LLM: local, Gemini or Ollama, and what degrades without each |
| **[Configuration](doc/configuration.md)** | Sources and match rules, filters, the ranking formula, digest size, retention |
| **[How it works](doc/pipeline.md)** | Each stage in turn, the multilingual design, and the known limits |
| **[Design notes](doc/design-notes.md)** | Why the pipeline is shaped this way, and which numbers are still guesses |
| **[CLI](doc/cli.md)** | Every command, plus recipes for trying things without breaking your digest |
| **[Attribution](doc/attribution.md)** | What is stored, what is published, and what is deliberately not |
| **[Rebuilding](doc/rebuilding.md)** | Starting over, what must survive, and the failure modes that look like bugs |
| **[Development](doc/development.md)** | Tests, layout, adding a provider, schema changes |

## This week

<!-- digest:start -->

# The Week in Global News
28 Sep – 4 Oct 2026

*Synthesized from ten newsletters, in two languages*

## 1. La crisis de vivienda en España se intensifica con nuevos desahucios

*La crisis de vivienda en España*

El desahucio de Maricarmen ha reavivado la crisis de vivienda en España, según El Orden Mundial, The Guardian. Morgan Stanley, según Público, se ha convertido en uno de los principales responsables de los desahucios en el país. Expertos analizan los decretos de vivienda, destacando avances y riesgos, según Público.

**Why it matters:** La situación de los desahucios y las medidas de vivienda sigue sin resolver.

**Not answered by this week's coverage:**
- ¿Cuál es el impacto real de los desahucios en la población vulnerable?
- ¿Qué papel juegan las grandes instituciones financieras en la crisis de vivienda?
- ¿Cómo se evalúan los riesgos de los decretos de vivienda en la solución a largo plazo?

In this story:
- España enfrenta crisis de vivienda tras desahucio de anciana — [El Orden Mundial](https://cxk-504.na1.hubspotlinks.com/Ctc/OQ+113/cXK-504/VX0R6-55G6XLN9llC2M6d3L4W2QY3Nh5VF2VYMJvQnP3qn9qW8wLKSR6lZ3nNW8-43jd4MQf56W2l_lbx8zWJf4W2T8hx08rs3T-W7pHPgy8ZCfkHW9d0nG-1y5MfyW5HRC9299RX-JVHRvTL7N7WCVW87Bcrr5-9VSxW3DRzv06_Mt5pW1f25-k94r74HW97fP1R6TcprKN1XdQ3BKBJ7HN8CRd8Z2V4PJW33PV4b70HyNCW5--xZx2K21MpW16tjfx4zBK4QW49XHCC6zPcfnW4c0D795tR1yLW5tW_h51qRl9PW7vTylP6GZHd_W5ZrB982rYDwmW4g9Jzw2Fh7PcMhwYd0NzY-gVYm7W429LXPxW3H-9vD16SMfJW6xgKyY6tYC5xW1plxJs7Yvd_nW2sCBSL1T56-fdYrKln04), [The Guardian](https://support.theguardian.com/eu/contribute?CMP_BUNIT=edtrl&CMP_TU=mawns)
- El brazo inmobiliario de Morgan Stanley se convierte en uno de los grandes desahuciadores en España — [Público](https://www.publico.es/politica/brazo-inmobiliario-morgan-stanley-convierte-grandes-desahuciadores-espana.html?segment=registrados&tpcc=nl_temas0625&pnespid=B_cr7hxb7TgakVeEsYHBFxBCvwxj0rhqugQTQfkDMcHKSC1KbjVdlco3_DJad1STOfY2XbTPJQ)
- Avances de calado, medidas temporales y una pieza clave en riesgo: expertos analizan los decretos de vivienda — [Público](https://www.publico.es/economia/avances-calado-medidas-temporales-pieza-clave-riesgo-expertos-analizan-decretos-vivienda.html?segment=registrados&tpcc=nl_enpocaspalabrasreg0626&pnespid=XuU77B5K8nQAiFKR9sHUGxtOpxMhmuhypxocF7pYPs.Ky5RfARZ.XJlX6oxacsOhRynzRwTnkA)

## 2. La guerra en Ucrania intensifica su impacto en la población civil

La ciudad de Kiev se prepara para el invierno con escasez de energía y la presencia de drones, según The Guardian. En paralelo, los residentes de Oleshky, ocupada por Rusia, enfrentan condiciones extremas y se ven obligados a comer hierbas para sobrevivir, según también The Guardian.

**Why it matters:** La situación humanitaria en zonas ocupadas y la preparación para el invierno en ciudades clave siguen sin resolverse.

**Not answered by this week's coverage:**
- ¿Cuál será el impacto del invierno en la infraestructura energética de Kiev?
- ¿Cómo se está abordando la crisis alimentaria en las zonas ocupadas por Rusia?
- ¿Qué medidas están tomando las autoridades ucranianas para garantizar la seguridad energética durante el invierno?

In this story:
- Kyiv has become a frontline city – and winter is on the way — [The Guardian](https://www.theguardian.com/world/2026/sep/26/after-summer-respite-kyiv-winter-dread)
- ‘Hell on earth’: last residents eat weeds to survive in Russian-occupied Oleshky — [The Guardian](https://www.theguardian.com/world/2026/sep/27/hell-on-earth-last-residents-eat-weeds-to-survive-in-russian-occupied-oleshky)

## 3. Todas las guerras de Etiopía: qué hay detrás del nuevo conflicto armado

El conflicto en Etiopía involucra divisiones regionales e internacionales.

**Why it matters:** La situación en Etiopía refleja tensiones geopolíticas y regionales.

Sources:
- [El Orden Mundial](https://cxk-504.na1.hubspotlinks.com/Ctc/OQ+113/cXK-504/VX0R6-55G6XLN9llC2M6d3L4W2QY3Nh5VF2VYMJvQp43qn9qW95jsWP6lZ3mkW78XzJ51PBb83W7_25Nb5QMftVW98Vn7T5LhkGpW4Ly7Z46CG0tsN2fY_cTjjNL2MgZB0ccv-hBW4G0NKH87ZXcXVmF_dw2dZmDNW2t3GFV9jDDVDW6g2cbc7bGMFzW2QXF_b5XPwrPW6wBs2j11zFbFW1XCThv1ZXfgCW5LJZV17Y07xzW5Z366y7VgSwhW5P3ynB4GYPKYW98T8Pq6RY7ZZW4Lk6Sv90WVZ-W4lLKG954-F9jW57Vf2K2qN6wnF537PkbXknwN1CMXL2mBQ4kW5L6MYB2G3MldW3jKGwc1sf6xPN3QMm6YtdXY-W1L9wPQ7JZvCYW6zQ3vm3ZBwXFW5wC3fc55xkypW3J_XMQ7LWYJ6N2hLBWyZk4Wcf8k56DP04)

## 4. Indian firm building $15bn Trump-announced steel mill has deep Russia ties

Una empresa india con fuertes vínculos rusos está construyendo una acería en Estados Unidos.

**Why it matters:** Las inversiones rusas pueden tener un impacto en la economía estadounidense.

Sources:
- [Al Jazeera](https://7aet5.r.a.d.sendibm1.com/mk/cl/f/sh/7nVU1aA2nfwFS2Kg5hF8Cra9EjYQrye/RvxHmrbNt2IC)

## 5. Un adelanto electoral con ventajas para el PSOE pero riesgos para la izquierda: las cuentas que sopesa Sánchez

El PSOE evalúa un adelanto electoral con ventajas pero riesgos para la izquierda.

**Why it matters:** Las decisiones políticas pueden afectar la estabilidad del partido y la izquierda en general.

Sources:
- [Público](https://www.publico.es/politica/gobierno/adelanto-electoral-ventajas-psoe-pero-riesgos-izquierda-cuentas-sopesa-sanchez.html?segment=registrados&tpcc=nl_temas0625&pnespid=G7ko.VAF4iMYlVKN_c6MCU9GsEol2LN49wldA6cfa5zKNgElUfXz_o5GRO1fEN9rR0OJAcZm6A)

## 6. “No imagino una Cuba multipartidista a corto plazo”: Carlos Alzugaray, exembajador cubano

Un exembajador cubano comenta la situación política de Cuba y su relación con Estados Unidos.

**Why it matters:** Las relaciones internacionales son clave para entender la política cubana.

Sources:
- [El Orden Mundial](https://cxk-504.na1.hubspotlinks.com/Ctc/OQ+113/cXK-504/VX0R6-55G6XLN9llC2M6d3L4W2QY3Nh5VF2VYMJvQm05nXHCW50kH_H6lZ3pQW5_RB1x5Z0c_HN8vRwqS5M5-FW3HBysD4n0q6cW392-bb25hKtzW61pDw47yhF4cW3ZSJdZ7NjBqfW7lfl3B6kRhQFW4rY1kV92NVvgW8bYdbL92Bv5sW19JxXk8wMZ8gVWBMgT7pBGdxW4pCgLw1tvdHQW2Jmnwp5szngSW58187Y2rWC79W50R0hk2-kSTtW78wlGc5T7vM4N963GyttDgqDVGS0ph7gKGN_W7l5fc341Z13QW2QxMQX4Yl2pLW1WsTmn3mXvhJW7KFy_78zf_r6W53Xn1y3QFG3RW79Xnqt22BQlkW7w5rJM68MN80W769QK_7KMH_PW7FNyz_6xBzp9W3pr2Wk5rxv84VNktY-2v_drRW8SMVWp8mcWfDW5MjNDb3Wn1J1W7ZSntH67Z8c7f5h7fGl04)

## 7. Tecnoptimismo, o cómo Silicon Valley quiere convencernos de que ellos deben gobernar el futuro

El tecnoptimismo de Silicon Valley se basa en la idea de que la tecnología debe gobernar el futuro.

**Why it matters:** La visión tecnológica puede influir en las decisiones políticas y económicas.

Sources:
- [El Orden Mundial](https://cxk-504.na1.hubspotlinks.com/Ctc/OQ+113/cXK-504/VX0R6-55G6XLN9llC2M6d3L4W2QY3Nh5VF2VYMJvQm05nXHCW50kH_H6lZ3mrV3YdlX5lHb2TW5N7tY61hy4QsW44mQ2w65zF0zW4yfnML8v7TCLVXR5vR27nWfbW2_Zdf04-fNfBVkwpdT1SYHptW4Dlmb_7ZjLKCW8X9Vb520PmcLW2WPGQT460WNSW6L3r-04X71dkW7pwH5c986BJyW54n4W23FStTgW1N3-Z96kBb0CN2qFVgYQP30KW4v_51539clv2W7w7wJK8pMC_7W5xvkzT1RcltbW2SsPWW5MGhWSW4JTFg94p_P5LW6RfCmC7_R3-gW2_9xhS3qbtTWVBY5Mt77sPJ9VDK80Q6hDCgvW5_Yg8s3Vl55-W2wY7jP7RVs5tN6tWzCJV1WqsN6ZZ80hy8WmfN1p4Qjx_DbHPW1PLRlw8761FYW6wrpPN7s6MVtW5k1CdJ3zDw6-f7bQd-j04)

## 8. Satellite images show Gaza in ruins three years into Israel’s genocidal war

Imágenes satelitales muestran el estado de destrucción de Gaza tras tres años de guerra israelí, con más de 74,000 muertos.

**Why it matters:** Refleja la gravedad de la crisis humanitaria en la región.

Sources:
- [Al Jazeera](https://7aet5.r.a.d.sendibm1.com/mk/cl/f/sh/7nVU1aA2ng01R5LI2pi2HaDlZSY0Lrg/d6YlvmOYIVJm)

## 9. Mapas para entender las elecciones en Brasil

Lula da Silva y Flávio Bolsonaro se enfrentan en elecciones marcadas por el crecimiento de la ultraderecha y el evangelismo.

**Why it matters:** Refleja la polarización política en Brasil y su impacto regional.

Sources:
- [El Orden Mundial](https://cxk-504.na1.hubspotlinks.com/Ctc/OQ+113/cXK-504/VX0R6-55G6XLN9llC2M6d3L4W2QY3Nh5VF2VYMJvQp43qn9qW95jsWP6lZ3kzN68wM8pxjc7HV3lbjB826hMDW8C260b2pd9_lW5-9Sss97p3mDW6v0PwN8Sq-tLW45FNBw9jPDzHW40zLCV76c-S_W527JF99f2DZnW7y0LxK3xhTD7W7jc54x7BxZXqVNYTbl3mHmkpW6l2qyH8q8vthW3yTlG17jTK1WVmwHh_5TGTLCW5nCPZW8cyX1MW4wyyrn8nhWjdW4jtcly63DQmlW6N-fYc8vcYsrW4sDHRR71p3ghW3SvqNC4YxLTNW4FV07q4f8DvZW1sM88n98Km2lW1WMFnb5BTbjmW6Qbt9D6ZC2bXW4GhLkl8cBbj9W18ydfF8HwDFnVpT-LN3MlZQ4W50l4v-3pVfmKW3VlzH48czmcfN62KKp3xP6Wbddxg-W04)

---

## Also worth knowing

- [Maricarmen y su desahucio, ¿la "chispa" que necesitan las izquierdas para recuperar la calle?](https://www.publico.es/politica/maricarmen-desahucio-chispa-necesitan-izquierdas-recuperar-calle.html?segment=registrados&tpcc=nl_temas0625&pnespid=HK87qBgL6T8biF2Ku5jOGUUSoxQ_wqVmoVtVBvwJLIvKW0spfgkmXvN8nGicco0X48Y_GyhexQ) — Público
- [La votación exprés de los decretos obligará a las derechas a retratarse en plena oleada de indignación por la vivienda](https://www.publico.es/politica/votacion-expres-decretos-obligara-derechas-retratarse-plena-oleada-indignacion-vivienda.html?segment=registrados&tpcc=nl_enpocaspalabrasreg0626&pnespid=ALoo8VtT7z5Ayw7JsszSFxpFp0tkn_oor1pdR78FaZ3Kx8Me4AvOYQH0Q5G4tLZm4TF.xbPsRw) — Público
- [Estonia blames Russia for arson attack](https://www.theguardian.com/world/2026/sep/29/estonia-blames-russia-arson-attack-drone-maker-ukraine) — The Guardian
- [El Gobierno madrileño y el Poder Judicial firmaron en 2019 un convenio para hacer frente a los desahucios que nunca se aplicó](https://www.publico.es/economia/vivienda/gobierno-madrileno-poder-judicial-firmaron-2019-convenio-frente-desahucios-nunca-aplico.html?segment=registrados&tpcc=nl_temas0625&pnespid=GOcy_1hG6ysbz1iR5oDOBBtBvw8y26tusVwQTaUeKpHKVOl61pmlvhnZSWVcm7fA_MkHaoVvgA) — Público
- [Polonia, el nuevo muro de la UE para defenderse de Rusia](https://cxk-504.na1.hubspotlinks.com/Ctc/OQ+113/cXK-504/VX0R6-55G6XLN9llC2M6d3L4W2QY3Nh5VF2VYMJvQp43qn9qW95jsWP6lZ3kRVksz5v8LknkjVjpV9s4jxw3QW9lFlch80H_j5W88Y-1640l7DrN2NqHQK4Nrz5W1SfFlc6_BSJxN29znkyDZqfVW2mfyWw9968RRW7JpsbQ5cY_BZW1b6tcS5D0SBlW8m38Rx2CbJtGW8dBJxR3YhF3bVzG40p8JK8Y2VnJvsc6wCMY8W6HQzwz27Tdj3W47LRjl2yjWxvW8Nmklh67PjCsW3S0QZk5Gf9vJW5G_Jjr4VvJCCW7Bxy1H312X3GW5jkDG86mkcpMVXrvBb1w60NJW6G8srB4QdqGQVZ-s397Kfy5ZW28lpkB3Y4WDHW3mL1-243L-bgW3Sk6_L8xXrnvN8lZD_7X6b4sW8Q9sNw1sQlRDV_FqxB5cqzjWf91sLQ204) — El Orden Mundial

---

*Built from 12 stories, 20 items, 4 publishers, en/es.*  
*Sources: Al Jazeera, El Orden Mundial, Público, The Guardian.*  
*10 items filtered as not news.*  
*11 of 12 stories rest on a single publisher.*  
*Silent this week: economist-weekly, el-salto, eldiario-catalunya, eldiario-director, elpais-weekly, guardian-saturday, reuters-world, rne-7dias, semafor-flagship (9 configured sources contributed nothing).*

<!-- digest:end -->
