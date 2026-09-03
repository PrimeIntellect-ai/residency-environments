# Data sources and attribution

The environment includes processed examples derived from four public research
resources. Source licenses and citation requirements continue to apply to the
derived data independently of the repository's code license.

## LINCS L1000

LINCS-funded Connectivity Map L1000 data are deposited in GEO. The environment
uses Level 5 MODZ signatures for six core cell lines.

- Data access: https://clue.io/connectopedia/pdf/lincs_cmap_data
- GEO accession: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE92742
- GEO reuse terms: https://www.ncbi.nlm.nih.gov/geo/info/disclaimer.html
- Citation: Subramanian et al., *Cell* 2017,
  https://doi.org/10.1016/j.cell.2017.10.049

## DepMap PRISM

The viability endpoint uses the PRISM Repurposing 24Q2 Extended Primary data.
DepMap-generated public-release data are distributed under CC BY 4.0; release-
specific terms should also be checked on the download page.

- Release data and license: https://figshare.com/articles/dataset/Repurposing_Public_24Q2/25917643
- Data portal: https://depmap.org/portal/data_page/
- Release citation: DepMap, Broad (2024), *DepMap 24Q2 Public*
- PRISM citation: Corsello et al., *Nature Cancer* 2020,
  https://doi.org/10.1038/s43018-019-0018-6

## Drug Repurposing Hub

Compound names, structures, targets, and mechanisms come from the Drug
Repurposing Hub. Its metadata are available under CC BY 4.0 for research use;
the source also states restrictions on clinical-treatment and commercial-
marketing use.

- Data and terms: https://repo-hub.broadinstitute.org/repurposing
- Citation: Corsello et al., *Nature Medicine* 2017,
  https://doi.org/10.1038/nm.4306

## MSigDB Hallmark gene sets

Signed pathway targets use the Hallmark collection from MSigDB 2024.1. Current
MSigDB releases in this range are available under CC BY 4.0, subject to the
source's listed collection-specific terms.

- License terms: https://www.gsea-msigdb.org/gsea/msigdb_license_terms.jsp
- Citation: Liberzon et al., *Cell Systems* 2015,
  https://doi.org/10.1016/j.cels.2015.12.004

## Derived files

| File | Contents |
| --- | --- |
| Hugging Face `data/train.parquet` and `data/test.parquet` | prompts, answers, splits, and assay provenance; evaluator-only |
| `compound_table.parquet` | compound identity and chemistry lookup table |
| `hallmark_names.json` | Hallmark ontology names used for scoring; no gene memberships |

The repository does not include raw source downloads. The original
[public construction pipeline](https://github.com/swpo/drug-perturbation-rl/tree/main/scripts)
produced the derived example table. `scripts/drug-perturbation/prepare_dataset.py`
in residency-environments repackages that frozen public table without changing
its rows. The original table SHA-256 is
`9f1cedc98d93f84069d4ed823325adb6e03fca26c611ca42e11628d057b9c048`.
