# Visualizations & Hypotheses

Each visualization group below includes a motivating question or hypothesis.

## Group A: Paper & Upvote Volume
- **a1** Papers per day: *Has the rate of AI paper submissions to HF Daily Papers been accelerating, or has it plateaued?*
- **a2** Upvotes per day: *Are the community's total upvotes growing faster than paper volume, suggesting increased engagement per capita?*
- **a3** Upvote density (upvotes/paper): *Is the average quality bar (as measured by upvotes per paper) rising or falling as volume increases? A declining density could indicate dilution.*

## Group B: Cumulative Growth
- **b1** Unique authors: *How quickly is the research community expanding? Is the author pool growing linearly or exponentially?*
- **b2** Unique affiliations: *Are papers coming from an increasingly diverse set of institutions, or is research concentrating at a few labs?*
- **b3** Unique collaborations: *How interconnected is the community? A fast-growing collaboration count relative to author count would suggest a densely networked field.*
- **b4** Repeat collaborations: *Do authors tend to collaborate with the same people repeatedly, or is there high turnover in co-authorship? A high repeat rate suggests stable research teams.*
- **b5** Unique institution collaborations: *Are institutions forming new partnerships over time, or do the same labs keep collaborating?*
- **b6** Repeat institution collaborations: *Which institution pairs have deep, recurring partnerships?*

## Group C: Author Activity Distributions
- **c1** Papers per author: *What fraction of authors are "one-hit wonders" who appear on a single paper and never return? A heavy left tail would suggest high churn.*
- **c2** Time between papers: *How long does the typical author wait between papers? Short gaps might indicate prolific labs; long gaps might indicate one-off contributions.*
- **c3** Affiliations per author: *How mobile are researchers across institutions? Most authors having 1 affiliation suggests stability; 2+ suggests career movement or multi-affiliation patterns.*
- **c4** Last authors per first author: *How many distinct senior collaborators does a typical lead researcher work with? A high count suggests breadth of mentorship or project diversity.*
- **c5** Affiliations of first vs last authors: *Are first authors (typically junior) more mobile across institutions than last authors (typically senior)? Or do senior researchers accumulate more affiliations over longer careers?*
- **c6** First & last author overlap: *How many researchers serve as both first and last author on different papers? A high overlap might indicate a shift from junior to senior roles within the dataset's timeframe.*
- **c7** Authors per paper: *What is the typical team size? Are most papers solo efforts, small teams, or large collaborations?*
- **c8** Affiliations per paper: *How many institutions typically contribute to a single paper? More institutions per paper could indicate increasingly collaborative, cross-institutional research.*
- **c9** Papers per author table: *A tabular view of the papers-per-author distribution — how steep is the power-law tail?*
- **c10** First authors per last author: *How many distinct junior collaborators does a typical senior researcher mentor? A high count suggests an active advisor role.*
- **c11** Authors per paper table: *Tabular view of team size distribution — what fraction of papers are solo vs large collaborations?*
- **c12** First-author papers per first author table: *How many lead-author papers does a typical first author accumulate? Is the distribution even more skewed than overall papers-per-author?*
- **c13** Last-author papers per last author table: *How many supervisory-role papers does a typical last author accumulate?*
- **c14** Time between first-author papers: *How frequently do lead researchers produce first-author papers? Faster cadence may indicate prolific junior researchers.*
- **c15** Time between last-author papers: *How frequently do senior researchers publish as last author? A faster cadence likely indicates a large active lab.*

## Group D: Top Authors & Affiliations
- **d1** Top authors: *Who are the most prolific and impactful contributors to the HF Daily Papers ecosystem?*
- **d2** Top affiliations: *Which institutions dominate in volume and impact? Is there a long tail or a clear oligopoly?*
- **d3** Top first authors: *First authorship typically indicates the primary contributor. Who are the most prolific lead researchers?*
- **d4** Top last authors: *Last authorship often indicates senior/supervisory roles. Who are the most active research advisors?*
- **d5** Papers per affiliation: *Complete ranking of institutions by paper count — reveals the full distribution from powerhouses to one-paper labs.*
- **d6** Top author collaborations: *Which author pairs publish together most frequently? Identifies the strongest research partnerships.*
- **d7** Top institution collaborations: *Which institution pairs co-author papers most frequently? Reveals structural inter-institutional partnerships.*

## Group E: Chinese vs Non-Chinese Analysis
- **e1-e3** Chinese authors/first/last: *What share of HF Daily Papers come from Chinese-name authors, and is this share growing?*
- **e4-e6** Non-Chinese authors/first/last: *How does the non-Chinese author contribution compare in volume and upvote reception?*
- **e7** Mixed authorship: *How common are cross-cultural collaborations, and do they receive more or fewer upvotes than single-origin papers?*
- **e8-e10** Affiliation-based: *Using institutional affiliation instead of names — do China-based institutions show different trends than name-based classification? This cross-checks the name heuristic.*
- **e11-e13** Time between papers (Chinese vs non-Chinese): *Do Chinese-name authors publish more frequently than non-Chinese authors? Does this differ for first authors vs last authors?*
- **e14-e16** Affiliations per author (Chinese vs non-Chinese): *Are Chinese-name authors more or less mobile across institutions compared to non-Chinese authors?*
- **e17** Non-Chinese author affiliation breakdown: *How many non-Chinese-name authors work at Chinese institutions, non-Chinese institutions, or both? This reveals cross-border talent flows.*
- **e18** Chinese author affiliation breakdown: *How many Chinese-name authors work at non-Chinese institutions? A high share could indicate brain drain or international mobility.*
- **e19-e20** Last co-author origin breakdown: *Do non-Chinese/Chinese authors tend to publish with Chinese or non-Chinese last (senior) co-authors? Reveals mentorship/supervision patterns across cultural lines.*
- **e21-e22** Multi-paper author affiliation breakdown: *Same as e17-e18 but restricted to multi-paper authors, where having multiple affiliations is more meaningful.*
- **e23-e24** Multi-paper author last co-author breakdown: *Same as e19-e20 but restricted to multi-paper authors.*

## Group F: Correlation Analysis
- **f1** # authors vs upvotes: *Do papers with more authors (larger teams) receive more upvotes? Large collaborations might signal more resources and polish.*
- **f2** Title length vs upvotes: *Are shorter, punchier titles associated with more upvotes? Marketing intuition says yes.*
- **f3** Abstract length vs upvotes: *Is there an optimal abstract length, or do longer/shorter abstracts correlate with engagement?*
- **f4** # institutions vs upvotes: *Do cross-institutional collaborations perform better in terms of community reception?*
- **f5** Upvote density by team size: *Is there an optimal team size for maximizing community engagement? Do very large teams or solo authors get more attention?*

## Group G: Institution Analysis
- **g1** Authors per institution: *Which institutions have the deepest bench of AI researchers publishing on HF Daily Papers?*

## Group H: Author Summary & Exhaustive Table
- **h1** Author summary statistics: *What percentage of authors are first authors, last authors, solo authors, Chinese-name, single-paper, multi-affiliation, etc.? A demographic snapshot of the research community.*
- **h2** Exhaustive author table: *A complete per-author dataset with all computed metadata and list of affiliations — suitable for downstream analysis, filtering, and cross-referencing.*
- **h3** Exhaustive surnames table: *All unique last names, paper count, author count, and whether the name is classified as Chinese. Useful for validating the surname heuristic.*
- **h4** Exhaustive affiliations table: *All unique affiliations, paper count, author count, and whether the institution is classified as Chinese. Useful for validating the affiliation heuristic.*

## TODOs

De-duplicate the affiliations
Assign Chinese or non-Chinese to last names and affiliations to formalize the maps, and use the maps formally in the analysis.