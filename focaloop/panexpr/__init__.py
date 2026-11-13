"""PanExpr: zero-LLM focused-streaming RNA-seq quantification for prokaryotes and spliced eukaryotes.

Modules
-------
manifest    ENA run manifests (PE/SE, RNA-seq/WGS) with exact byte sizes
catalog     Unified pangenome (multi-strain CDS + decoys) and spliced transcriptome targets
streamer    Byte-budget FASTQ streaming from local files or live HTTP 206 ranges
kernels     Mapping kernels (mappy, bwa-mem2, bowtie2, strobealign, minimap2, salmon, kallisto, STAR)
tournament  Startup empirical kernel tournament with the mate-tie symmetry gate
solver      Pass-2 cohort EM / WGS conditioning, strandedness, PyDESeq2, calibrated uncertainty
loop        Variance-guided focused loop with the tech_var << bio_var stopping rule
zoom        Borderline-gene 31-mer bait zoom
baselines   Arms A-E helpers (100% align, 100% pseudo, matched-byte naive subsample)
metrics     Stratified concordance metrics and the standardized study JSON emitter

No module imports or calls any large language model: see tests/test_zero_llm.py.
"""

__version__ = "2.3.0"
