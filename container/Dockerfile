# ghcr.io/galaxyproject/focaloop:0.1.0
#
# Build from the repository root:
#   docker build -f container/Dockerfile -t ghcr.io/galaxyproject/focaloop:0.1.0 .
# Apptainer (Galaxy job runners on garage / cluster):
#   apptainer build focaloop_0.1.0.sif docker-daemon://ghcr.io/galaxyproject/focaloop:0.1.0
FROM mambaorg/micromamba:2.0.5

LABEL org.opencontainers.image.title="focaloop" \
      org.opencontainers.image.version="0.1.0" \
      org.opencontainers.image.description="Byte-budgeted focal-loop analysis of INSDC reads with Galaxy blueprints" \
      org.opencontainers.image.source="https://github.com/galaxyproject/focaloop" \
      org.opencontainers.image.licenses="MIT"

COPY --chown=$MAMBA_USER:$MAMBA_USER container/environment.yml /tmp/environment.yml
RUN micromamba install -y -n base -f /tmp/environment.yml \
    && micromamba clean --all --yes

ARG MAMBA_DOCKERFILE_ACTIVATE=1
COPY --chown=$MAMBA_USER:$MAMBA_USER pyproject.toml README.md /opt/focaloop/src/
COPY --chown=$MAMBA_USER:$MAMBA_USER focaloop /opt/focaloop/src/focaloop
RUN python -m pip install --no-deps --no-cache-dir /opt/focaloop/src

# Builtin blueprints: `focaloop run --blueprint panexpr` resolves /opt/focaloop/blueprints/panexpr/
COPY --chown=$MAMBA_USER:$MAMBA_USER blueprints/panexpr/blueprint.focaloop.yml /opt/focaloop/blueprints/panexpr/
COPY --chown=$MAMBA_USER:$MAMBA_USER blueprints/malariavar/blueprint.focaloop.yml /opt/focaloop/blueprints/malariavar/
COPY --chown=$MAMBA_USER:$MAMBA_USER blueprints/focaloop_query/blueprint.focaloop.yml /opt/focaloop/blueprints/focaloop_query/
COPY --chown=$MAMBA_USER:$MAMBA_USER blueprints/focaloop_bait/blueprint.focaloop.yml /opt/focaloop/blueprints/focaloop_bait/
ENV FOCALOOP_BLUEPRINT_PATH=/opt/focaloop/blueprints \
    PATH=/opt/conda/bin:$PATH

# Smoke test at build time: CLI + every bundled blueprint + every aligner binary resolves.
RUN focaloop --version \
    && focaloop validate panexpr malariavar focaloop_query focaloop_bait \
    && (bwa 2>&1 | grep -q '^Version: 0.7') && bwa-mem2 version && strobealign --version && minimap2 --version && bowtie2 --version \
    && salmon --version && piscem --version && samtools --version && bcftools --version

WORKDIR /data
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh"]
CMD ["focaloop", "--help"]
