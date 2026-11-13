# ghcr.io/galaxyproject/focaloop:0.2.0
#
# Build from the repository root:
#   docker build -f container/Dockerfile -t ghcr.io/galaxyproject/focaloop:0.2.0 .
# Apptainer (Galaxy job runners on garage / cluster):
#   apptainer build focaloop_proto_0.2.0.sif docker-daemon://ghcr.io/galaxyproject/focaloop:0.2.0
FROM mambaorg/micromamba:2.0.5-ubuntu24.04

LABEL org.opencontainers.image.title="focaloop" \
      org.opencontainers.image.version="0.2.0" \
      org.opencontainers.image.description="Adaptive RNA-seq (PanExpr) and Plasmodium variation (MalariaVar) focal loops for Galaxy" \
      org.opencontainers.image.source="https://github.com/galaxyproject/focaloop" \
      org.opencontainers.image.licenses="MIT"

USER root
RUN apt-get update && apt-get install -y --no-install-recommends iproute2 ca-certificates gcc libc6-dev libgomp1 && rm -rf /var/lib/apt/lists/*
USER $MAMBA_USER

COPY --chown=$MAMBA_USER:$MAMBA_USER container/environment.yml /tmp/environment.yml
RUN micromamba install -y -n base -f /tmp/environment.yml \
    && micromamba clean --all --yes

ARG MAMBA_DOCKERFILE_ACTIVATE=1
COPY --chown=$MAMBA_USER:$MAMBA_USER pyproject.toml README.md /opt/focaloop/src/
COPY --chown=$MAMBA_USER:$MAMBA_USER focaloop /opt/focaloop/src/focaloop
RUN mkdir -p /opt/focaloop/native_cache \
    && python -m pip install --no-deps --no-cache-dir /opt/focaloop/src \
    && FOCALOOP_PORTABLE_ISA=1 FOCALOOP_BUILD_NATIVE_DIR=/opt/focaloop/native_cache \
       python -c "from focaloop.galaxy_runners import prebuild_all_native_libs; print(prebuild_all_native_libs())"

# Builtin blueprints: `focaloop run --blueprint panexpr` resolves /opt/focaloop/blueprints/panexpr/
COPY --chown=$MAMBA_USER:$MAMBA_USER blueprints/panexpr/blueprint.focaloop.yml /opt/focaloop/blueprints/panexpr/
COPY --chown=$MAMBA_USER:$MAMBA_USER blueprints/malariavar/blueprint.focaloop.yml /opt/focaloop/blueprints/malariavar/
COPY --chown=$MAMBA_USER:$MAMBA_USER blueprints/focaloop_query/blueprint.focaloop.yml /opt/focaloop/blueprints/focaloop_query/
COPY --chown=$MAMBA_USER:$MAMBA_USER blueprints/focaloop_bait/blueprint.focaloop.yml /opt/focaloop/blueprints/focaloop_bait/
ENV FOCALOOP_BLUEPRINT_PATH=/opt/focaloop/blueprints \
    FOCALOOP_NATIVE_DIR=/opt/focaloop/native_cache \
    PATH=/opt/conda/bin:$PATH

# Smoke test at build time: CLI + every bundled blueprint + native C libraries + aligner binaries.
RUN focaloop --version \
    && focaloop validate panexpr malariavar focaloop_query focaloop_bait \
    && (bwa 2>&1 | grep -q '^Version: 0.7') && bwa-mem2 version && strobealign --version \
    && minimap2 --version && bowtie2 --version && hisat2 --version \
    && salmon --version && piscem --version && samtools --version && bcftools --version

WORKDIR /data
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh"]
CMD ["focaloop", "--help"]
