# ghcr.io/nekrut/focaloop:0.4.0
#
# Standalone multi-stage Docker build (no host-staged files required):
#   docker build -f container/Dockerfile -t ghcr.io/nekrut/focaloop:0.4.0 .
# Convert to Apptainer/Singularity SIF for CVMFS/Galaxy:
#   apptainer build focaloop_0.4.0.sif docker-daemon://ghcr.io/nekrut/focaloop:0.4.0
# Or build the SIF directly from container/Apptainer.def:
#   apptainer build focaloop_0.4.0.sif container/Apptainer.def

FROM ubuntu:24.04 AS xsra_builder

ARG XSRA_VERSION=0.2.29
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git build-essential cmake pkg-config \
        clang libclang-dev flex bison libssl-dev zlib1g-dev libbz2-dev \
    && rm -rf /var/lib/apt/lists/* \
    && curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable \
    && . "$HOME/.cargo/env" \
    && cargo install --locked xsra --version "${XSRA_VERSION}" --root /usr/local \
    && /usr/local/bin/xsra --version

FROM mambaorg/micromamba:2.0.5-ubuntu24.04

LABEL org.opencontainers.image.title="focaloop" \
      org.opencontainers.image.version="0.4.0" \
      org.opencontainers.image.description="Adaptive streaming genomics tools for ExpressionLoop, VariationLoop, GeneLoop, and Focaloop Query" \
      org.opencontainers.image.source="https://github.com/nekrut/focaloop" \
      org.opencontainers.image.licenses="MIT"

ARG THEMISTO_VERSION=3.2.2
ARG TIBERIUS_TAG=v2.0.9

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
        iproute2 ca-certificates curl git gcc make libc6-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /opt/Tiberius /opt/themisto /opt/focaloop/native_cache /opt/focaloop/tiberius_weights /data \
    && chown -R $MAMBA_USER:$MAMBA_USER /opt/Tiberius /opt/themisto /opt/focaloop /data

COPY --from=xsra_builder /usr/local/bin/xsra /usr/local/bin/xsra
RUN chmod 0755 /usr/local/bin/xsra

USER $MAMBA_USER

COPY --chown=$MAMBA_USER:$MAMBA_USER container/environment.yml /tmp/environment.yml
RUN micromamba install -y -n base -f /tmp/environment.yml \
    && micromamba clean --all --yes \
    && rm -f /tmp/environment.yml

ARG MAMBA_DOCKERFILE_ACTIVATE=1

# Install prebuilt Themisto v3.2.2 Linux binary
RUN curl -sSL --fail --retry 5 \
        "https://github.com/algbio/themisto/releases/download/v${THEMISTO_VERSION}/themisto_linux-v${THEMISTO_VERSION}.tar.gz" \
        -o /tmp/themisto.tar.gz \
    && tar -xzf /tmp/themisto.tar.gz -C /tmp \
    && cp -f "/tmp/themisto_linux-v${THEMISTO_VERSION}/themisto" /opt/themisto/themisto \
    && chmod 0755 /opt/themisto/themisto \
    && printf '#!/bin/sh\nLD_LIBRARY_PATH=/opt/conda/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} exec /opt/themisto/themisto "$@"\n' > /opt/conda/bin/themisto \
    && chmod 0755 /opt/conda/bin/themisto \
    && rm -rf /tmp/themisto.tar.gz "/tmp/themisto_linux-v${THEMISTO_VERSION}"

# Install CPU-only TensorFlow + Tiberius v2.0.9 runtime
ENV TF_USE_LEGACY_KERAS=0
RUN python -m pip install --no-cache-dir --no-deps "bricks2marble==0.1.1" "hidten==1.0.0" \
    && python -m pip install --no-cache-dir "tensorflow-cpu>=2.17,<2.21" "keras==3.15.1" "rich==13.9.4" "pydantic>=2.0" \
    && git clone --depth 1 --branch "${TIBERIUS_TAG}" https://github.com/Gaius-Augustus/Tiberius.git /opt/Tiberius \
    && (cd /opt/Tiberius && python -m pip install --no-deps --no-cache-dir .) \
    && chmod +x /opt/Tiberius/tiberius.py /opt/Tiberius/tiberius/*.py

# Bundle all 7 official Tiberius v2 clade weight packs (~195 MB total) so GeneLoop works offline
RUN set -eu; \
    unpack_weights() { \
        url="$1"; dest="$2"; \
        mkdir -p /tmp/tw_stage; rm -rf /tmp/tw_stage/*; \
        curl -sSL --fail --retry 5 "$url" -o /tmp/tw.tar.gz; \
        tar -xzf /tmp/tw.tar.gz -C /tmp/tw_stage; \
        src_dir="$(find /tmp/tw_stage -maxdepth 3 \( -name weights.h5 -o -name model_config.json \) -printf '%h\n' | head -n 1)"; \
        test -n "$src_dir"; \
        rm -rf "/opt/focaloop/tiberius_weights/$dest"; \
        mv "$src_dir" "/opt/focaloop/tiberius_weights/$dest"; \
        rm -rf /tmp/tw.tar.gz /tmp/tw_stage; \
    }; \
    unpack_weights "https://bioinf.uni-greifswald.de/bioinf/tiberius/models/tiberius_nosm_weights_v2.tar.gz" "tiberius_nosm_weights_v2"; \
    unpack_weights "https://bioinf.uni-greifswald.de/bioinf/tiberius/models/vertebrates_weights.tar.gz" "vertebrates_weights"; \
    unpack_weights "https://bioinf.uni-greifswald.de/bioinf/tiberius/models/insecta_unmasked_v2.tar.gz" "insecta_unmasked_v2"; \
    unpack_weights "https://bioinf.uni-greifswald.de/bioinf/tiberius/models/angiosperms.tar.gz" "angiosperms_weights"; \
    unpack_weights "https://bioinf.uni-greifswald.de/bioinf/tiberius/models/fungi.tar.gz" "fungi_weights"; \
    unpack_weights "https://bioinf.uni-greifswald.de/bioinf/tiberius/models/chlorophyta.tar.gz" "chlorophyta_weights"; \
    unpack_weights "https://bioinf.uni-greifswald.de/bioinf/tiberius/models/diatoms_unmasked.tar.gz" "diatoms_weights"

# Install Focaloop 0.4.0 and precompile all portable (-msse4.2 -mpopcnt) C99/OpenMP binaries & libs
COPY --chown=$MAMBA_USER:$MAMBA_USER pyproject.toml README.md Makefile /opt/focaloop/src/
COPY --chown=$MAMBA_USER:$MAMBA_USER scripts /opt/focaloop/src/scripts
COPY --chown=$MAMBA_USER:$MAMBA_USER focaloop /opt/focaloop/src/focaloop
RUN (cd /opt/focaloop/src && FOCALOOP_PORTABLE_ISA=1 make install-native PREFIX=/opt/conda PYTHON=python) \
    && python -m pip install --no-deps --no-cache-dir /opt/focaloop/src \
    && FOCALOOP_PORTABLE_ISA=1 \
       FOCALOOP_BUILD_NATIVE_DIR=/opt/focaloop/native_cache \
       FOCALOOP_NATIVE_CACHE=/opt/focaloop/native_cache \
       FOCALOOP_BIN_CACHE=/opt/focaloop/native_cache \
       python -c "from focaloop.galaxy_runners import prebuild_all_native_libs; libs = prebuild_all_native_libs(); print('Native libs:', len(libs), sorted(libs))"

# Built-in blueprints
COPY --chown=$MAMBA_USER:$MAMBA_USER blueprints/panexpr/blueprint.focaloop.yml /opt/focaloop/blueprints/panexpr/
COPY --chown=$MAMBA_USER:$MAMBA_USER blueprints/malariavar/blueprint.focaloop.yml /opt/focaloop/blueprints/malariavar/
COPY --chown=$MAMBA_USER:$MAMBA_USER blueprints/geneloop/blueprint.focaloop.yml /opt/focaloop/blueprints/geneloop/
COPY --chown=$MAMBA_USER:$MAMBA_USER blueprints/focaloop_query/blueprint.focaloop.yml /opt/focaloop/blueprints/focaloop_query/
COPY --chown=$MAMBA_USER:$MAMBA_USER blueprints/focaloop_bait/blueprint.focaloop.yml /opt/focaloop/blueprints/focaloop_bait/

ENV FOCALOOP_BLUEPRINT_PATH=/opt/focaloop/blueprints \
    FOCALOOP_NATIVE_DIR=/opt/focaloop/native_cache \
    FOCALOOP_NATIVE_CACHE=/opt/focaloop/native_cache \
    FOCALOOP_BIN_CACHE=/opt/focaloop/native_cache \
    FOCALOOP_PORTABLE_ISA=1 \
    TIBERIUS_WEIGHTS_DIR=/opt/focaloop/tiberius_weights \
    TF_USE_LEGACY_KERAS=0 \
    PANEXPR_DESEQ2_CPUS=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/opt/focaloop/native_cache:/opt/Tiberius:/opt/Tiberius/tiberius:/opt/conda/bin:$PATH

# Build-time smoke verification: CLI, blueprints, native C tiling, Tiberius + all 7 weights, aligners & tools
RUN focaloop --version \
    && focaloop validate panexpr malariavar geneloop focaloop_query focaloop_bait \
    && xsra --version \
    && tiberius.py --help >/dev/null \
    && python -c "import subprocess; \
from focaloop.geneloop.seed import detect_tiberius_runtime, discover_tiberius_weights; \
from focaloop.malariavar.logan_var import ensure_var_logan_bins; \
rt = detect_tiberius_runtime(); assert rt['backend'] == 'native', rt; \
models = ('mammalia_nosofttmasking_v2','vertebrates','insecta','angiosperms','fungi','chlorophyta','diatoms'); \
found = {m: discover_tiberius_weights(m) for m in models}; \
assert all(found.values()), found; \
prep, pile = ensure_var_logan_bins(); \
r = subprocess.run([str(prep),'1.5','0','-','0','0','3.0'], input='>0 ka:f:12.0'+chr(10)+('ACGT'*8500)+chr(10), capture_output=True, text=True, check=True); \
tiles = [ln for ln in r.stdout.splitlines() if ln.startswith('>')]; \
assert len(tiles) >= 16 and 'c' in tiles[0], tiles[:3]; \
print('FOCALOOP_040_DOCKER_VERIFIED_OK', len(found), prep, len(tiles))" \
    && ((bwa 2>&1 || true) | grep -q '^Version: 0.7') && bwa-mem2 version && strobealign --version \
    && minimap2 --version && bowtie2 --version && hisat2 --version \
    && salmon --version && piscem --version && kallisto version \
    && samtools --version && bcftools --version \
    && miniprot --version && diamond version && stringtie --version && compleasm -h >/dev/null \
    && test -x /opt/themisto/themisto && test -x /opt/conda/bin/themisto

WORKDIR /data
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh"]
CMD ["focaloop", "--help"]
