#!/usr/bin/env bash
# Install a dedicated Galaxy 25.0 server for focaloop.
#   GALAXY_ROOT=/srv/galaxy-focaloop galaxy/setup_galaxy25.sh
#   GALAXY_ROOT=$HOME/galaxy-focaloop galaxy/setup_galaxy25.sh
# Requires: git, python3 >= 3.10, apptainer >= 1.2, outbound HTTPS.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FOCALOOP_VERSION="$(python3 -c 'import sys, xml.etree.ElementTree as ET
root = ET.parse(sys.argv[1]).getroot()
print(next(t.text.strip() for t in root.findall("token") if t.attrib.get("name") == "@TOOL_VERSION@"))
' "$REPO/tools/focaloop/macros.xml")"

GALAXY_ROOT="${GALAXY_ROOT:-/srv/galaxy-focaloop}"
GALAXY_RELEASE="${GALAXY_RELEASE:-release_25.0}"
FOCALOOP_IMAGE="${FOCALOOP_IMAGE:-docker://ghcr.io/nekrut/focaloop:${FOCALOOP_VERSION}}"

command -v apptainer >/dev/null || { echo "apptainer not found" >&2; exit 1; }
mkdir -p "$GALAXY_ROOT"/{config,data/objects,data/tmp,data/jobs,data/tool-data,data/containers/singularity/mulled,images}

if [ ! -d "$GALAXY_ROOT/server/.git" ]; then
    git clone --depth 1 --branch "$GALAXY_RELEASE" https://github.com/galaxyproject/galaxy.git "$GALAXY_ROOT/server"
fi

secret="$(python3 -c 'import secrets; print(secrets.token_hex(16))')"
sed -e "s#__GALAXY_ROOT__#$GALAXY_ROOT#g" -e "s#__ID_SECRET__#$secret#g" \
    "$REPO/galaxy/galaxy_25_focaloop.yml" > "$GALAXY_ROOT/config/galaxy.yml"
sed -e "s#__FOCALOOP_REPO__#$REPO#g" "$REPO/galaxy/focaloop_tool_conf.xml" > "$GALAXY_ROOT/config/focaloop_tool_conf.xml"

# Pre-pull the focaloop image so the first job does not pay the pull.
sif="$GALAXY_ROOT/images/focaloop_${FOCALOOP_VERSION}.sif"
[ -f "$sif" ] || apptainer pull "$sif" "$FOCALOOP_IMAGE"
apptainer exec "$sif" focaloop validate panexpr malariavar geneloop focaloop_query focaloop_bait

# Outbound Range check (tier 3, ENA): expect HTTP 206.
code="$(curl -s -o /dev/null -w '%{http_code}' -r 0-1023 \
    https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR000/SRR000001/SRR000001.fastq.gz || true)"
echo "ENA range probe: HTTP $code (206 expected)"

cd "$GALAXY_ROOT/server"
GALAXY_CONFIG_FILE="$GALAXY_ROOT/config/galaxy.yml" sh scripts/common_startup.sh --skip-client-build
echo "Start:  GALAXY_CONFIG_FILE=$GALAXY_ROOT/config/galaxy.yml $GALAXY_ROOT/server/.venv/bin/galaxyctl start"
