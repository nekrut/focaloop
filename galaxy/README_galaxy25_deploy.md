# Dedicated Galaxy 25.0+ server for focaloop

This directory holds a ready-to-run configuration for a Galaxy `25.0+` instance that serves the four
focaloop blueprints both ways: as ToolShed-style XML tools in a **Focaloop** tool panel section, and as
unprivileged user-defined tools (UDTs) that users register from `blueprints/*/galaxy_udt.yml`.

| File | Purpose |
|------|---------|
| `galaxy_25_focaloop.yml` | `galaxy.yml` template: gravity, SQLite, UDTs on, Apptainer-only job environment |
| `focaloop_tool_conf.xml` | Tool panel section pointing at `blueprints/*/galaxy_tool.xml` in this checkout |
| `setup_galaxy25.sh` | Clones Galaxy `release_25.0`, renders the config, pulls the focaloop image, probes HTTP 206 |

## Targets

| Host | `GALAXY_ROOT` | Notes |
|------|---------------|-------|
| garage | `/media/anton/hd2/galaxy-focaloop` | default; data, jobs and the `.sif` live on `hd2` |
| cluster | `$HOME/galaxy-focaloop` (or scratch) | same script; set `GALAXY_ROOT` |

## Steps

1. Build and push the image (from the repository root), or build the `.sif` locally:
   ```bash
   docker build -f container/Dockerfile -t ghcr.io/galaxyproject/focaloop:0.1.0 .
   docker push ghcr.io/galaxyproject/focaloop:0.1.0          # needs push rights to the org
   ```
   Without push rights, build the `.sif` on the host and skip the pull:
   `apptainer build $GALAXY_ROOT/images/focaloop_0.1.0.sif docker-daemon://ghcr.io/galaxyproject/focaloop:0.1.0`.
2. Install: `GALAXY_ROOT=/media/anton/hd2/galaxy-focaloop galaxy/setup_galaxy25.sh`.
3. Start: `GALAXY_CONFIG_FILE=$GALAXY_ROOT/config/galaxy.yml $GALAXY_ROOT/server/.venv/bin/galaxyctl start`
   (port `8080`).
4. Register `anton@nekrut.org` (the configured admin), then in **Admin → Roles** create or open the
   role that grants custom tool execution (named "Custom Tool Execution" in the Galaxy 25 UDT docs)
   and add the users allowed to create UDTs. Confirm against
   <https://docs.galaxyproject.org/en/master/admin/user_defined_tools.html> for the exact release.
5. Register the UDTs as a user (any of):
   - Galaxy MCP: `create_user_tool` with the parsed `blueprints/<id>/galaxy_udt.yml`, then `run_user_tool`;
   - REST: `POST /api/unprivileged_tools` with the parsed YAML as the tool representation (see the UDT docs for the payload).

## Design choices

- **Containers only.** `require_container: true` and `conda_auto_install: false`: every job runs in
  `ghcr.io/galaxyproject/focaloop:0.1.0` via Apptainer, which is also what a UDT demands.
- **Outbound Range access.** Apptainer shares the host network namespace, so `focaloop` can issue
  HTTPS `Range` requests to ENA (tier 3), `sra-pub-run-odp` on S3 (tier 2) and remote CRAM/BAM hosts
  (tier 1). `setup_galaxy25.sh` probes ENA and prints the HTTP status (expect `206`).
- **No TPV.** All tools, including `tool_type_user_defined`, land in `apptainer_local`. If TPV is added
  later, route user-defined tools explicitly; the default TPV rules reject them with HTTP 403.
- **Builtin blueprints.** The image sets `FOCALOOP_BLUEPRINT_PATH=/opt/focaloop/blueprints`, so UDTs call
  `focaloop run --blueprint panexpr`; XML tools pass `$__tool_directory__/blueprint.focaloop.yml`.

## Not done here

Nothing in this directory has been executed against a live server: the image is not yet published,
and bringing up a server on garage or cluster waits for sign-off.
